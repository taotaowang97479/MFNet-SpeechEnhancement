# code from  https://github.com/ioyy900205/MFNet/issues/1

import torch
import torch.nn as nn
import torch.nn.functional as F


def down_sampling(ch_in):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_in * 2, kernel_size=2, stride=2, bias=True),
        )
    )


def up_sampling(ch_in):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_in * 2, 1, bias=False),  # [B, C*2, H, W]
            nn.PixelShuffle(2),  # [B, C/2, H*2, W*2]
        )
    )


class LayerNormChannel(nn.Module):
    """
    Metaformer Layer Norm : https://github.com/sail-sg/poolformer/blob/main/models/poolformer.py#L86
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))  # Learnable
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GLFB(nn.Module):
    def __init__(self, ch_in, dialation=1):
        super(GLFB, self).__init__()
        # [B, C, H, W]
        self.layernorm_1 = LayerNormChannel(ch_in)

        self.pw_1 = nn.Conv2d(ch_in, ch_in * 2, 1, 1, 0, bias=True)
        padding_num = (3 - 1) * dialation
        padding = (padding_num // 2, padding_num - padding_num // 2,
                   padding_num // 2, padding_num - padding_num // 2,)
        self.pad = nn.ZeroPad2d(padding)
        self.dw = nn.Conv2d(ch_in * 2, ch_in * 2, kernel_size=3, stride=1, padding=0, dilation=dialation,
                            groups=ch_in * 2, bias=True)
        self.gate_1 = SimpleGate()

        self.ch_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_in, ch_in, 1, 1, 0, bias=True),
        )

        self.pw_2 = nn.Conv2d(ch_in, ch_in, 1, 1, 0, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, ch_in, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, ch_in, 1, 1)), requires_grad=True)

        self.layernorm_2 = LayerNormChannel(ch_in)
        self.pw_3 = nn.Conv2d(ch_in, ch_in * 2, 1, 1, 0, bias=True)
        self.gate_2 = SimpleGate()
        self.pw_4 = nn.Conv2d(ch_in, ch_in, 1, 1, 0, bias=True)

    def forward(self, inp):
        # [B, C, H, W]
        # [B, 1, 256, T]

        x = self.layernorm_1(inp)  # Layer Norm
        x = self.pw_1(x)
        x = self.pad(x)  # Point Conv
        x = self.dw(x)  # DW Conv
        x = self.gate_1(x)  # Gate
        x = x * self.ch_attention(x)  # Channel Attention
        x = self.pw_2(x)  # Point Conv

        # y = inp + x                                 # Add
        y = inp + x * self.beta  # Add

        x = self.layernorm_2(y)  # Layer Norm
        x = self.pw_3(x)  # Point Conv
        x = self.gate_2(x)  # Gate
        x = self.pw_4(x)  # Point Conv

        # return x + y                                # Add
        return y + x * self.gamma  # Add


class NS(nn.Module):
    def __init__(self, model_ch=32, mid_glfb=6, down_glfb=[1, 8, 4], up_glfb=[1, 1, 1]):
        super(NS, self).__init__()
        # projection
        self.in_proj = nn.Conv2d(1, model_ch, 3, 1, 1)

        self.in_glfb = GLFB(model_ch, dialation=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        ch = model_ch
        for num in down_glfb:
            self.downs.append(down_sampling(ch))
            ch = ch * 2
            self.encoders.append(
                nn.Sequential(
                    *[GLFB(ch, dialation=2 ** (_ + 1)) for _ in range(num)]
                )
            )

        self.mid_down = down_sampling(ch)
        ch = ch * 2
        self.middle_blks = \
            nn.Sequential(
                *[GLFB(ch) for _ in range(mid_glfb)]
            )
        self.mid_up = up_sampling(ch)
        ch = ch // 2

        for num in up_glfb:
            self.decoders.append(
                nn.Sequential(
                    *[GLFB(ch) for _ in range(num)]
                )
            )
            self.ups.append(up_sampling(ch))
            ch = ch // 2

        assert ch == model_ch
        self.out_glfb = GLFB(model_ch)

        # projection
        self.out_proj = nn.Conv2d(model_ch, 1, 3, 1, 1)
        self.padder = 2 ** (len(self.encoders) + 1)

    def pad(self, x):
        _, _, t, f = x.shape
        T_pad = (self.padder - t % self.padder) % self.padder
        F_pad = (self.padder - f % self.padder) % self.padder
        x = F.pad(x, (0, F_pad, 0, T_pad))
        return x

    def forward(self, x):
        # x [B, C(1), H(320), W(T)]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # zero padding for down/up sampling
        B, C, H, W = x.shape
        # pad x

        x = self.pad(x)
        inp = x
        x = self.in_proj(x)
        encs = []
        x = self.in_glfb(x)
        encs.append(x)

        for encoder, down in zip(self.encoders, self.downs):
            x = down(x)
            x = encoder(x)
            encs.append(x)

        x = self.mid_down(x)
        x = self.middle_blks(x)
        x = self.mid_up(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = x + enc_skip
            x = decoder(x)
            x = up(x)

        x = x + encs[0]
        x = self.out_glfb(x)
        x = self.out_proj(x)  # [B, 1, H*, W*]

        x = x + inp
        return x[..., :H, :W]


if __name__ == "__main__":
    net = NS()
    input = torch.randn(1, 1, 99, 320)  # BCTF
    output = net(input)
    print('done')


    def cout_param(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(cout_param(net))

    from thop import clever_format
    from thop import profile

    input = torch.randn(1, 1, 99, 320)

    flops, params = profile(net, inputs=(input,))
    flops, params = clever_format([flops, params], "%.5f")

    print(flops, params)