import torch
import torch_dct
import numpy as np


def frame(signal, frame_length, hop_length, window=None, device='cpu'):
    """
    将信号分帧，并可选择地应用窗口。

    参数:
    - signal (torch.Tensor): 输入信号.
    - frame_length (int): 每帧的长度.
    - hop_length (int): 帧之间的跳跃长度.
    - window (Optional[torch.Tensor]): 可选的窗口函数.

    返回:
    - torch.Tensor: 分帧后的张量.
    """
    frames = signal.unfold(-1, frame_length, hop_length)
    frames = frames.to(device)

    if window is not None:
        window = window.to(device)  # window在cuda上
        frames = frames * window

    return frames


def stdct(signal, frame_length, hop_length, window=None):
    frames = frame(signal, frame_length, hop_length, window)
    return torch_dct.dct(frames, norm=None)


def istdct(dct, frame_length, hop_length, window=None):
    frames = torch_dct.idct(dct, norm=None).squeeze(1)
    num_frames = frames.shape[-2]
    audio = torch.zeros((frames.shape[0], num_frames * hop_length + frame_length - hop_length), device=frames.device)
    overlap = torch.zeros((frames.shape[0], num_frames * hop_length + frame_length - hop_length), device=frames.device)
    if window is None:
        window = torch.ones(frame_length, device=frames.device)
    for i in range(num_frames):
        audio[:, i * hop_length : i * hop_length + frame_length] += frames[:, i, :] * window
        overlap[:, i * hop_length : i * hop_length + frame_length] += window ** 2
    overlap = overlap.clamp(min=1e-8)
    audio /= overlap
    return audio


if __name__ == '__main__':
    duration = 4.0  # seconds
    sample_rate = 16000
    t = torch.arange(0, duration, 1.0 / sample_rate)
    frequency = 440.0
    wav = 0.5 * torch.sin(2 * np.pi * frequency * t)

    wav = wav.unsqueeze(0)
    window = torch.sqrt(torch.hann_window(320))
    wav_dct = stdct(wav, 320, 160, window=window)
    wav_idct = istdct(wav_dct, 320, 160, window=window)

    wav = wav[:, :wav_idct.shape[-1]]
    err = (wav - wav_idct).sum() / len(wav)
    print(err)
