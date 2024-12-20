import torch.nn as nn
import torch


class MSEDCTLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(MSEDCTLoss, self).__init__()
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()

    def forward(self, S_pred, S_true):
        """
        input: the compress DCT spectrum
        S_pred : [B,C,T,F]
        S_true : [B,C,T,F]
        """

        # Mean-Square Error (MSE) loss for absolute values
        Loss_abs = self.mse_loss(torch.abs(S_true), torch.abs(S_pred))

        # MSE loss for polar values
        Loss_polar = self.mse_loss(S_true, S_pred)
        
        # total loss
        Loss_MFNet = self.gamma * Loss_abs + (1 - self.gamma) * Loss_polar

        return Loss_MFNet