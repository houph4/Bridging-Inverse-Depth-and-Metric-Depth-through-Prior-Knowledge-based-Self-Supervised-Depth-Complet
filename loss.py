import torch
import torch.nn as nn

def gradient_yx(T):
    '''
    Computes gradients in the y and x directions

    Arg(s):
        T : torch.Tensor[float32]
            N x C x H x W tensor
    Returns:
        torch.Tensor[float32] : gradients in y direction
        torch.Tensor[float32] : gradients in x direction
    '''

    dx = T[:, :, :, :-1] - T[:, :, :, 1:]
    dy = T[:, :, :-1, :] - T[:, :, 1:, :]
    return dy, dx

def smoothness_loss_func(predict, image, w=None):
    '''
    Computes the local smoothness loss

    Arg(s):
        predict : torch.Tensor[float32]
            N x 1 x H x W predictions
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    Returns:
        torch.Tensor[float32] : smoothness loss
    '''

    predict_dy, predict_dx = gradient_yx(predict)
    image_dy, image_dx = gradient_yx(image)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    if w is not None:
        weights_x = weights_x * w[:, :, :, :-1]
        weights_y = weights_y * w[:, :, :-1, :]

    smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))

    return smoothness_x + smoothness_y


class UcertRELossL1(nn.Module):
    def __init__(self):
        super(UcertRELossL1, self).__init__()

    def forward(self, pred, Ucert, target,rgb):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        Ucert = Ucert[valid_mask]+1
        self.lossl1 = (diff * Ucert).abs().mean()
        # self.lossl1 = diff.abs().mean()
        smooth = smoothness_loss_func(pred, rgb)
        self.lossl2 = ((diff * Ucert) ** 2).mean()
        return  self.lossl1+self.lossl2+0.01*smooth

class UcertRELossL11(nn.Module):
    def __init__(self):
        super(UcertRELossL11, self).__init__()

    def forward(self, pred, target,rgb):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        # Ucert = Ucert[valid_mask]+1
        # self.lossl1 = (diff * Ucert).abs().mean()
        self.lossl1 = diff.abs().mean()
        smooth = smoothness_loss_func(pred, rgb)
        # self.lossl2 = ((diff * Ucert) ** 2).mean()
        self.lossl2 = (diff ** 2).mean()
        return  self.lossl1+self.lossl2+0.01*smooth
