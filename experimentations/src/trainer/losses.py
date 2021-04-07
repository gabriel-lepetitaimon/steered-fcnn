import torch

def binary_dice_loss(pred, target):
    eps = 1e-6
    dice = 2*torch.sum(pred*target)/(torch.sum(pred+target)+eps)
    return 1-dice