import torch
from torch.nn.modules.loss import _Loss
# from monai.losses import DiceLoss
from typing import Any, Callable, List, Optional, Sequence, Union, Dict
import torch.nn as nn

class KLLoss(_Loss):
    """
    Compute two losses: Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of DiceCE loss is shown in ``monai.losses.DiceCELoss``.
    """

    def __init__(
        self,
        alpha: int, 
        beta: int,
        weight: torch.Tensor = None
    ) -> None:
        
        super().__init__()
        
        self.bce_loss = torch.nn.BCELoss(
            weight,
            reduction='mean'
        )
        self.dice_loss = Dice_Loss()
        self.beta = beta
        self.alpha = alpha
        

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        # print("heree")
        
        if input.shape != target.shape:

            raise ValueError("the number of dimensions for input and target should be the same.")
        
        # VAE adds sigmoid to its output already
        input = input.float()
        target = target.float()
        # print("input min", torch.min(input), "max", torch.max(input))
        # print("target min", torch.min(target), "max", torch.max(target))
        
        dice =  self.dice_loss(input, target)
        bce = self.bce_loss(input, target)
        kld = -0.5* torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        # print("bce",bce, "kld" , "dice", dice)
        return self.alpha*dice + bce + self.beta  *kld , kld
    
    def forward(self, model_output: tuple, target: torch.Tensor):
        
        input, mu, log_var, _ = model_output
        loss, avg_kl = self.compute_loss(input, target, mu, log_var)
        return loss, avg_kl

class Dice_Loss(nn.Module):
    '''
    Input shape format is (output of model for 2 classes): B, 2, Z, X, Y
    Target shape format is: B, Z, X, Y.
        
    Both have to become B, 1, Z, X, Y, like the shape of the input images.
    '''
    
    def __init__(self, output='softmax',  smooth=0.001):
        super(Dice_Loss, self).__init__()
        
        if output not in ['softmax', 'sigmoid']: 
            raise ValueError('Distance metric must be one of "softmax", "sigmoid".')
            
        self.smooth = smooth
        self.output = output


    
    def indiv_dice_loss(self, inputs, targets):
        
        # only do the following operations along the 1st dimension
        iflat = inputs.view(len(inputs), -1)#batchx 5
        tflat = targets.view(len(targets), -1)

        intersection = (iflat * tflat).sum(-1)#
        
        return 1- ((2. * intersection + self.smooth) / (iflat.sum(-1) + tflat.sum(-1) + self.smooth))
    
    
    def forward(self, inputs, targets, reduction = 'mean'): 

        return self.indiv_dice_loss(inputs, targets).mean()
    
class Dice(nn.Module):

    
    def __init__(self, output='softmax',  smooth=0.001):
        super(Dice, self).__init__()
        
        if output not in ['softmax', 'sigmoid']: 
            raise ValueError('Distance metric must be one of "softmax", "sigmoid".')
            
        self.smooth = smooth
        self.output = output


    
    def indiv_dice_loss(self, inputs, targets):
        
        # only do the following operations along the 1st dimension
        iflat = inputs.view(len(inputs), -1)#batchx 5
        tflat = targets.view(len(targets), -1)

        intersection = (iflat * tflat).sum(-1)#
        
        return ((2. * intersection + self.smooth) / (iflat.sum(-1) + tflat.sum(-1) + self.smooth))
    
    
    def forward(self, inputs, targets, reduction = 'mean'): 

        return self.indiv_dice_loss(inputs, targets).mean()