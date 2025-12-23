import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES


@LOSSES.register_module(force=True)
class TeethSingleHeadLoss(nn.Module):
    """Loss for single head teeth caries segmentation.
    
    Directly segments caries without conditioning on teeth.
    Classes: 0 (background), 1 (suspect), 2 (confirmed)
    
    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss item.
        use_dice (bool): Whether to use DiceCE loss. Defaults to False (FocalTversky+CE).
    """
    def __init__(self, 
                 loss_weight=1.0, 
                 loss_name='teeth_single_head_loss',
                 use_dice=False):
        super(TeethSingleHeadLoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.use_dice = use_dice
        
        # 損失の重み
        self.tv_weight = 1.0
        self.dice_weight = 1.2
        self.ce_weight = 0.5

    @property
    def loss_name(self):
        return self._loss_name

    def forward(self, pred, label, weight=None, avg_factor=None, 
                reduction_override=None, ignore_index=-100, **kwargs):
        """Forward function."""
        loss = self.loss_weight * self.calculate_loss(pred, label)
        return loss

    def calculate_loss(self, pred, label):
        """Calculate loss."""
        # pred: [batch, 3, H, W] - 3 classes (background, suspect, confirmed)
        # label: [batch, 1, H, W] or [batch, 3, H, W]
        #   - If [batch, 1, H, W]: class indices from PackSegInputs, need to convert to one-hot
        #   - If [batch, 3, H, W]: already one-hot encoded (backward compatibility)

        # Convert class indices to one-hot if necessary
        if label.shape[1] == 1:
            # label is class indices (N, 1, H, W), convert to one-hot (N, C, H, W)
            num_classes = pred.shape[1]
            label = label.squeeze(1).long()  # (N, H, W)
            label = F.one_hot(label, num_classes=num_classes)  # (N, H, W, C)

            # AMPのため
            label = label.permute(0, 3, 1, 2).to(pred.dtype)  # (N, C, H, W)
        else:
            # AMPのため
            label = label.to(pred.dtype)

        # Softmax to get probabilities
        pred = F.softmax(pred, dim=1)
        
        if self.use_dice:
            # DiceCE Loss
            dice_loss = self._calculate_dice_loss(pred, label)
            ce_loss = self._calculate_ce_loss(pred, label)
            loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        else:
            # FocalTversky + CE Loss
            focal_tversky_loss = self._calculate_focal_tversky_loss(pred, label)
            ce_loss = self._calculate_ce_loss(pred, label)
            loss = self.tv_weight * focal_tversky_loss + self.ce_weight * ce_loss
            
        return loss

    def _calculate_dice_loss(self, pred, gt):
        """Calculate Dice loss."""
        # フォアグラウンド（confirmed + suspect）のマスクを作成
        fg_mask = torch.any(gt[:, 1:, :, :] > 0.5, dim=1).to(torch.float32)
        
        # Dice損失を計算（フォアグラウンドクラスのみ）
        d_num = 2.0 * pred[:, 1:, :, :] * gt[:, 1:, :, :]
        d_deno = pred[:, 1:, :, :] + gt[:, 1:, :, :]
        
        mask_expanded = fg_mask.unsqueeze(1)
        eps = 1e-5
        dice = 1 - (mask_expanded * d_num).sum((2, 3)) / ((mask_expanded * d_deno).sum((2, 3)) + eps)
        
        return torch.mean(dice)

    def _calculate_ce_loss(self, pred, gt, weight_map=None):
        """Calculate cross entropy loss."""

        # クラス数に応じて自動生成
        num_classes = pred.shape[1]
        ce_weight = torch.ones(num_classes, device=pred.device)
        ce_weight[0] = 0.1  # background（低重み）
        if num_classes > 1:
          ce_weight[1:] = torch.ones(num_classes - 1, device=pred.device)
          # ce_weight[1:] = torch.linspace(2.0, 3.5, num_classes - 1)

        # クラスごとの重み
        # ce_weight = torch.tensor([0.1, 2.0, 3.0], device=pred.device)  # background, confirmed, suspect

        # Improve numerical stability - only clamp lower bound to avoid gradient issues
        # Upper bound clamping causes gradient to be zero for confident predictions
        eps = 1e-7
        logp = -torch.log(pred.clamp_min(eps))
        ce_pix = (logp * gt).sum(1)

        # クラス重みを適用
        w_map = ce_weight[gt.argmax(1)]
        ce_pix = ce_pix * w_map

        return ce_pix.mean()

    def _calculate_focal_tversky_loss(self, pred, gt, alpha=0.3, beta=0.7, gamma=0.75):
        """Calculate Focal Tversky loss."""
        eps = 1e-7

        # Create clamped version for numerical stability, but preserve original for gradient flow
        safe_pred = torch.clamp(pred, min=eps, max=1.0 - eps)

        # フォアグラウンドクラス（confirmed, suspect）のみで計算
        pos_mask = (gt > 0.5).to(torch.float32)

        # Use safe_pred for operations that could cause numerical issues
        pred_comp = torch.clamp(1.0 - safe_pred, min=eps, max=1.0)

        # Use original pred for gradient-critical operations
        rep_pred = pos_mask * (1.0 - torch.pow(pred_comp, 2)) + (1.0 - pos_mask) * torch.pow(safe_pred, 2)

        tv_num = rep_pred * gt
        tv_deno = rep_pred * gt + alpha * rep_pred * (1.0 - gt) + beta * (1.0 - rep_pred) * gt

        # Use proper epsilon in denominator
        tversky = 1.0 - (torch.sum(tv_num, dim=[2, 3])) / (torch.sum(tv_deno, dim=[2, 3]) + eps)

        # Clamp tversky before power operation to avoid NaN
        tversky = torch.clamp(tversky, min=0.0, max=1.0)
        tversky = torch.pow(tversky, gamma)

        # 背景クラスを無視
        tversky = tversky[:, 1:]  # confirmed, suspectのみ

        return torch.mean(tversky)
