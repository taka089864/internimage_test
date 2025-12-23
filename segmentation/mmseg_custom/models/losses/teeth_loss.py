import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES


@LOSSES.register_module(force=True)
class TeethCELoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='teeth_ce_loss'):
        super(TeethCELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)

        # Note: for BCE loss, label < 0 is invalid.
        loss_cls = self.loss_weight * self.teeth_cross_entropy(cls_score, label, **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


    def teeth_cross_entropy(self, pred, label):
        loss_weight = 1.0
        
        pred1 = pred[0]
        pred2 = pred[1]
        gt1 = label[0]
        gt2 = label[1]

        fg_mask1 = torch.any(gt1>0.01,axis=1)
        fg_mask1 = fg_mask1.to(torch.float32)
        fg_cnt1 = torch.maximum(torch.sum(fg_mask1,dim=[1,2]),torch.tensor([0.1]).to(fg_mask1.device))
        
        fg_mask2 = torch.any(gt2>0.01,axis=1)
        fg_mask2 = fg_mask2.to(torch.float32)
        fg_cnt2 = torch.maximum(torch.sum(fg_mask2,dim=[1,2]),torch.tensor([0.1]).to(fg_mask2.device))

        ce1 = torch.sum(-torch.log(pred1+1e-8)*gt1,dim=1)
        ce2 = torch.sum(-torch.log(pred2+1e-8)*gt2,dim=1)

        loss1 = torch.sum(ce1*fg_mask1,dim=[1,2])/fg_cnt1
        loss2 = torch.sum(ce2*fg_mask2,dim=[1,2])/fg_cnt2

        loss = loss_weight * torch.mean(loss1) + torch.mean(loss2)
        
        return loss



@LOSSES.register_module(force=True)
class TeethFocalTverCELoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """
    def __init__(self, reduction='mean', loss_weight=1.0, loss_name='teeth_focal_tver_ce_loss'):
        super(TeethFocalTverCELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

        # 歯用のlossの重み
        self.dice_weight = 1.2

        self.ce1_weight = 0.7
        # self.ce1_weight = 0.4

        # う蝕用のlossの重み
        self.tv_weight = 1.0
        self.ce2_weight = 0.5
        self.out_weight = 0.2

        # total lossに関する重み
        self.loss1_weight = 0.5
        self.loss2_weight = 1.0


    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):

        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)

        # Note: for BCE loss, label < 0 is invalid.
        loss_cls = self.loss_weight * self.teeth_cross_entropy(cls_score, label, **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

    def _calculate_dice_loss(self, pred, gt, mask):
        """Dice損失を計算
        
        Args:
            pred: 予測値 (batch, channels, height, width)
            gt: Ground Truth (batch, channels, height, width)
            mask: フォアグラウンドマスク (batch, height, width)
        
        Returns:
            Dice損失
        """
        d_num = 2.0 * pred * gt
        d_deno = pred + gt
        
        mask_expanded = mask.unsqueeze(1)
        # dice = 1.0 - (torch.sum(mask_expanded * d_num, dim=[2, 3]) /
        #              torch.maximum(
        #                  torch.sum(mask_expanded * d_deno, dim=[2, 3]),
        #                  torch.tensor(1.0).to(mask.device))
        #                  # torch.tensor([1.0]).to(mask.device))
        #               )

        eps = 1e-5
        dice = 1 - (mask_expanded * d_num).sum((2, 3)) / ((mask_expanded * d_deno).sum((2, 3)) + eps)
        # dice = 1 - (mask_expanded * d_num).sum((2, 3)) / (mask_expanded * d_deno).sum((2, 3)).clamp(min=1.0)

        dice = torch.mean(dice)

        return dice

    def _calculate_ce_loss(self, pred, gt, mask=None, weight=None):
        logp = -torch.log(pred.clamp_min(1e-8))
        ce_pix = (logp * gt).sum(1)

        if weight is not None:
            w_map = weight[gt.argmax(1)]
            ce_pix = ce_pix * w_map

        if mask is None:
            return ce_pix.mean()
        else:
            ce_sum = (ce_pix * mask).sum((1,2))
            denom  = mask.sum((1,2)).clamp(min=0.1)
            return (ce_sum / denom).mean()

    def _calculate_focal_tversky_loss(self, pred, gt, mask, alpha=0.3, beta=0.7, gamma=0.75):
        """Focal Tversky損失を計算
        
        Args:
            pred: 予測値 (batch, channels, height, width)
            gt: Ground Truth (batch, channels, height, width)
            mask: 条件付きマスク (batch, height, width)
            alpha: False Positive重み
            beta: False Negative重み
            gamma: Focal重み
        
        Returns:
            Focal Tversky損失
        """
        pos_mask = (gt > 0.5).to(torch.float32)
        rep_pred = pos_mask * (1.0 - torch.pow(1.0 - pred, 2)) + (1.0 - pos_mask) * torch.pow(pred, 2)
        
        mask_expanded = mask.unsqueeze(1)
        tv_num = rep_pred * gt * mask_expanded
        tv_deno = (rep_pred * gt + alpha * rep_pred * (1.0 - gt) + 
                   beta * (1.0 - rep_pred) * gt) * mask_expanded
        
        tversky = 1.0 - (torch.sum(tv_num, dim=[2, 3])) / (torch.sum(tv_deno, dim=[2, 3]) + 1.0)
        tversky = torch.pow(tversky, gamma)
        
        # 背景クラスを無視
        tversky = tversky[:, 1:]
        
        return torch.mean(tversky)


    def teeth_cross_entropy(self, pred, label):
        pred1 = pred[0]
        pred2 = pred[1]
        gt1 = label[0]
        gt2 = label[1]

        ### teeth Class (DiceCE)
        # 背景と歯の両方を含む全体で損失を計算
        # マスクを使わず、全ピクセルで評価
        all_mask = torch.ones((gt1.shape[0], gt1.shape[2], gt1.shape[3]), device=gt1.device, dtype=torch.float32)

        # 歯の領域マスクを作成（歯である確率が50%以上の場所のみ）
        # gt1[:, 1, :, :] は歯のチャンネル（背景がchannel 0、歯がchannel 1）
        teeth_mask = (gt1[:, 1, :, :] > 0.5).to(torch.float32)

        # 歯だけでdiceを計算
        dice_loss1 = self._calculate_dice_loss(pred1[:, 1:], gt1[:, 1:], teeth_mask)
        # dice_loss1 = self._calculate_dice_loss(pred1, gt1, teeth_mask)

        ce_weight  = torch.tensor([0.2, 1.0], device=pred1.device)
        # ce_weight  = torch.tensor([0.1, 1.0], device=pred1.device)
        ce_loss1 = self._calculate_ce_loss(pred1, gt1, all_mask, ce_weight)

        loss1 = self.dice_weight * dice_loss1 + self.ce1_weight * ce_loss1

        ### Damage Class (FocalTverskyCE) - Conditional on Teeth P(caries|teeth)
        # う蝕のマスクを作成（う蝕チャンネル1,2のいずれかが0.01より大きい場所）
        # gt2[:, 0, :, :] は背景、gt2[:, 1:, :, :] はう蝕（suspect, confirmed）
        fg_mask2 = torch.any(gt2[:, 1:, :, :] > 0.01, dim=1)
        fg_mask2 = fg_mask2.to(torch.float32)

        # 条件付き確率：歯の領域内でのみう蝕を考慮
        # P(う蝕 ∧ 歯) の領域を作成
        fg_mask2_conditional = fg_mask2 * teeth_mask

        # P(う蝕, 歯) の予測を作成
        # pred1[:, 1:2, :, :] は歯の予測確率（channel 1）
        pred_teeth_prob = pred1[:, 1:2, :, :]  # pred2と同じサイズに拡張

        pred_joint = torch.zeros_like(pred2)
        pred_joint[:, 1:] = pred_teeth_prob * pred2[:, 1:]  # suspect, confirmed
        pred_joint[:, 0]  = (1 - pred_teeth_prob.squeeze(1)) + pred_teeth_prob.squeeze(1) * pred2[:, 0]

        # focal_tverskyはteeth_mask
        focal_tversky_loss2 = self._calculate_focal_tversky_loss(pred_joint, gt2, teeth_mask, alpha=0.2, beta=0.8, gamma=1.5)

        # cross entropyはall_maskで
        ce_weight2  = torch.tensor([0.05, 3, 2], device=pred1.device)
        ce_loss2 = self._calculate_ce_loss(pred_joint, gt2, all_mask, ce_weight2)

        # 歯のない場所にう蝕が発生した場合にペナルティを課す(fpを減らす)
        outside_loss = 0.2 * (pred_joint[:,1:].sum(1) * (1 - teeth_mask)).mean()
        # outside_loss = 0.2 * (pred2[:,1:].sum(1) * (1 - teeth_mask)).mean()

        loss2 = self.tv_weight * focal_tversky_loss2 + self.ce2_weight * ce_loss2 + self.out_weight * outside_loss

        # 総損失 = 歯の損失 + う蝕の損失（条件付き）
        loss = self.loss1_weight * loss1 + self.loss2_weight * loss2
        
        return loss

    def teeth_cross_entropy_backup(self, pred, label):
        loss_weight = 1.0

        pred1 = pred[0]
        pred2 = pred[1]
        gt1 = label[0]
        gt2 = label[1]

        fg_mask1 = torch.any(gt1>0.01, axis=1)
        fg_mask1 = fg_mask1.to(torch.float32)
        fg_cnt1 = torch.maximum(torch.sum(fg_mask1,dim=[1,2]),torch.tensor([0.1]).to(fg_mask1.device))

        d_num1 = 2.0 * pred1 * gt1
        d_deno1 = pred1 + gt1

        fg_mask1e = fg_mask1.unsqueeze(1)
        dice1 = 1.0 - torch.sum(fg_mask1e*d_num1,dim=[2,3])/torch.maximum(torch.sum(fg_mask1e*d_deno1,dim=[2,3]),torch.tensor([1.0]).to(fg_mask1.device))

        ce1 = torch.sum(-torch.log(pred1+1e-8)*gt1,dim=1)

        loss1 = torch.mean(dice1) + torch.mean(torch.sum(ce1*fg_mask1,dim=[1,2])/fg_cnt1)


        ### Damage Class (FocalTverskyCE)
        fg_mask2 = torch.any(gt2>0.01,axis=1)
        fg_mask2 = fg_mask2.to(torch.float32)
        fg_cnt2 = torch.maximum(torch.sum(fg_mask2,dim=[1,2]),torch.tensor([0.1]).to(fg_mask2.device))

        pos_mask = (gt2 > 0.5)
        pos_mask = pos_mask.to(torch.float32)
        rep_pred2 = pos_mask*(1.0-torch.pow(1.0-pred2,2)) + (1.0-pos_mask)*torch.pow(pred2,2)

        tv_num2 = rep_pred2*gt2
        tv_deno2 = rep_pred2*gt2 + 0.3*rep_pred2*(1.0-gt2) + 0.7*(1.0-rep_pred2)*gt2

        fg_mask2e = fg_mask2.unsqueeze(1)
        tver2 = 1.0 - (torch.sum(fg_mask2e*tv_num2,dim=[2,3]))/(torch.sum(fg_mask2e*tv_deno2,dim=[2,3])+1.0)
        tver2 = torch.pow(tver2,0.75)

        tver2 = tver2[:,1:] # ignore background class

        ce2 = torch.sum(-torch.log(pred2+1e-8)*gt2,dim=1)

        loss2 = torch.mean(tver2) +  torch.mean(torch.sum(ce2*fg_mask2,dim=[1,2])/fg_cnt2)


        loss = loss_weight * loss1 + loss2

        return loss

    def _calculate_ce_loss__(self, pred, gt, mask):
        """クロスエントロピー損失を計算

        Args:
            pred: 予測値 (batch, channels, height, width)
            gt: Ground Truth (batch, channels, height, width)
            mask: フォアグラウンドマスク (batch, height, width)

        Returns:
            CE損失
        """
        ce = torch.sum(-torch.log(pred + 1e-8) * gt, dim=1)
        fg_cnt = torch.maximum(torch.sum(mask, dim=[1,2]), torch.tensor([0.1]).to(mask.device))
        return torch.mean(torch.sum(ce * mask, dim=[1,2]) / fg_cnt)


@LOSSES.register_module(force=True)
class TeethSingleDiceCELoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='teeth_single_dice_ce_loss'):
        super(TeethSingleDiceCELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)

        # Note: for BCE loss, label < 0 is invalid.
        loss_cls = self.loss_weight * self.teeth_cross_entropy(cls_score,label,**kwargs)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


    def teeth_cross_entropy(self, pred, label):
        fg_mask = torch.any(label>0.01,axis=1)
        fg_mask = fg_mask.to(torch.float32)
        fg_cnt = torch.maximum(torch.sum(fg_mask,dim=[1,2]),torch.tensor([0.1]).to(fg_mask.device))
        
        d_num = 2.0*pred*label
        d_deno = pred + label

        fg_mask1e = fg_mask.unsqueeze(1)
        dice = 1.0 - torch.sum(fg_mask1e*d_num,dim=[2,3])/torch.maximum(torch.sum(fg_mask1e*d_deno,dim=[2,3]),torch.tensor([1.0]).to(fg_mask.device))

        ce = torch.sum(-torch.log(pred+1e-8)*label,dim=1)

        loss = torch.mean(dice) + torch.mean(torch.sum(ce*fg_mask,dim=[1,2])/fg_cnt)
        
        return loss


