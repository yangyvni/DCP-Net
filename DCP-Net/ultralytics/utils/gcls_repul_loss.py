import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics.utils.torch_utils import autocast

from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors, bbox2dist
from .metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


# GCls: Gaussian classification Loss
class GCls(nn.Module):
    def __init__(self, loss_fcn: nn.Module, sigma: float = 0.1):
        super().__init__()
        self.loss_fcn = loss_fcn # 保存基础损失
        self.sigma = sigma
        self.orig_reduction = getattr(loss_fcn, "reduction", "mean") # 保存原始损失归约方式
        
        self.loss_fcn.reduction = "none"
    
    def forward(self, pred, true, auto_iou: float = 0.5):
        if isinstance(auto_iou, torch.Tensor):
            auto_iou = float(auto_iou.mean().detach().cpu().item())
        sigma = max(self.sigma, 1e-4)
        base_loss = self.loss_fcn(pred, true)
        mu = auto_iou 
        weight = torch.exp(-((true - mu) ** 2) / (2 * sigma**2)) + 1.0 
        loss = base_loss * weight
        return loss.mean() if self.orig_reduction == "mean" else loss.sum()

# GRegloss: Gaussian Regression Loss
def GReg_loss(pred, target, eps=1e-6, xyxy=True, mode='l1'):

    if xyxy:
        pred = xyxy2xywh(pred)
        target = xyxy2xywh(target)

    px, py, pw, ph = pred[:, 0], pred[:, 1], pred[:, 2].clamp(min=eps), pred[:, 3].clamp(min=eps)
    gx, gy, gw, gh = target[:, 0], target[:, 1], target[:, 2].clamp(min=eps), target[:, 3].clamp(min=eps)

    a1, b1 = pw.pow(2) / 12, ph.pow(2) / 12
    a2, b2 = gw.pow(2) / 12, gh.pow(2) / 12

    dx2 = (px - gx).pow(2)
    dy2 = (py - gy).pow(2)

    # 3. KL/Bhattacharyya 
    denom = (a1 + a2) * (b1 + b2) + eps

    # t1/t2: position term
    t1 = 0.25 * (a1 + a2) * dy2 / denom
    t2 = 0.25 * (b1 + b2) * dx2 / denom

    # t3: shape term
    t3 = 0.5 * torch.log(denom / (4.0 * torch.sqrt(a1 * b1 * a2 * b2) + eps) + eps)

    # 4. Bhattacharyya距离 (KL近似形式)
    Bd = t1 + t2 + t3
    Bd = torch.clamp(Bd, eps, 100.0)

    # 5. 映射为概率形式的相似度和损失
    sim = torch.exp(-Bd)
    l1 = torch.sqrt(1.0 - sim + eps)
    l2 = -torch.log(1.0 - l1.pow(2) + eps)

    if mode == 'l1':
        return l1, sim
    else:
        return l2, sim

# Repulsion Loss
def repulsion_loss(pred_boxes, gt_boxes, pnms=0.1, gtnms=0.1):
    device = pred_boxes.device
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    pgiou = bbox_iou(pred_boxes, gt_boxes, xywh=False)
   
    ppiou = bbox_iou(pred_boxes, pred_boxes, xywh=False)
    ppiou = ppiou * (1 - torch.eye(ppiou.size(0), device=device))

    max_iou, argmax = pgiou.max(dim=1)
    mask = max_iou > gtnms
    repgt_loss = torch.tensor(0.0, device=device)
    if mask.any():
        pred_sel = pred_boxes[mask]
        gt_sel = gt_boxes[argmax[mask]]
        p_cx = (pred_sel[:, 0] + pred_sel[:, 2]) / 2
        p_cy = (pred_sel[:, 1] + pred_sel[:, 3]) / 2
        p_w = (pred_sel[:, 2] - pred_sel[:, 0]).abs()
        p_h = (pred_sel[:, 3] - pred_sel[:, 1]).abs()
        g_cx = (gt_sel[:, 0] + gt_sel[:, 2]) / 2
        g_cy = (gt_sel[:, 1] + gt_sel[:, 3]) / 2
        g_w = (gt_sel[:, 2] - gt_sel[:, 0]).abs()
        g_h = (gt_sel[:, 3] - gt_sel[:, 1]).abs()

        piou_loss, sim = GReg_loss(
            torch.stack([p_cx, p_cy, p_w, p_h], dim=1),
            torch.stack([g_cx, g_cy, g_w, g_h], dim=1),
            xyxy=False,
        )
   
        repgt_loss = smooth_ln(sim).mean()

    repbox_loss = torch.tensor(0.0, device=device)
    if (ppiou > pnms).any():
        repbox_loss = smooth_ln(ppiou[ppiou > pnms]).mean()
   
    return repgt_loss, repbox_loss

def smooth_ln(x, t=0.5):
    x = x.clamp(0.0, 1.0)
    return torch.where(
        x <= t,
        torch.log((1 + x) / (1 - x + 1e-7)),
        2 * (x - t) / (1 - t**2 + 1e-7) + np.log((1 + t) / (1 - t + 1e-7)),
    )

# DFLoss + BboxLoss
class DFLoss(nn.Module):
    def __init__(self, reg_max=16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # print(f"=============weight: {weight}")
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, DIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initialize FocalLoss class with no parameters."""
        super().__init__()


    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.5):
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
    
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()

# =========================================================
# GCRv8DetectionLoss
# =========================================================
class GCRv8DetectionLoss:
    def __init__(self, model, tal_topk: int = 10):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]

        self.device = device
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.use_dfl = m.reg_max > 1
        
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.fl = FocalLoss()
        self.GCls_fl = GCls(self.fl, sigma=0.2) #从0.1->0.2
        self.GCls_bce = GCls(self.bce, sigma=0.2) #从0.1->0.2

        self.bbox_loss = BboxLoss(m.reg_max).to(device)

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        self.gamma = 0.8
        self.eta = 0.2
    
    def preprocess(self, targets, batch_size, scale_tensor):
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)
        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        counts = counts.to(dtype=torch.int32)
        for j in range(batch_size):
            matches = i == j
            if n := matches.sum():
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            proj = torch.arange(self.reg_max, dtype=pred_dist.dtype, device=pred_dist.device)
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)

        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # GCls
        auto_iou = target_scores.mean() if isinstance(target_scores, torch.Tensor) else float(target_scores.mean())
        cls_loss_fl = self.GCls_fl(pred_scores, target_scores.to(dtype), auto_iou=auto_iou) # 自适应设置阈值，即目标分数的平均值
        cls_loss_bce = self.GCls_bce(pred_scores, target_scores.to(dtype), auto_iou=auto_iou)
        loss[1] = (cls_loss_fl + cls_loss_bce) / target_scores_sum

     
        # GReg + Repulsion + DFL
        if fg_mask.sum():
            target_bboxes = target_bboxes / stride_tensor

            # === IoU + DFL ===
            loss_iou, loss_dfl = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask
            )

            # === GReg + Repulsion ===
            repgt, repbox = repulsion_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            
            greg_loss, sim = GReg_loss(
                (pred_bboxes[fg_mask][:, [0, 1, 2, 3]] + 1e-6),
                (target_bboxes[fg_mask][:, [0, 1, 2, 3]] + 1e-6),
                xyxy=True,
            )
    
            at_loss = self.eta * greg_loss.mean() + self.gamma * loss_iou # attraction loss + IoU loss
            inter_loss =  repgt + repbox + at_loss # attraction loss + repulsion loss

            loss[0] = inter_loss
            loss[2] = loss_dfl

        # loss
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss * batch_size, loss.detach()
