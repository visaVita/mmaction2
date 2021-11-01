# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from ...core import top_k_accuracy
from ..builder import build_loss
import torch.nn.functional as F


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``forward``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0,
                 topk=(3, 5)):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps

        if self.multi_class:
            if topk is None:
                self.topk = ()
            elif isinstance(topk, int):
                self.topk = (topk, )
            elif isinstance(topk, tuple):
                assert all([isinstance(k, int) for k in topk])
                self.topk = topk
            else:
                raise TypeError('topk should be int or tuple[int], '
                                f'but get {type(topk)}')
            # Class 0 is ignored when calculaing multilabel accuracy,
            # so topk cannot be equal to num_classes
            assert all([k < num_classes for k in self.topk])

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    @staticmethod
    def recall_prec(pred_vec, target_vec):
        """
        Args:
            pred_vec (tensor[N x C]): each element is either 0 or 1
            target_vec (tensor[N x C]): each element is either 0 or 1

        """
        """ import ipdb
        ipdb.set_trace() """
        correct = pred_vec & target_vec
        # Seems torch 1.5 has no auto type conversion
        recall = correct.sum(1) / (target_vec.sum(1) + 1e-6)
        prec = correct.sum(1) / (pred_vec.sum(1) + 1e-6)
        return recall.mean(), prec.mean()

    def multi_label_accuracy(self, pred, target, thr=0.5):
        pred = pred.sigmoid()
        pred_vec = pred > thr
        # Target is 0 or 1, so using 0.5 as the borderline is OK
        target_vec = target > 0.5
        # recall_thr, prec_thr = self.recall_prec(pred_vec, target_vec)

        recalls, precs = [], []
        for k in self.topk:
            _, pred_label = pred.topk(k, 1, True, True)
            pred_vec = pred.new_full(pred.size(), 0, dtype=torch.bool)

            num_sample = pred.shape[0]
            for i in range(num_sample):
                pred_vec[i, pred_label[i]] = 1
            recall_k, prec_k = self.recall_prec(pred_vec, target_vec)
            recalls.append(recall_k)
            precs.append(prec_k)
        return recalls, precs

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)
            loss = self.loss_cls(cls_score, labels, **kwargs)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        if self.multi_class and cls_score is not None:
            # Only use the cls_score
            
            loss = self.loss_cls(cls_score, labels, **kwargs)
            #======================================================
            """ bce_loss = F.binary_cross_entropy_with_logits

            loss = bce_loss(cls_score, labels, reduction='none')
            
            pt = torch.exp(-loss)
            F_loss = self.focal_alpha * (1 - pt)**self.focal_gamma * loss
            loss = torch.mean(F_loss) """
            #==========================================================
            # multilabel_loss = F.multilabel_soft_margin_loss
            # loss = multilabel_loss(cls_score, labels)
            # =========================================================
            # losses['loss_action_cls'] = loss_cls

            recall_k, prec_k = self.multi_label_accuracy(
                cls_score, labels, thr=0.5)
            
            for i, k in enumerate(self.topk):
                losses[f'recall@top{k}'] = recall_k[i]
                losses[f'prec@top{k}'] = prec_k[i]

        # loss_cls may be dictionary or single tensor
        if isinstance(loss, dict):
            losses.update(loss)
        else:
            losses['loss_cls'] = loss

        return losses
