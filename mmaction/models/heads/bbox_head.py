# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.core.bbox import bbox_target
from ..common import PositionalEnconding, TranST
from mmcv.cnn import normal_init, kaiming_init
from ..builder import build_loss
import math
try:
    from mmdet.models.builder import HEADS as MMDET_HEADS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


class BBoxHeadAVA(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively.

    Args:
        temporal_pool_type (str): The temporal pool type. Choices are 'avg' or
            'max'. Default: 'avg'.
        spatial_pool_type (str): The spatial pool type. Choices are 'avg' or
            'max'. Default: 'max'.
        in_channels (int): The number of input channels. Default: 2048.
        focal_alpha (float): The hyper-parameter alpha for Focal Loss.
            When alpha == 1 and gamma == 0, Focal Loss degenerates to
            BCELossWithLogits. Default: 1.
        focal_gamma (float): The hyper-parameter gamma for Focal Loss.
            When alpha == 1 and gamma == 0, Focal Loss degenerates to
            BCELossWithLogits. Default: 0.
        num_classes (int): The number of classes. Default: 81.
        dropout_ratio (float): A float in [0, 1], indicates the dropout_ratio.
            Default: 0.
        dropout_before_pool (bool): Dropout Feature before spatial temporal
            pooling. Default: True.
        topk (int or tuple[int]): Parameter for evaluating multilabel accuracy.
            Default: (3, 5)
        multilabel (bool): Whether used for a multilabel task. Default: True.
            (Only support multilabel == True now).
    """

    def __init__(
            self,
            temporal_pool_type='avg',
            spatial_pool_type='max',
            in_channels=2048,
            # The first class is reserved, to classify bbox as pos / neg
            focal_gamma=0.,
            focal_alpha=1.,
            num_classes=81,
            dropout_ratio=0,
            dropout_before_pool=True,
            topk=(3, 5),
            multilabel=True):

        super(BBoxHeadAVA, self).__init__()
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.dropout_ratio = dropout_ratio
        self.dropout_before_pool = dropout_before_pool

        self.multilabel = multilabel

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

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

        # Handle AVA first
        assert self.multilabel

        in_channels = self.in_channels
        # Pool by default
        if self.temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if self.spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)

        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.debug_imgs = None

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        if self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)

        x = self.temporal_pool(x)
        x = self.spatial_pool(x)

        if not self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        # We do not predict bbox, so return None
        return cls_score, None

    @staticmethod
    def get_targets(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        cls_reg_targets = bbox_target(pos_proposals, neg_proposals,
                                      pos_gt_labels, rcnn_train_cfg)
        return cls_reg_targets

    @staticmethod
    def recall_prec(pred_vec, target_vec):
        """
        Args:
            pred_vec (tensor[N x C]): each element is either 0 or 1
            target_vec (tensor[N x C]): each element is either 0 or 1

        """
        correct = pred_vec & target_vec
        # Seems torch 1.5 has no auto type conversion
        recall = correct.sum(1) / (target_vec.sum(1).float() + 1e-6)
        prec = correct.sum(1) / (pred_vec.sum(1) + 1e-6)
        return recall.mean(), prec.mean()

    def multi_label_accuracy(self, pred, target, thr=0.5):
        pred = pred.sigmoid()
        pred_vec = pred > thr
        # Target is 0 or 1, so using 0.5 as the borderline is OK
        target_vec = target > 0.5
        recall_thr, prec_thr = self.recall_prec(pred_vec, target_vec)

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
        return recall_thr, prec_thr, recalls, precs

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets=None,
             bbox_weights=None,
             reduce=True):

        losses = dict()
        if cls_score is not None:
            # Only use the cls_score
            # labels = labels[:, 1:]
            # pos_inds = torch.sum(labels, dim=-1) > 0
            # cls_score = cls_score[pos_inds, 1:]
            # labels = labels[pos_inds]

            bce_loss = F.binary_cross_entropy_with_logits

            loss = bce_loss(cls_score, labels, reduction='none')
            pt = torch.exp(-loss)
            F_loss = self.focal_alpha * (1 - pt)**self.focal_gamma * loss
            losses['loss_action_cls'] = torch.mean(F_loss)

            recall_thr, prec_thr, recall_k, prec_k = self.multi_label_accuracy(
                cls_score, labels, thr=0.5)
            losses['recall@thr=0.5'] = recall_thr
            losses['prec@thr=0.5'] = prec_thr
            for i, k in enumerate(self.topk):
                losses[f'recall@top{k}'] = recall_k[i]
                losses[f'prec@top{k}'] = prec_k[i]
        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       img_shape,
                       flip=False,
                       crop_quadruple=None,
                       cfg=None):

        # might be used by testing w. augmentation
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        assert self.multilabel

        scores = cls_score.sigmoid() if cls_score is not None else None
        bboxes = rois[:, 1:]
        assert bboxes.shape[-1] == 4

        # First reverse the flip
        img_h, img_w = img_shape
        if flip:
            bboxes_ = bboxes.clone()
            bboxes_[:, 0] = img_w - 1 - bboxes[:, 2]
            bboxes_[:, 2] = img_w - 1 - bboxes[:, 0]
            bboxes = bboxes_

        # Then normalize the bbox to [0, 1]
        bboxes[:, 0::2] /= img_w
        bboxes[:, 1::2] /= img_h

        def _bbox_crop_undo(bboxes, crop_quadruple):
            decropped = bboxes.clone()

            if crop_quadruple is not None:
                x1, y1, tw, th = crop_quadruple
                decropped[:, 0::2] = bboxes[..., 0::2] * tw + x1
                decropped[:, 1::2] = bboxes[..., 1::2] * th + y1

            return decropped

        bboxes = _bbox_crop_undo(bboxes, crop_quadruple)
        return bboxes, scores


class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias
        # self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class BBoxTranSTLHeadAVA(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively.

    Args:
        temporal_pool_type (str): The temporal pool type. Choices are 'avg' or
            'max'. Default: 'avg'.
        spatial_pool_type (str): The spatial pool type. Choices are 'avg' or
            'max'. Default: 'max'.
        in_channels (int): The number of input channels. Default: 2048.
        focal_alpha (float): The hyper-parameter alpha for Focal Loss.
            When alpha == 1 and gamma == 0, Focal Loss degenerates to
            BCELossWithLogits. Default: 1.
        focal_gamma (float): The hyper-parameter gamma for Focal Loss.
            When alpha == 1 and gamma == 0, Focal Loss degenerates to
            BCELossWithLogits. Default: 0.
        num_classes (int): The number of classes. Default: 81.
        dropout_ratio (float): A float in [0, 1], indicates the dropout_ratio.
            Default: 0.
        dropout_before_pool (bool): Dropout Feature before spatial temporal
            pooling. Default: True.
        topk (int or tuple[int]): Parameter for evaluating multilabel accuracy.
            Default: (3, 5)
        multilabel (bool): Whether used for a multilabel task. Default: True.
            (Only support multilabel == True now).
    """

    def __init__(
            self,
            num_classes,
            in_channels,
            loss_cls=dict(type='CrossEntropyLoss'),
            dropout_ratio=0.5,
            tranST=dict(
                hidden_dim = 768,
                enc_layer_num = 2,
                stld_layer_num = 4,
                n_head = 8,
                dim_feedforward=1024,
                dropout=0.1,
                drop_path_rate=0.1,
                normalize_before=True,
                fusion=True,
                rm_self_attn_dec=False,
                rm_first_self_attn=False,
                activation="relu",
                return_intermediate_dec=False,
                t_only=False
            ),
            topk=(3, 5),
            multilabel=True,
            **kwargs):

        super(BBoxTranSTLHeadAVA, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.multilabel = multilabel
        self.dropout_ratio = dropout_ratio

        # self.focal_gamma = focal_gamma
        # self.focal_alpha = focal_alpha

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

        # Handle AVA first
        assert self.multilabel

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = nn.Identity()
        self.tranST_config = tranST
        self.TranST = TranST(d_temporal_branch=tranST['hidden_dim'],
                             d_spatial_branch=tranST['hidden_dim'],
                             n_head=tranST['n_head'],
                             fusion=tranST['fusion'],
                             num_encoder_layers=tranST['enc_layer_num'],
                             num_decoder_layers=tranST['stld_layer_num'],
                             dim_feedforward=tranST['dim_feedforward'],
                             dropout=tranST['dropout'],
                             drop_path_rate=tranST['drop_path_rate'],
                             activation=tranST['activation'],
                             normalize_before=tranST['normalize_before'],
                             return_intermediate_dec=tranST['return_intermediate_dec'],
                             rm_res_self_attn=tranST['rm_res_self_attn'],
                             rm_first_self_attn=tranST['rm_first_self_attn'],
                             t_only=tranST['t_only']
        )
        # self.drop_path = DropPath(tranST['drop_path_rate']) if tranST['drop_path_rate'] > 0. else nn.Identity()
        self.pos_enc_module = PositionalEnconding(d_model=tranST['hidden_dim'], dropout=tranST['dropout'], max_len=10000, ret_enc=True)
        if in_channels!=tranST['hidden_dim']:
            # self.transform = SEModule(in_channels, tranST['hidden_dim'], 1/16)
            self.transform = nn.Conv3d(in_channels, tranST['hidden_dim'], 1)
        else:
            self.transform = nn.Identity()

        if tranST['t_only'] or tranST['stld_layer_num']==0:
            self.fc_cls = GroupWiseLinear(num_classes, tranST['hidden_dim'], bias=True)
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
            # self.fc_cls = nn.Linear(tranST['hidden_dim']*2, num_classes)
            self.fc_cls = GroupWiseLinear(num_classes, tranST['hidden_dim']*2, bias=True)
        
        self.base_cls = nn.Conv3d(self.in_channels, num_classes, 1)
        # self.base_cls = nn.Linear(tranST['hidden_dim'], num_classes)
        # self.norm = nn.LayerNorm((num_classes, tranST['hidden_dim']))
        self.spatial_pool  = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        # if not self.tranST_config['t_only']:
        #     normal_init(self.fc_cls, std=self.init_std)
        kaiming_init(self.base_cls, bias=0)
        if self.transform is not None:
            kaiming_init(self.transform, bias=0)
    
    def label_embed(self, x):
        # x = x.view(x.size(0), -1).unsqueeze(2)
        mask = self.base_cls(x)
        coarse_socre = mask.view(mask.size(0), -1)
        mask = torch.sigmoid(mask)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        # mask = mask.transpose(1, 2)

        x = self.transform(x)
        residual = x
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(mask, x)
        return x, residual, coarse_socre

    def coarse_cls(self, x):
        # x = x.view(x.size(0), -1).unsqueeze(2)
        x = self.base_cls(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0].mean(dim=-1)
        return x

    def label_embed_r(self, x):
        x = x.view(x.size(0), -1)
        coarse_score = self.base_cls(x)
        label_embedding = x.unsqueeze(1).repeat(1, self.num_classes, 1)
        label_embedding = label_embedding * self.base_cls.weight
        # label_embedding = self.norm(label_embedding)
        return label_embedding, coarse_score

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # x = self.transform(x)
        # x = self.transform_norm(x.permute(0, 2, 3, 4, 1).contiguous()).permute(0, 4, 1, 2, 3).contiguous()
        # print(x.shape)
        x = self.dropout(x)
        f = self.global_avg_pool(x)

        # ==========================================
        label_embedding, x, score1 = self.label_embed(f)
        # label_embedding, x, score1 = self.label_embed_r(f)
        if not self.tranST_config['t_only']:
            F_s = self.temporal_pool(x)
            F_s = F_s.flatten(2)
            pos_s = self.pos_enc_module(F_s.permute(0,2,1))
            pos_s = pos_s.permute(0,2,1)

        F_t = self.spatial_pool(x)
        F_t = F_t.flatten(2)
        pos_t = self.pos_enc_module(F_t.permute(0,2,1))
        pos_t = pos_t.permute(0,2,1)

        if self.tranST_config['stld_layer_num']==0:
            h = label_embedding
        elif self.tranST_config['t_only']:
            h = self.TranST(None, F_t, label_embedding, None, pos_t)
        else:
            hs, ht = self.TranST(F_s, F_t, label_embedding, pos_s, pos_t)
            # residual structure
            # hs = hs + label_embedding
            # ht = ht + label_embedding
            h = torch.cat([hs, ht], dim=3)
        # =======================================
        # if self.dropout is not None:
        score2 = self.fc_cls(h[-1])
        # score2=self.fc_cls(label_embedding)
        cls_score = (score1 + score2) / 2.
        # return tuple([score1, score2])
        return cls_score, None

    @staticmethod
    def get_targets(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        cls_reg_targets = bbox_target(pos_proposals, neg_proposals,
                                      pos_gt_labels, rcnn_train_cfg)
        return cls_reg_targets

    @staticmethod
    def recall_prec(pred_vec, target_vec):
        """
        Args:
            pred_vec (tensor[N x C]): each element is either 0 or 1
            target_vec (tensor[N x C]): each element is either 0 or 1

        """
        correct = pred_vec & target_vec
        # Seems torch 1.5 has no auto type conversion
        recall = correct.sum(1) / (target_vec.sum(1).float() + 1e-6)
        prec = correct.sum(1) / (pred_vec.sum(1) + 1e-6)
        return recall.mean(), prec.mean()

    def multi_label_accuracy(self, pred, target, thr=0.5):
        pred = pred.sigmoid()
        pred_vec = pred > thr
        # Target is 0 or 1, so using 0.5 as the borderline is OK
        target_vec = target > 0.5
        recall_thr, prec_thr = self.recall_prec(pred_vec, target_vec)

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
        return recall_thr, prec_thr, recalls, precs

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets=None,
             bbox_weights=None,
             reduce=True,
             **kwargs):

        losses = dict()
        if cls_score is not None:
            # Only use the cls_score
            # labels = labels[:, 1:]
            # pos_inds = torch.sum(labels, dim=-1) > 0
            # cls_score = cls_score[pos_inds, 1:]
            # labels = labels[pos_inds]

            # bce_loss = F.binary_cross_entropy_with_logits

            # loss = bce_loss(cls_score, labels, reduction='none')
            # pt = torch.exp(-loss)
            # F_loss = self.focal_alpha * (1 - pt)**self.focal_gamma * loss
            # losses['loss_action_cls'] = torch.mean(F_loss)
            losses['loss_action_cls'] = self.loss_cls(cls_score, labels, **kwargs)

            recall_thr, prec_thr, recall_k, prec_k = self.multi_label_accuracy(
                cls_score, labels, thr=0.5)
            losses['recall@thr=0.5'] = recall_thr
            losses['prec@thr=0.5'] = prec_thr
            for i, k in enumerate(self.topk):
                losses[f'recall@top{k}'] = recall_k[i]
                losses[f'prec@top{k}'] = prec_k[i]
        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       img_shape,
                       flip=False,
                       crop_quadruple=None,
                       cfg=None):

        # might be used by testing w. augmentation
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        assert self.multilabel

        scores = cls_score.sigmoid() if cls_score is not None else None
        bboxes = rois[:, 1:]
        assert bboxes.shape[-1] == 4

        # First reverse the flip
        img_h, img_w = img_shape
        if flip:
            bboxes_ = bboxes.clone()
            bboxes_[:, 0] = img_w - 1 - bboxes[:, 2]
            bboxes_[:, 2] = img_w - 1 - bboxes[:, 0]
            bboxes = bboxes_

        # Then normalize the bbox to [0, 1]
        bboxes[:, 0::2] /= img_w
        bboxes[:, 1::2] /= img_h

        def _bbox_crop_undo(bboxes, crop_quadruple):
            decropped = bboxes.clone()

            if crop_quadruple is not None:
                x1, y1, tw, th = crop_quadruple
                decropped[:, 0::2] = bboxes[..., 0::2] * tw + x1
                decropped[:, 1::2] = bboxes[..., 1::2] * th + y1

            return decropped

        bboxes = _bbox_crop_undo(bboxes, crop_quadruple)
        return bboxes, scores


class BBoxX3DHeadAVA(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 focal_gamma=0.,
                 focal_alpha=1.,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 fc1_bias=False,
                 topk=(3, 5),
                 multilabel=True
                 ):
        super(BBoxX3DHeadAVA, self).__init__()
        self.multilabel = multilabel
        
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

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

        # Handle AVA first
        assert self.multilabel

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.in_channels = in_channels
        self.mid_channels = 2048
        self.num_classes = num_classes
        self.fc1_bias = fc1_bias

        self.fc1 = nn.Linear(
            self.in_channels, self.mid_channels, bias=self.fc1_bias)
        self.fc2 = nn.Linear(self.mid_channels, self.num_classes)

        self.relu = nn.ReLU()

        self.pool = None
        if self.spatial_type == 'avg':
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif self.spatial_type == 'max':
            self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        else:
            raise NotImplementedError

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc1, std=self.init_std)
        normal_init(self.fc2, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, T, H, W]
        assert self.pool is not None
        x = self.pool(x)
        # [N, in_channels, 1, 1, 1]
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        x = self.fc1(x)
        # [N, 2048]
        x = self.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc2(x)
        # [N, num_classes]
        return cls_score, None

    @staticmethod
    def get_targets(sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        cls_reg_targets = bbox_target(pos_proposals, neg_proposals,
                                      pos_gt_labels, rcnn_train_cfg)
        return cls_reg_targets

    @staticmethod
    def recall_prec(pred_vec, target_vec):
        """
        Args:
            pred_vec (tensor[N x C]): each element is either 0 or 1
            target_vec (tensor[N x C]): each element is either 0 or 1

        """
        correct = pred_vec & target_vec
        # Seems torch 1.5 has no auto type conversion
        recall = correct.sum(1) / target_vec.sum(1).float()
        prec = correct.sum(1) / (pred_vec.sum(1) + 1e-6)
        return recall.mean(), prec.mean()

    def multi_label_accuracy(self, pred, target, thr=0.5):
        pred = pred.sigmoid()
        pred_vec = pred > thr
        # Target is 0 or 1, so using 0.5 as the borderline is OK
        target_vec = target > 0.5
        recall_thr, prec_thr = self.recall_prec(pred_vec, target_vec)

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
        return recall_thr, prec_thr, recalls, precs

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets=None,
             bbox_weights=None,
             reduce=True):

        losses = dict()
        if cls_score is not None:
            # Only use the cls_score
            labels = labels[:, 1:]
            pos_inds = torch.sum(labels, dim=-1) > 0
            cls_score = cls_score[pos_inds, 1:]
            labels = labels[pos_inds]

            bce_loss = F.binary_cross_entropy_with_logits

            loss = bce_loss(cls_score, labels, reduction='none')
            pt = torch.exp(-loss)
            F_loss = self.focal_alpha * (1 - pt)**self.focal_gamma * loss
            losses['loss_action_cls'] = torch.mean(F_loss)

            recall_thr, prec_thr, recall_k, prec_k = self.multi_label_accuracy(
                cls_score, labels, thr=0.5)
            losses['recall@thr=0.5'] = recall_thr
            losses['prec@thr=0.5'] = prec_thr
            for i, k in enumerate(self.topk):
                losses[f'recall@top{k}'] = recall_k[i]
                losses[f'prec@top{k}'] = prec_k[i]
        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       img_shape,
                       flip=False,
                       crop_quadruple=None,
                       cfg=None):

        # might be used by testing w. augmentation
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        assert self.multilabel

        scores = cls_score.sigmoid() if cls_score is not None else None
        bboxes = rois[:, 1:]
        assert bboxes.shape[-1] == 4

        # First reverse the flip
        img_h, img_w = img_shape
        if flip:
            bboxes_ = bboxes.clone()
            bboxes_[:, 0] = img_w - 1 - bboxes[:, 2]
            bboxes_[:, 2] = img_w - 1 - bboxes[:, 0]
            bboxes = bboxes_

        # Then normalize the bbox to [0, 1]
        bboxes[:, 0::2] /= img_w
        bboxes[:, 1::2] /= img_h

        def _bbox_crop_undo(bboxes, crop_quadruple):
            decropped = bboxes.clone()

            if crop_quadruple is not None:
                x1, y1, tw, th = crop_quadruple
                decropped[:, 0::2] = bboxes[..., 0::2] * tw + x1
                decropped[:, 1::2] = bboxes[..., 1::2] * th + y1

            return decropped

        bboxes = _bbox_crop_undo(bboxes, crop_quadruple)
        return bboxes, scores

if mmdet_imported:
    MMDET_HEADS.register_module()(BBoxHeadAVA)
    MMDET_HEADS.register_module()(BBoxTranSTLHeadAVA)
    MMDET_HEADS.register_module()(BBoxX3DHeadAVA)
