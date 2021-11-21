import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead
from ...core import top_k_accuracy
from ..common import PositionalEnconding, TranST


@HEADS.register_module()
class TranSTHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        tranST (dict): Config for building tranST.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 tranST=dict(
                     hidden_dim = 512,
                     enc_layer_num = 2,
                     stld_layer_num = 4,
                     n_head = 8,
                     dim_feedforward=1024,
                     dropout=0.1,
                     normalize_before=True,
                     fusion=True,
                     rm_self_attn_dec=False,
                     rm_first_self_attn=False,
                     activation="relu",
                     return_intermediate_dec=False,
                     t_only=False
                 ),
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        # self.with_tranST = with_tranST
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.tranST_config = tranST
        self.TranST = TranST(d_temporal_branch=tranST['hidden_dim'],
                             d_spatial_branch=tranST['hidden_dim'],
                             n_head=tranST['n_head'],
                             fusion=tranST['fusion'],
                             num_encoder_layers=tranST['enc_layer_num'],
                             num_decoder_layers=tranST['stld_layer_num'],
                             dim_feedforward=tranST['dim_feedforward'],
                             dropout=tranST['dropout'],
                             activation=tranST['activation'],
                             normalize_before=tranST['normalize_before'],
                             return_intermediate_dec=tranST['return_intermediate_dec'],
                             rm_self_attn_dec=tranST['rm_self_attn_dec'],
                             rm_first_self_attn=tranST['rm_first_self_attn'],
                             t_only=tranST['t_only']
                             )
            
        self.pos_enc_module = PositionalEnconding(d_model=tranST['hidden_dim'], dropout=tranST['dropout'], max_len=5000, ret_enc=True)
        if in_channels != tranST['hidden_dim']:
            self.transform = nn.Sequential(
                nn.Conv3d(in_channels, tranST['hidden_dim'], 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(tranST['hidden_dim'])
            )
        else:
            self.transform = None
        self.fc_cls = nn.Linear(tranST['hidden_dim'], num_classes)
        if not tranST['t_only']:
            
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
            self.w_add_1 = nn.Parameter(torch.tensor(0.5))
            self.w_add_2 = nn.Parameter(torch.tensor(0.5))
            self.fc_cls = nn.Linear(tranST['hidden_dim']*2, num_classes)
        self.base_cls = nn.Linear(tranST['hidden_dim'], num_classes)
        
        self.spatial_pool  = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        if not self.tranST_config['t_only']:
            normal_init(self.fc_cls, std=self.init_std)
        normal_init(self.base_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        
        if self.transform is not None:
            x = self.transform(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if not self.tranST_config['t_only']:
            F_s = self.temporal_pool(x)
            F_s = F_s.view(F_s.size(0), F_s.size(1), -1)
            pos_s = self.pos_enc_module(F_s.permute(0,2,1))
            pos_s = pos_s.permute(0,2,1)
        F_t = self.spatial_pool(x)
        F_t = F_t.view(F_t.size(0), F_t.size(1), -1)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        b, _ = x.shape
        score1 = self.base_cls(x)
        label_embedding = x.repeat(1, self.num_classes)
        label_embedding = label_embedding.view(b, self.num_classes, -1)
        label_embedding = label_embedding * self.base_cls.weight

        pos_t = self.pos_enc_module(F_t.permute(0,2,1))
        pos_t = pos_t.permute(0,2,1)
        if not self.tranST_config['t_only']:
            hs, ht = self.TranST(F_s, F_t, label_embedding, pos_s, pos_t)
            h = torch.cat([hs, ht], dim=3)
            score2 = self.fc_cls(h[-1])
            mask_mat = self.mask_mat.detach()
            score2= (score2 * mask_mat).sum(-1)
            cls_score = score1 + score2
            # cls_, ind_ = torch.sigmoid(cls_score).topk(3)
            return cls_score
        ht = self.TranST(None, F_t, label_embedding, None, pos_t)
        score2 = self.fc_cls(ht[-1])
        mask_mat = self.mask_mat.detach()
        score2= (score2 * mask_mat).sum(-1)
        cls_score = score1 + score2
        # cls_, ind_ = torch.sigmoid(cls_score).topk(3)
        return cls_score
        

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
        loss = dict()
        if isinstance(cls_score, tuple):
            cls_score, _ = cls_score
            # loss['base_loss'] = self.loss_cls(base_score, labels, **kwargs)

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        if self.multi_class and cls_score is not None:
            # Only use the cls_score
            
            loss['loss_cls'] = self.loss_cls(cls_score, labels, **kwargs)

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
