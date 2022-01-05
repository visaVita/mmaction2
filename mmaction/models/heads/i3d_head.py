import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead
from ...core import top_k_accuracy
from ..common import PositionalEnconding, tranST, Transformer


@HEADS.register_module()
class I3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
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
                 Transformer=False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.transformer = Transformer
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        if self.transformer:
            hidden_dim = 1024
            self.transformer_s = Transformer(d_model=hidden_dim,
                                            nhead=8,
                                            num_encoder_layers=0,
                                            num_decoder_layers=2,
                                            dim_feedforward=hidden_dim*2,
                                            dropout=0.1,
                                            rm_first_self_attn=False,
                                            rm_self_attn_dec=False
            )
            self.transformer_t = Transformer(d_model=hidden_dim,
                                            nhead=8,
                                            num_encoder_layers=0,
                                            num_decoder_layers=2,
                                            dim_feedforward=hidden_dim*2,
                                            dropout=0.1,
                                            rm_first_self_attn=False,
                                            rm_self_attn_dec=False
            )
            self.pos_enc_module = PositionalEnconding(d_model=hidden_dim, dropout=0.1, max_len=5000, ret_enc=True)
            if in_channels != hidden_dim:
                self.transform = nn.Sequential(
                    nn.Conv3d(in_channels, hidden_dim, 1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm3d(hidden_dim)
                )
            else:
                self.transform = None
            self.base_cls = nn.Linear(hidden_dim, num_classes)
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
            self.spatial_pool  = nn.AdaptiveAvgPool3d((None, 1, 1))
            self.global_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
            self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
            self.fc_cls = nn.Linear(hidden_dim*2, num_classes)

        else:
            self.fc_cls = nn.Linear(self.in_channels, self.num_classes)


            if self.spatial_type == 'avg':
                # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
                self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            else:
                self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        if not self.transformer:
            normal_init(self.fc_cls, std=self.init_std)
        else:
            normal_init(self.fc_cls, std=self.init_std)
            normal_init(self.base_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if not self.transformer:
            
            if self.avg_pool is not None:
                x = self.avg_pool(x)
            # [N, in_channels, 1, 1, 1]
            if self.dropout is not None:
                x = self.dropout(x)
            # [N, in_channels, 1, 1, 1]
            x = x.view(x.shape[0], -1)

            # [N, in_channels]
            cls_score = self.fc_cls(x)
            # [N, num_classes]
        
        else:
            if self.transform is not None:
                x = self.transform(x)
            if self.dropout is not None:
                x = self.dropout(x)
            F_s = self.temporal_pool(x)
            F_t = self.spatial_pool(x)
            F_s = F_s.view(F_s.size(0), F_s.size(1), -1)
            F_t = F_t.view(F_t.size(0), F_t.size(1), -1)
            x = self.global_max_pool(x)
            x = x.view(x.size(0), -1)
            b, _ = x.shape
            score_1 = self.base_cls(x)
            label_embedding = x.repeat(1, self.num_classes)
            label_embedding = label_embedding.view(b, self.num_classes, -1)
            label_embedding = label_embedding * self.base_cls.weight

            pos_s = self.pos_enc_module(F_s.permute(0,2,1))
            pos_s = pos_s.permute(0,2,1)
            pos_t = self.pos_enc_module(F_t.permute(0,2,1))
            pos_t = pos_t.permute(0,2,1)

            hs = self.transformer_s(F_s, label_embedding, pos_s)[0]
            ht = self.transformer_t(F_t, label_embedding, pos_t)[0]
            h = torch.cat([hs, ht], dim=3)
            score_2 = self.fc_cls(h[-1])
            mask_mat = self.mask_mat.detach()
            score_2 = (score_2 * mask_mat).sum(-1)

            cls_score = (score_1 + score_2) / 2.

        # if self.multi_class:
        #     cls_score = torch.sigmoid(cls_score)
        return cls_score
