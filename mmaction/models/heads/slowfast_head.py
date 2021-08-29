from torch.nn.modules.pooling import AdaptiveAvgPool1d
from mmaction.models.common import transformer
from mmaction.models.common.transformer import Transformer
import torch
import torch.nn as nn
import math
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead
from ...core import top_k_accuracy
from ..common import PositionalEnconding, Transformer
import torch.nn.functional as F

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

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

@HEADS.register_module()
class SlowFastHead(BaseHead):
    """The classification head for SlowFast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.8,
                 init_std=0.01,
                 focal_gamma=0.,
                 focal_alpha=1.,
                 topk=(3, 5),
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

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
            # to put the feature into transformer, dim of the feature must be (B, C, N) 
            # B = batch_size
            # C = channel
            # N = num_of_class 
            self.transformer_s = Transformer(d_model=2048, nhead=4, num_encoder_layers=1, 
                                           num_decoder_layers=2, dim_feedforward=256, 
                                           dropout=0.1)
            self.transformer_t = Transformer(d_model=256, nhead=4, num_encoder_layers=0, 
                                           num_decoder_layers=2, dim_feedforward=256, 
                                           dropout=0.1)
            self.pos_enc_module = PositionalEnconding(d_model=128, dropout=0.1, max_len=5000, ret_enc=True)
            # self.input_proj_s = nn.Conv1d(2048, 2048, 1)
            # self.input_proj_t = nn.Conv1d(256, 128, 1)
            self.query_embed = nn.Embedding(num_classes, 128)
            self.fc_cls = GroupWiseLinear(num_classes, in_channels)
            self.temporal_pool = nn.AdaptiveAvgPool3d(1, None, None)
            self.spatial_pool  = nn.AdaptiveAvgPool3d(None, 1, 1)
        else:
            self.fc_cls = nn.Linear(in_channels, num_classes)

        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.AdaptiveMaxPool3d((1, 1, 1))


    def init_weights(self):
        """Initiate the parameters from scratch."""
        if not self.multi_class:
            normal_init(self.fc_cls, std=self.init_std)
        else:
            self.fc_cls.reset_parameters()

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # ([N, channel_fast, T, H, W], [(N, channel_slow, T, H, W)])
        x_fast, x_slow = x
        # ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1])
        # x_fast = self.avg_pool(x_fast)
        # x_slow = self.avg_pool(x_slow)
        # [N, channel_fast + channel_slow, 1, 1, 1]
        # x = torch.cat((x_slow, x_fast), dim=1)
        
        # if self.dropout is not None:
        #     x = self.dropout(x)

        # [N x C]
        # x = x.view(x.size(0), -1)
        # [N x num_classes]

        if self.multi_class:
            F_s = self.temporal_pool(x_slow)
            F_t = self.spatial_pool(x_fast)
            if self.dropout is not None:
                F_s = self.dropout(F_s)
                F_t = self.dropout(F_t)
            F_s = F_s.view(F_s.size(0), F_s.size(1), -1)
            F_t = F_t.view(F_t.size(0), F_t.size(1), -1)

            # bs, c = x.shape
            # x = x.repeat(self.num_classes, 1, 1).permute(1, 2, 0)
            # x = self.input_proj(x)
            pos_s = self.pos_enc_module(F_s.permute(0, 2, 1))
            pos_s = pos_s.permute(0, 2, 1)
            pos_t = self.pos_enc_module(F_t.permute(0, 2, 1))
            pos_t = pos_t.permute(0, 2, 1)
            # print(x.shape, pos.shape)
            query_input = self.query_embed.weight
            hs = self.transformer_s(F_s, query_input, pos_s)[0]
            ht = self.transformer_t(F_t, query_input, pos_t)[0]
            h = torch.cat([hs, ht], dim=2)
            cls_score = self.fc_cls(h[-1])
        else:
            cls_score = self.fc_cls(x)

        return cls_score
