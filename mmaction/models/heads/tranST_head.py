import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init

from ..builder import HEADS
from .base import BaseHead
from ...core import top_k_accuracy
from ..common import PositionalEnconding, TranST
import math
from torch.nn import init


class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c = x.size()
        y = self.fc(x).view(b, c, 1)
        return x * y.expand_as(x)

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
                 dropout_ratio=0.5,
                 init_std=0.01,
                 tranST=dict(
                     hidden_dim = 512,
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
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        # self.with_tranST = with_tranST
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
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

        self.pos_enc_module = PositionalEnconding(d_model=tranST['hidden_dim'], dropout=tranST['dropout'], max_len=10000, ret_enc=True)
        # self.pos_emb_s = nn.Parameter(torch.randn(1, 49, tranST['hidden_dim']))
        # self.pos_emb_t = nn.Parameter(torch.randn(1, 32, tranST['hidden_dim']))
        # self.se = SEAttention(channel=tranST['hidden_dim'], reduction=16)
        # self.label_embedding = nn.Embedding(num_classes, tranST['hidden_dim'])
        if in_channels != tranST['hidden_dim']:
            self.transform = nn.Sequential(
                nn.Conv3d(in_channels, tranST['hidden_dim'], 1),
                nn.ReLU(inplace=True)
            )
            # self.transform = nn.Conv3d(in_channels, tranST['hidden_dim'], 1)
        else:
            self.transform = nn.Identity()
        # self.fc_cls = GroupWiseLinear(num_classes, tranST['hidden_dim'], bias=True)
        # self.fc_cls = nn.Linear(tranST['hidden_dim'], num_classes)
        if not tranST['t_only']:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
            # self.fc_cls = nn.Linear(tranST['hidden_dim']*2, num_classes)
            self.fc_cls = GroupWiseLinear(num_classes, tranST['hidden_dim']*2, bias=True)
        else:
            self.fc_cls = GroupWiseLinear(num_classes, tranST['hidden_dim'], bias=True)
        
        # self.base_cls = nn.Conv3d(self.in_channels, num_classes, 1)
        self.base_cls = nn.Linear(self.in_channels, num_classes)
        
        self.spatial_pool  = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # self.w_add_1 = nn.Parameter(torch.Tensor(0.5))
        # self.w_add_2 = nn.Parameter(torch.Tensor(0.5))
        # self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())

    def init_weights(self):
        """Initiate the parameters from scratch."""
        # if not self.tranST_config['t_only']:
        #     normal_init(self.fc_cls, std=self.init_std)
        kaiming_init(self.base_cls, bias=0)
        if self.transform is not None:
            kaiming_init(self.transform)
        # normal_init(self.pos_emb_t, std=self.init_std)
        # normal_init(self.pos_emb_s, std=self.init_std)
        # normal_init(self.mask_mat, std=self.init_std)
        # normal_init(self.label_embedding)
    
    def label_embed(self, x):
        # x = x.view(x.size(0), -1).unsqueeze(2)
        mask = self.base_cls(x)
        mask = torch.sigmoid(mask)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        # mask = mask.transpose(1, 2)
        
        x = self.transform(x)
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(mask, x)
        return x

    def coarse_cls(self, x):
        # x = x.view(x.size(0), -1).unsqueeze(2)
        x = self.base_cls(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0].mean(dim=-1)
        return x
    
    def label_embed_r(self, x):
        label_embedding = x.repeat(1, 1, self.num_classes, 1, 1)
        x = x.view(x.size(0), -1)
        coarse_score = self.base_cls(x)
        
        label_embedding = label_embedding.view(x.size(0), self.num_classes, -1)
        label_embedding = label_embedding * self.base_cls.weight
        label_embedding = label_embedding.transpose(1, 2).unsqueeze(3).unsqueeze(4)
        label_embedding = self.transform(label_embedding)
        label_embedding = label_embedding.transpose(1, 2).contiguous()
        label_embedding = label_embedding.view(label_embedding.size(0), label_embedding.size(1), -1)
        return label_embedding, coarse_score


    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        x = self.dropout(x)
        f = self.global_avg_pool(x)
        # score1 = self.coarse_cls(f)
        # label_embedding = self.label_embed(f)

        # ==========================================
        label_embedding, score1 = self.label_embed_r(f)
        # ==========================================

        # F_s = x.view(x.size(0), -1, x.size(2), x.size(3))
        # F_s = self.transform_s(F_s)

        # F_t = x.transpose(1, 2)
        # F_t = F_t.view(F_t.size(0), F_t.size(1), -1)
        # F_t = F_t.transpose(1, 2)
        # F_t = self.transform_t(F_t)

        # if self.transform is not None:
        #     x = self.transform(x)

        # pos_s = self.pos_enc_module(F_s.permute(0,2,1))
        # pos_s = pos_s.permute(0,2,1)

        # pos_t = self.pos_enc_module(F_t.permute(0,2,1))
        # pos_t = pos_t.permute(0,2,1)

        # ========================================
        
        x = self.transform(x)
        if not self.tranST_config['t_only']:
            F_s = self.temporal_pool(x)
            F_s = F_s.view(F_s.size(0), F_s.size(1), -1)
            pos_s = self.pos_enc_module(F_s.permute(0,2,1))
            pos_s = pos_s.permute(0,2,1)


        F_t = self.spatial_pool(x)
        F_t = F_t.view(F_t.size(0), F_t.size(1), -1)
        pos_t = self.pos_enc_module(F_t.permute(0,2,1))
        pos_t = pos_t.permute(0,2,1)
        # =======================================

        
        
        # =======================================
        # label_embedding = self.label_embed(x)
        # score1 = self.coarse_cls(x)
        # label_embedding = x.repeat(1, self.num_classes)
        # label_embedding = label_embedding.view(b, self.num_classes, -1)
        # label_embedding = label_embedding * self.base_cls.weight

        # query = self.label_embedding.weight
        # =======================================
        if not self.tranST_config['t_only']:
            hs, ht = self.TranST(F_s, F_t, label_embedding, pos_s, pos_t)
            # residual structure
            # hs = hs + label_embedding
            # ht = ht + label_embedding
            h = torch.cat([hs, ht], dim=3)
        else:
            h = self.TranST(None, F_t, label_embedding, None, pos_t)
            # h = h + label_embedding
        # =======================================
        # if self.dropout is not None:
        score2 = self.fc_cls(h[-1])
        # score2=self.fc_cls(label_embedding)
        cls_score = (score1 + score2) / 2.
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
            base_score, tranST_score = cls_score
            cls_score = (tranST_score + base_score) / 2.
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels + self.label_smooth_eps / self.num_classes)

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
