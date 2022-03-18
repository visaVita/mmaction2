import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from ..builder import HEADS
from .base import BaseHead
from ..common import PositionalEnconding, TranST
import math

class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class SEModule(nn.Module):

    def __init__(self, in_channels, out_channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.bottleneck = self._round_width(in_channels, reduction)
        self.fc1 = nn.Conv3d(
            in_channels, self.bottleneck, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(
            self.bottleneck, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.fc_re = nn.Conv3d(in_channels, out_channels, 1)

    @staticmethod
    def _round_width(width, multiplier, min_width=8, divisor=8):
        width *= multiplier
        min_width = min_width or divisor
        width_out = max(min_width,
                        int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def forward(self, x):
        module_input = x
        module_input = self.fc_re(module_input)
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

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
        # self.loss_TranSTL = build_loss(loss_cls)
        # self.loss_base = build_loss(loss_cls)
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = nn.Identity()
        self.tranST_config = tranST
        if tranST['stld_layer_num']>0:
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
        
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        # self.temporal_trans = nn.Conv3d(self.in_channels, tranST['hidden_dim'], 1)
        # self.transform = nn.Linear(self.in_channels, tranST['hidden_dim'])
        # self.fc_cls = GroupWiseLinear(num_classes, tranST['hidden_dim']*2, bias=True)
        
        self.base_cls = nn.Linear(self.in_channels, num_classes, bias=True)
        # self.base_cls = nn.Linear(tranST['hidden_dim'], num_classes)
        self.fc_cls = GroupWiseLinear(num_classes, tranST['hidden_dim']*2, bias=True)
        # self.norm = nn.LayerNorm((num_classes, tranST['hidden_dim']))
        self.spatial_pool  = nn.AdaptiveAvgPool3d((None, 1, 1))
        # self.spatial_trans = nn.Conv3d(self.in_channels, tranST['hidden_dim'], 1)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())

    def init_weights(self):
        """Initiate the parameters from scratch."""
        # if not self.tranST_config['t_only']:
        #     normal_init(self.fc_cls, std=self.init_std)
        normal_init(self.base_cls, std=self.init_std, bias=0)
        # normal_init(self.fc_cls, std=self.init_std, bias=0)
        # kaiming_init(self.spatial_trans, bias=0)
        # kaiming_init(self.temporal_trans, bias=0)
        # if self.transform is not None:
        #     kaiming_init(self.transform, bias=0)
    
    def label_embed(self, x):
        # x = x.view(x.size(0), -1).unsqueeze(2)
        mask = self.base_cls(x)
        coarse_socre = mask.view(mask.size(0), -1)
        mask = torch.sigmoid(mask)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        # mask = mask.transpose(1, 2)

        x = self.transform(x)
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(mask, x)
        return x, coarse_socre

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
        # print(label_embedding.shape)
        # print(self.fc_cls.weight.shape)
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
        # print(x.shape)
        x = self.dropout(x)
        # x = self.transform(x)
        # x = self.transform_norm(x.permute(0, 2, 3, 4, 1).contiguous()).permute(0, 4, 1, 2, 3).contiguous()
        
        f = self.global_avg_pool(x)
        
        # ==========================================
        label_embedding, score1 = self.label_embed_r(f)
        # temp = label_embedding
        # label_embedding = self.transform(label_embedding)
        # x = self.transform(x)
        # label_embedding, x, score1 = self.label_embed_r(f)
        if self.tranST_config['stld_layer_num']>0:
            if not self.tranST_config['t_only']:
                F_s = self.temporal_pool(x)
                # F_s = self.temporal_trans(F_s)
                F_s = F_s.flatten(2)
                pos_s = self.pos_enc_module(F_s.permute(0,2,1))
                pos_s = pos_s.permute(0,2,1)

            F_t = self.spatial_pool(x)
            # F_t = self.spatial_trans(F_t)
            F_t = F_t.flatten(2)
            pos_t = self.pos_enc_module(F_t.permute(0,2,1))
            pos_t = pos_t.permute(0,2,1)
        # print(F_s.shape, F_t.shape, label_embedding.shape, pos_s.shape, pos_t.shape)
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
            # h = hs + ht
        # =======================================
        # if self.dropout is not None:
        # score2 = self.fc_cls(h[-1])
        # score2=self.fc_cls(label_embedding)
        score2 = self.fc_cls(h[-1]) # B*1*num_classes

        # mask_mat = self.mask_mat.detach()
        # score2 = (score2 * mask_mat).sum(-1)
        cls_score = (score1 + score2) / 2.
        return cls_score
        # return score2
        
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
            # cls_score = (tranST_score + base_score) / 2.
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            if (isinstance(cls_score, tuple) and cls_score[0].size()[0] == 1) or cls_score.size()[0] == 1 :
                labels = labels.unsqueeze(0)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels + self.label_smooth_eps / self.num_classes)

        if self.multi_class and cls_score is not None:
            # Only use the cls_score
            if isinstance(cls_score, tuple):
                loss['base_loss'] = self.loss_cls(base_score, labels, **kwargs)
                loss['Tran_loss'] = self.loss_cls(tranST_score, labels, **kwargs)
                cls_score = (tranST_score + base_score) / 2.
            else:
                loss['loss_cls'] = self.loss_cls(cls_score, labels, **kwargs)

            recall_thr, prec_thr, recall_k, prec_k = self.multi_label_accuracy(
                cls_score, labels, thr=0.5)
            losses['recall@thr=0.5'] = recall_thr
            losses['prec@thr=0.5'] = prec_thr
            for i, k in enumerate(self.topk):
                losses[f'recall@top{k}'] = recall_k[i]
                losses[f'prec@top{k}'] = prec_k[i]

        # loss_cls may be dictionary or single tensor
        if isinstance(loss, dict):
            losses.update(loss)
        else:
            losses['loss_cls'] = loss

        return losses


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)