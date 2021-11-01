# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from ..common import PositionalEnconding, Transformer
from ..builder import HEADS
from .base import BaseHead

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
                 topk=(1, 3),
                 transformer=False,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.transformer = transformer
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

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

        if self.transformer:
            
            # to put the feature into transformer, dim of the feature must be (B, C, N) 
            # B = batch_size
            # C = channel
            # N = num_of_class
        
            self.transformer_s = Transformer(d_model=2048, nhead=8, num_encoder_layers=0,
                                           num_decoder_layers=2, dim_feedforward=2048,
                                           dropout=0.2, rm_first_self_attn=False, rm_self_attn_dec=False)
            self.transformer_t = Transformer(d_model=256, nhead=8, num_encoder_layers=0,
                                           num_decoder_layers=2, dim_feedforward=2048,
                                           dropout=0.2, rm_first_self_attn=False, rm_self_attn_dec=False)
            self.pos_enc_module_s = PositionalEnconding(d_model=2048, dropout=0.1, max_len=5000, ret_enc=True)
            self.pos_enc_module_t = PositionalEnconding(d_model=256, dropout=0.1, max_len=5000, ret_enc=True)

            # self.transform_s = nn.Sequential(
            #     nn.Conv3d(2048, 1024, 1),
            #     nn.ReLU(inplace=True),
            #     nn.BatchNorm3d(1024)
            # )
            # self.transform_t = nn.Sequential(
            #     nn.Conv3d(256, 1024, 1),
            #     nn.ReLU(inplace=True),
            #     nn.BatchNorm3d(1024)
            # )


            # self.pos_enc_module_t = PositionalEnconding(d_model=256, dropout=0.1, max_len=5000, ret_enc=True)
            # self.input_proj_s = nn.Conv1d(2048, 2048, 1)
            # self.input_proj_t = nn.Conv1d(256, 128, 1)
            # self.query_embed_s = nn.Embedding(num_classes, 2048)
            # self.query_embed_t = nn.Embedding(num_classes, 256)

            self.base_cls_slow = nn.Linear(2048, num_classes)
            self.base_cls_fast = nn.Linear(256, num_classes)

            # self.fc_cls = GroupWiseLinear(num_classes, in_channels)
            
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
            self.spatial_pool  = nn.AdaptiveAvgPool3d((None, 1, 1))
            self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
            self.delta = nn.Parameter(torch.tensor(0.5))
            self.theta = nn.Parameter(torch.tensor(0.25))
            self.gamma = nn.Parameter(torch.tensor(0.25))
            self.fc_cls = nn.Linear(in_channels, num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        # self.focal_gamma = focal_gamma
        # self.focal_alpha = focal_alpha

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.AdaptiveMaxPool3d((1, 1, 1))


    def init_weights(self):
        """Initiate the parameters from scratch."""
        if not self.transformer:
            normal_init(self.fc_cls, std=self.init_std)
        else:
            normal_init(self.fc_cls, std=self.init_std)
            normal_init(self.base_cls_fast, std=self.init_std)
            normal_init(self.base_cls_slow, std=self.init_std)
            # self.fc_cls.reset_parameters()

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # ([N, channel_fast, T, H, W], [(N, channel_slow, T, H, W)])
        if not self.transformer:
            x_slow, x_fast = x
            # ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1])
            x_fast = self.avg_pool(x_fast)
            x_slow = self.avg_pool(x_slow)
            # [N, channel_fast + channel_slow, 1, 1, 1]
            x = torch.cat((x_slow, x_fast), dim=1)
            
            if self.dropout is not None:
                x = self.dropout(x)

            # [N x C]
            x = x.view(x.size(0), -1)
            # [N x num_classes]
            cls_score = self.fc_cls(x)
            return cls_score
        ###################################################
        else:
            x_slow, x_fast = x
            #score_1 = self.base_cls(x)
            # x_slow = self.transform_s(x_slow)
            # x_fast = self.transform_t(x_fast)
            x_s = self.avg_pool(x_slow)
            x_t = self.avg_pool(x_fast)
            # import ipdb; ipdb.set_trace()
            if self.dropout is not None:
                x_s = self.dropout(x_s)
                x_t = self.dropout(x_t)
            
            x_s = x_s.view(x_s.size(0), -1)
            x_t = x_t.view(x_t.size(0), -1)

            score_fast = self.base_cls_fast(x_t)
            score_slow = self.base_cls_slow(x_s)
            
            # score_1 = (score_11 + score_12) / 2
            b, _ = x_s.shape
            
            label_embedding_s = x_s.repeat(1, self.num_classes)
            label_embedding_s = label_embedding_s.view(b, self.num_classes, -1)
            label_embedding_s = label_embedding_s * self.base_cls_slow.weight
            label_embedding_t = x_t.repeat(1, self.num_classes)
            label_embedding_t = label_embedding_t.view(b, self.num_classes, -1)
            label_embedding_t = label_embedding_t * self.base_cls_fast.weight
            
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
            pos_s = self.pos_enc_module_s(F_s.permute(0, 2, 1))
            pos_s = pos_s.permute(0, 2, 1)
            pos_t = self.pos_enc_module_t(F_t.permute(0, 2, 1))
            pos_t = pos_t.permute(0, 2, 1)
            
            # import ipdb; ipdb.set_trace()
            # print(x.shape, pos.shape)
            # query_input_s = self.query_embed_s.weight
            # query_input_t = self.query_embed_t.weight
            
            hs = self.transformer_s(F_s, label_embedding_s, pos_s)[0]
            ht = self.transformer_t(F_t, label_embedding_t, pos_t)[0]
            
            h = torch.cat([hs, ht], dim=3)
            score_2 = self.fc_cls(h[-1])
            mask_mat = self.mask_mat.detach()
            score_2 = (score_2 * mask_mat).sum(-1)
            cls_score = self.delta*score_2 + self.theta*score_fast + self.gamma*score_slow
            # cls_score =score_2
            """
            import time
            att_mat = dict(
                cc_s  = sim_mat_cc_s.detach().cpu().numpy(),
                cc_t  = sim_mat_cc_t.detach().cpu().numpy(),
                cs    = sim_mat_cs.detach().cpu().numpy(),
                ct    = sim_mat_ct.detach().cpu().numpy(),
                score = torch.sigmoid(cls_score).detach().cpu().numpy()
            )
            time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            with open("/home/ckai/project/mmaction2/work_dirs/slowfast_cot_r50_8x8x1_40e_charades_rgb/att_mat/" + time + ".npy", "wb") as f:
                np.save(f, att_mat) """
        return cls_score
