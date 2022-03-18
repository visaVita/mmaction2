from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
import math
from torch.autograd import Variable
from einops import rearrange
from typing import Optional
from mmcv.cnn import build_norm_layer, constant_init
from mmcv.cnn.bricks.registry import ATTENTION, FEEDFORWARD_NETWORK
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.runner.base_module import BaseModule
from mmcv.utils import digit_version


@ATTENTION.register_module()
class DividedTemporalAttentionWithNorm(BaseModule):
    """Temporal Attention in Divided Space Time Attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder.
        num_frames (int): Number of frames in the video.
        attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
            0..
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Defaults to 0..
        dropout_layer (dict): The dropout_layer used when adding the shortcut.
            Defaults to `dict(type='DropPath', drop_prob=0.1)`.
        norm_cfg (dict): Config dict for normalization layer. Defaults to
            `dict(type='LN')`.
        init_cfg (dict | None): The Config for initialization. Defaults to
            None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.1),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

        if digit_version(torch.__version__) < digit_version('1.9.0'):
            kwargs.pop('batch_first', None)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.temporal_fc = nn.Linear(self.embed_dims, self.embed_dims)

        self.init_weights()

    def init_weights(self):
        constant_init(self.temporal_fc, val=0, bias=0)

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        assert residual is None, (
            'Always adding the shortcut in the forward function')

        init_cls_token = query[:, 0, :].unsqueeze(1)
        identity = query_t = query[:, 1:, :]

        # query_t [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_t.size()
        p, t = pt // self.num_frames, self.num_frames

        # res_temporal [batch_size * num_patches, num_frames, embed_dims]
        query_t = self.norm(query_t.reshape(b * p, t, m)).permute(1, 0, 2)
        res_temporal = self.attn(query_t, query_t, query_t)[0].permute(1, 0, 2)
        res_temporal = self.dropout_layer(
            self.proj_drop(res_temporal.contiguous()))
        res_temporal = self.temporal_fc(res_temporal)

        # res_temporal [batch_size, num_patches * num_frames, embed_dims]
        res_temporal = res_temporal.reshape(b, p * t, m)

        # ret_value [batch_size, num_patches * num_frames + 1, embed_dims]
        new_query_t = identity + res_temporal
        new_query = torch.cat((init_cls_token, new_query_t), 1)
        return new_query


@ATTENTION.register_module()
class DividedSpatialAttentionWithNorm(BaseModule):
    """Spatial Attention in Divided Space Time Attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder.
        num_frames (int): Number of frames in the video.
        attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
            0..
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Defaults to 0..
        dropout_layer (dict): The dropout_layer used when adding the shortcut.
            Defaults to `dict(type='DropPath', drop_prob=0.1)`.
        norm_cfg (dict): Config dict for normalization layer. Defaults to
            `dict(type='LN')`.
        init_cfg (dict | None): The Config for initialization. Defaults to
            None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.1),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
        if digit_version(torch.__version__) < digit_version('1.9.0'):
            kwargs.pop('batch_first', None)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # init DividedSpatialAttentionWithNorm by default
        pass

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        assert residual is None, (
            'Always adding the shortcut in the forward function')

        identity = query
        init_cls_token = query[:, 0, :].unsqueeze(1)
        query_s = query[:, 1:, :]

        # query_s [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_s.size()
        p, t = pt // self.num_frames, self.num_frames

        # cls_token [batch_size * num_frames, 1, embed_dims]
        cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t,
                                                           m).unsqueeze(1)

        # query_s [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)
        query_s = torch.cat((cls_token, query_s), 1)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = self.norm(query_s).permute(1, 0, 2)
        res_spatial = self.attn(query_s, query_s, query_s)[0].permute(1, 0, 2)
        res_spatial = self.dropout_layer(
            self.proj_drop(res_spatial.contiguous()))

        # cls_token [batch_size, 1, embed_dims]
        cls_token = res_spatial[:, 0, :].reshape(b, t, m)
        cls_token = torch.mean(cls_token, 1, True)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = rearrange(
            res_spatial[:, 1:, :], '(b t) p m -> b (p t) m', p=p, t=t)
        res_spatial = torch.cat((cls_token, res_spatial), 1)

        new_query = identity + res_spatial
        return new_query


@FEEDFORWARD_NETWORK.register_module()
class FFNWithNorm(FFN):
    """FFN with pre normalization layer.

    FFNWithNorm is implemented to be compatible with `BaseTransformerLayer`
    when using `DividedTemporalAttentionWithNorm` and
    `DividedSpatialAttentionWithNorm`.

    FFNWithNorm has one main difference with FFN:

    - It apply one normalization layer before forwarding the input data to
        feed-forward networks.

    Args:
        embed_dims (int): Dimensions of embedding. Defaults to 256.
        feedforward_channels (int): Hidden dimension of FFNs. Defaults to 1024.
        num_fcs (int, optional): Number of fully-connected layers in FFNs.
            Defaults to 2.
        act_cfg (dict): Config for activate layers.
            Defaults to `dict(type='ReLU')`
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Defaults to 0..
        add_residual (bool, optional): Whether to add the
            residual connection. Defaults to `True`.
        dropout_layer (dict | None): The dropout_layer used when adding the
            shortcut. Defaults to None.
        init_cfg (dict): The Config for initialization. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer. Defaults to
            `dict(type='LN')`.
    """

    def __init__(self, *args, norm_cfg=dict(type='LN'), **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

    def forward(self, x, residual=None):
        assert residual is None, ('Cannot apply pre-norm with FFNWithNorm')
        return super().forward(self.norm(x), x)


class PositionalEnconding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000,ret_enc=False):
        '''
        :param d_model: 词嵌入维度
        :param dropout: 丢失率
        :param max_len: 每个句子的最长长度
        '''
        super(PositionalEnconding,self).__init__()
        # 实例化dropout层
        self.dpot = nn.Dropout(p=dropout)
        self.ret_enc = ret_enc
        # 初始化位置编码矩阵
        pos_enc = torch.zeros(max_len,d_model)

        # 初始化绝对位置矩阵
        # position矩阵size为(max_len,1)
        position = torch.arange(0,max_len).unsqueeze(1)

        # 将绝对位置矩阵和位置编码矩阵特征融合
        # 定义一个变换矩阵 跳跃式初始化
        div_term = torch.exp(torch.arange(0,d_model,2) * -(math.log(10000)/d_model))
        pos_enc[:,0::2] = torch.sin(position * div_term)
        pos_enc[:,1::2] = torch.cos(position * div_term)

        # 将二维张量扩充成三维张量
        pos_enc = pos_enc.unsqueeze((0))

        # 把pe位置编码矩阵注册成模型的buffer
        # 模型保存后重加载时和模型结构与参数一同被加载
        self.register_buffer('pos_enc',pos_enc)

    def forward(self,x):
        '''
        :param x: 文本的词嵌入表示
        :return:
        '''
        if(self.ret_enc):
            return Variable(self.pos_enc[:,:x.size(1)], requires_grad=False)
        x = x + Variable(self.pos_enc[:,:x.size(1)], requires_grad=False)
        return self.dpot(x)

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, 
                 rm_self_attn_dec=True, rm_first_self_attn=True
                 ):
        super().__init__()

        self.num_encoder_layers = num_encoder_layers
        if num_decoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.d_model = d_model
        self.nhead = nhead
        self.rm_self_attn_dec = rm_self_attn_dec
        self.rm_first_self_attn = rm_first_self_attn
        self._reset_parameters()
        if self.rm_self_attn_dec or self.rm_first_self_attn:
            self.rm_self_attn_dec_func()

        # self.debug_mode = False
        # self.set_debug_mode(self.debug_mode)

    def rm_self_attn_dec_func(self):
        total_modifie_layer_num = 0
        rm_list = []
        for idx, layer in enumerate(self.decoder.layers):
            if idx == 0 and not self.rm_first_self_attn:
                continue
            if idx != 0 and not self.rm_self_attn_dec:
                continue
            
            layer.omit_selfattn = True
            del layer.self_attn
            del layer.dropout1
            del layer.norm1

            total_modifie_layer_num += 1
            rm_list.append(idx)
        # remove some self-attention layer
        # print("rm {} layer: {}".format(total_modifie_layer_num, rm_list))

    def set_debug_mode(self, status):
        print("set debug mode to {}!!!".format(status))
        self.debug_mode = status
        if hasattr(self, 'encoder'):
            for idx, layer in enumerate(self.encoder.layers):
                layer.debug_mode = status
                layer.debug_name = str(idx)
        if hasattr(self, 'decoder'):
            for idx, layer in enumerate(self.decoder.layers):
                layer.debug_mode = status
                layer.debug_name = str(idx)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed, mask=None):
        # flatten NxCxHxW to HWxNxC
        #bs, c, h, w = src.shape
        #src = src.flatten(2).permute(2, 0, 1) 
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        bs, c, n_cls = src.shape
        src = src.permute(2, 0, 1)
        pos_embed = pos_embed.permute(2, 0, 1)
        query_embed = query_embed.transpose(0, 1)
        #query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        if self.num_encoder_layers > 0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        else:
            memory = src

        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        
        return hs.transpose(1, 2), memory[:n_cls].permute(1, 2, 0).view(bs, c, n_cls)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output, _ , _ = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.debug_mode = False
        self.debug_name = None

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2= self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
                            
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.debug_mode = False
        self.debug_name = None
        self.omit_selfattn = False

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        if not self.omit_selfattn:
            tgt2, sim_mat_1 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        tgt2, sim_mat_2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                key=self.with_pos_embed(memory, pos),
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if not self.omit_selfattn:
            return tgt
        else:
            return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
                            
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=False,
        rm_self_attn_dec=not args.keep_other_self_attn_dec,
        rm_first_self_attn=not args.keep_first_self_attn_dec,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
