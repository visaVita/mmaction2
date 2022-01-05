import torch
from torch._C import set_flush_denormal
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
from typing import Optional
from einops import rearrange

"""
TranST: spatial_encoder, temporal_encoder, 

Args:

"""

class TranST(nn.Module):
    def __init__(self,
                 d_temporal_branch=512,
                 d_spatial_branch=512,
                 n_head=8,
                 fusion=False,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 drop_path_rate=0.1,
                 activation="relu",
                 normalize_before=False,
                 return_intermediate_dec=False,
                 rm_first_self_attn=False,
                 rm_res_self_attn=False,
                 t_only = False
    ):
        super().__init__()
        self.t_only = t_only
        self.num_encoder_layers = num_encoder_layers
        self.rm_first_self_attn = rm_first_self_attn
        self.rm_res_self_attn = rm_res_self_attn
        if num_encoder_layers > 0:
            if not t_only:
                spatial_encoder_layer = TransformerEncoderLayer(d_spatial_branch, n_head, dim_feedforward,
                                                            dropout, activation, normalize_before)
                spatial_encoder_norm = nn.LayerNorm(d_spatial_branch) if normalize_before else None
                self.spatial_encoder = TransformerEncoder(spatial_encoder_layer, num_encoder_layers, spatial_encoder_norm)
            temporal_encoder_layer = TransformerEncoderLayer(d_temporal_branch, n_head, dim_feedforward,
                                                             dropout, activation, normalize_before)
            temporal_encoder_norm = nn.LayerNorm(d_temporal_branch) if normalize_before else None
            self.temporal_encoder = TransformerEncoder(temporal_encoder_layer, num_encoder_layers, temporal_encoder_norm)

        if not t_only:
            spatial_decoder_layer = TransformerDecoderLayer(d_spatial_branch, n_head, dim_feedforward,
                                                        dropout, activation, normalize_before)
            spatial_decoder_norm = nn.LayerNorm(d_spatial_branch)
        else:
            spatial_decoder_layer = None
            spatial_decoder_norm = None
        temporal_decoder_layer = TransformerDecoderLayer(d_temporal_branch, n_head, dim_feedforward,
                                                        dropout, activation, normalize_before)
        temporal_decoder_norm = nn.LayerNorm(d_temporal_branch)
        
        self.STLD = STLD(spatial_decoder_layer, temporal_decoder_layer, num_decoder_layers,
                         spatial_decoder_norm, temporal_decoder_norm,
                         d_spatial_branch=d_spatial_branch, d_temporal_branch=d_temporal_branch,
                         return_intermediate=return_intermediate_dec, fusion=fusion, temporal_only=t_only
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.rm_self_attn_dec_func()

    def rm_self_attn_dec_func(self):
        total_modifie_layer_num = 0
        rm_list = []

        layer_stack = self.STLD.temporal_layers if self.t_only else zip(self.STLD.spatial_layers, self.STLD.temporal_layers)
        for idx, (layer_s, layer_t) in enumerate(layer_stack):
            if idx == 0 and not self.rm_first_self_attn:
                continue
            if idx != 0 and not self.rm_res_self_attn:
                continue
            if not self.t_only:
                layer_s.omit_selfattn = True
                del layer_s.self_attn
                del layer_s.dropout1
                del layer_s.norm1
            
            layer_t.omit_selfattn = True
            del layer_t.self_attn
            del layer_t.dropout1
            del layer_t.norm1

            total_modifie_layer_num += 1
            rm_list.append(idx)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_s, src_t, query_embed, pos_s, pos_t, mask=None):
        # flatten NxCxHxW to HWxNxC
        #bs, c, h, w = src.shape
        #src = src.flatten(2).permute(2, 0, 1) 
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # bs, _, _ = src_t.shape
        if not self.t_only:
            src_s = src_s.permute(2, 0, 1)
            pos_s = pos_s.permute(2, 0, 1)
        src_t = src_t.permute(2, 0, 1)
        pos_t = pos_t.permute(2, 0, 1)
        query_embed = query_embed.transpose(0, 1)
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        if self.num_encoder_layers > 0:
            if not self.t_only:
                memory_s = self.spatial_encoder(src_s, src_key_padding_mask=mask, pos=pos_s)
            memory_t = self.temporal_encoder(src_t, src_key_padding_mask=mask, pos=pos_t)
        else:
            if not self.t_only:
                memory_s = src_s
            memory_t = src_t

        tgt = torch.zeros_like(query_embed)
        if not self.t_only:
            hs, ht = self.STLD(tgt=tgt,
                            memory_s=memory_s, memory_t=memory_t,
                            pos_s=pos_s, pos_t=pos_t,
                            query_pos_s=query_embed,
                            query_pos_t=query_embed
            )
            hs = self.drop_path(hs)
            ht = self.drop_path(ht)
            return hs.transpose(1, 2), ht.transpose(1, 2)
        else:
            ht = self.STLD( tgt=tgt,
                            memory_s=None, memory_t=memory_t,
                            pos_s=None, pos_t=pos_t,
                            query_pos_s=None,
                            query_pos_t=query_embed
            )
            ht = self.drop_path(ht)
            return ht.transpose(1, 2)

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


#  目前应该有两种fusion方案，一种是add, 一种是concat+Conv,
#  目前先不考虑slowfast，太特殊
class STLD(nn.Module):
    def __init__(self, spatial_decoder_layer, temporal_decoder_layer,
                 num_layers, spatial_norm, temporal_norm,
                 return_intermediate=False,
                 d_temporal_branch=512,
                 d_spatial_branch=512,
                 fusion = False,
                 temporal_only = False
                ):
        super().__init__()
        self.t_only = temporal_only
        self.fusion = fusion
        if not self.t_only:
            self.spatial_layers = _get_clones(spatial_decoder_layer, num_layers)
            self.spatial_norm = spatial_norm
            self.d_spatial = d_spatial_branch
        self.temporal_layers = _get_clones(temporal_decoder_layer, num_layers)
        self.num_layers = num_layers
        
        self.temporal_norm = temporal_norm
        self.d_temporal = d_temporal_branch
        
        self.return_intermediate = return_intermediate
        
        

    def forward(self, tgt, memory_s, memory_t,
                tgt_mask_s: Optional[Tensor] = None,
                tgt_mask_t: Optional[Tensor] = None,
                memory_mask_s: Optional[Tensor] = None,
                memory_mask_t: Optional[Tensor] = None,
                tgt_key_padding_mask_s: Optional[Tensor] = None,
                tgt_key_padding_mask_t: Optional[Tensor] = None,
                memory_key_padding_mask_s: Optional[Tensor] = None,
                memory_key_padding_mask_t: Optional[Tensor] = None,
                pos_s: Optional[Tensor] = None,
                pos_t: Optional[Tensor] = None,
                query_pos_s: Optional[Tensor] = None,
                query_pos_t: Optional[Tensor] = None):
        if self.t_only:
            output_t = tgt
            intermediate_t = []
            layer_num = len(self.temporal_layers)
            current_layer = 0
            for layer_t in self.temporal_layers:
                current_layer+=1
                output_t = layer_t(output_t, memory_t, tgt_mask=tgt_mask_t,
                                    memory_mask=memory_mask_t,
                                    tgt_key_padding_mask=tgt_key_padding_mask_t,
                                    memory_key_padding_mask=memory_key_padding_mask_t,
                                    pos=pos_t, query_pos=query_pos_t)
            
                if self.return_intermediate:
                    intermediate_t.append(self.norm(output_t))

            if self.temporal_norm is not None:
                output_t = self.temporal_norm(output_t)
                if self.return_intermediate:
                    intermediate_t.pop()
                    intermediate_t.append(output_t)

            if self.return_intermediate:
                return torch.stack(intermediate_t)

            return output_t.unsqueeze(0)
        else:
            output_s = tgt
            output_t = tgt

            intermediate_s = []
            intermediate_t = []

            layer_num = len(self.temporal_layers)
            
            current_layer = 0
            for layer_s, layer_t in zip(self.spatial_layers, self.temporal_layers):
                current_layer+=1
                output_s = layer_s(output_s, memory_s, tgt_mask=tgt_mask_s,
                                        memory_mask=memory_mask_s,
                                        tgt_key_padding_mask=tgt_key_padding_mask_s,
                                        memory_key_padding_mask=memory_key_padding_mask_s,
                                        pos=pos_s, query_pos=query_pos_s)

                output_t = layer_t(output_t, memory_t, tgt_mask=tgt_mask_t,
                                        memory_mask=memory_mask_t,
                                        tgt_key_padding_mask=tgt_key_padding_mask_t,
                                        memory_key_padding_mask=memory_key_padding_mask_t,
                                        pos=pos_t, query_pos=query_pos_t)
                if self.fusion:
                    if not (current_layer == 1 or current_layer == layer_num):
                        tmp = output_s
                        output_s = output_t
                        output_t = tmp
            
                if self.return_intermediate:
                    intermediate_t.append(self.norm(output_t))
                    intermediate_s.append(self.norm(output_s))

            if self.spatial_norm is not None:
                output_s = self.spatial_norm(output_s)
                output_t = self.temporal_norm(output_t)
                if self.return_intermediate:
                    intermediate_s.pop()
                    intermediate_t.pop()
                    intermediate_s.append(output_s)
                    intermediate_t.append(output_t)

            if self.return_intermediate:
                return torch.stack(intermediate_s), torch.stack(intermediate_t)

            return output_s.unsqueeze(0), output_t.unsqueeze(0)


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
        # self.dropout1 = nn.Identity()
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
        src2, _ = self.self_attn(q, k, value=src, attn_mask=src_mask,
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
        src2, _ = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
                            
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
        # self.dropout1 = nn.Identity()
        # self.dropout2 = nn.Identity()
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
            tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        tgt2, _ = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
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
        tgt2, _ = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
                            
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

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "hardswish":
        return F.hardswish
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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