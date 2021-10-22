from ..builder import HEADS
from .base import BaseHead
from ..common import PositionalEnconding, Transformer

from collections import OrderedDict
import torch
from torch.nn.modules.utils import _triple, _pair
import torch.nn.functional as F
from typing import Any, Callable, Optional, Tuple, Union
from einops import rearrange
from torch import nn, Tensor

class Hardsigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = (0.2 * x + 0.5).clamp(min=0.0, max=1.0)
        return x


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class CausalModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activation = None

    def reset_activation(self) -> None:
        self.activation = None


class TemporalCGAvgPool3D(CausalModule):
    def __init__(self,) -> None:
        super().__init__()
        self.n_cumulated_values = 0
        self.register_forward_hook(self._detach_activation)

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape
        device = x.device
        cumulative_sum = torch.cumsum(x, dim=2)
        if self.activation is None:
            self.activation = cumulative_sum[:, :, -1:].clone()
        else:
            cumulative_sum += self.activation
            self.activation = cumulative_sum[:, :, -1:].clone()
        divisor = (torch.arange(1, input_shape[2]+1,
                   device=device)[None, None, :, None, None]
                   .expand(x.shape))
        x = cumulative_sum / (self.n_cumulated_values + divisor)
        self.n_cumulated_values += input_shape[2]
        return x

    @staticmethod
    def _detach_activation(module: CausalModule,
                           input: Tensor,
                           output: Tensor) -> None:
        module.activation.detach_()

    def reset_activation(self) -> None:
        super().reset_activation()
        self.n_cumulated_values = 0


class Conv2dBNActivation(nn.Sequential):
    def __init__(
                 self,
                 in_planes: int,
                 out_planes: int,
                 *,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None,
                 **kwargs: Any,
                 ) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        if norm_layer is None:
            norm_layer = nn.Identity
        if activation_layer is None:
            activation_layer = nn.Identity
        self.kernel_size = kernel_size
        self.stride = stride
        dict_layers = OrderedDict({
                            "conv2d": nn.Conv2d(in_planes, out_planes,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                groups=groups,
                                                **kwargs),
                            "norm": norm_layer(out_planes, eps=0.001),
                            "act": activation_layer()
                            })

        self.out_channels = out_planes
        super(Conv2dBNActivation, self).__init__(dict_layers)


class Conv3DBNActivation(nn.Sequential):
    def __init__(
                 self,
                 in_planes: int,
                 out_planes: int,
                 *,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 padding: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None,
                 **kwargs: Any,
                 ) -> None:
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        if norm_layer is None:
            norm_layer = nn.Identity
        if activation_layer is None:
            activation_layer = nn.Identity
        self.kernel_size = kernel_size
        self.stride = stride

        dict_layers = OrderedDict({
                                "conv3d": nn.Conv3d(in_planes, out_planes,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    groups=groups,
                                                    **kwargs),
                                "norm": norm_layer(out_planes, eps=0.001),
                                "act": activation_layer()
                                })

        self.out_channels = out_planes
        super(Conv3DBNActivation, self).__init__(dict_layers)

class ConvBlock3D(CausalModule):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            *,
            kernel_size: Union[int, Tuple[int, int, int]],
            tf_like: bool,
            causal: bool,
            conv_type: str,
            padding: Union[int, Tuple[int, int, int]] = 0,
            stride: Union[int, Tuple[int, int, int]] = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation_layer: Optional[Callable[..., nn.Module]] = None,
            bias: bool = False,
            **kwargs: Any,
            ) -> None:
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        self.conv_2 = None
        if tf_like:
            # We neek odd kernel to have even padding
            # and stride == 1 to precompute padding,
            if kernel_size[0] % 2 == 0:
                raise ValueError('tf_like supports only odd'
                                 + ' kernels for temporal dimension')
            padding = ((kernel_size[0]-1)//2, 0, 0)
            if stride[0] != 1:
                raise ValueError('illegal stride value, tf like supports'
                                 + ' only stride == 1 for temporal dimension')
            if stride[1] > kernel_size[1] or stride[2] > kernel_size[2]:
                # these values are not tested so should be avoided
                raise ValueError('tf_like supports only'
                                 + '  stride <= of the kernel size')

        if causal is True:
            padding = (0, padding[1], padding[2])
        if conv_type != "2plus1d" and conv_type != "3d":
            raise ValueError("only 2plus2d or 3d are "
                             + "allowed as 3d convolutions")

        if conv_type == "2plus1d":
            self.conv_1 = Conv2dBNActivation(in_planes,
                                             out_planes,
                                             kernel_size=(kernel_size[1],
                                                          kernel_size[2]),
                                             padding=(padding[1],
                                                      padding[2]),
                                             stride=(stride[1], stride[2]),
                                             activation_layer=activation_layer,
                                             norm_layer=norm_layer,
                                             bias=bias,
                                             **kwargs)
            if kernel_size[0] > 1:
                self.conv_2 = Conv2dBNActivation(in_planes,
                                                 out_planes,
                                                 kernel_size=(kernel_size[0],
                                                              1),
                                                 padding=(padding[0], 0),
                                                 stride=(stride[0], 1),
                                                 activation_layer=activation_layer,
                                                 norm_layer=norm_layer,
                                                 bias=bias,
                                                 **kwargs)
        elif conv_type == "3d":
            self.conv_1 = Conv3DBNActivation(in_planes,
                                             out_planes,
                                             kernel_size=kernel_size,
                                             padding=padding,
                                             activation_layer=activation_layer,
                                             norm_layer=norm_layer,
                                             stride=stride,
                                             bias=bias,
                                             **kwargs)
        self.padding = padding
        self.kernel_size = kernel_size
        self.dim_pad = self.kernel_size[0]-1
        self.stride = stride
        self.causal = causal
        self.conv_type = conv_type
        self.tf_like = tf_like

    def _forward(self, x: Tensor) -> Tensor:
        device = x.device
        if self.dim_pad > 0 and self.conv_2 is None and self.causal is True:
            x = self._cat_stream_buffer(x, device)
        shape_with_buffer = x.shape
        if self.conv_type == "2plus1d":
            x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.conv_1(x)
        if self.conv_type == "2plus1d":
            x = rearrange(x,
                          "(b t) c h w -> b c t h w",
                          t=shape_with_buffer[2])

            if self.conv_2 is not None:
                if self.dim_pad > 0 and self.causal is True:
                    x = self._cat_stream_buffer(x, device)
                w = x.shape[-1]
                x = rearrange(x, "b c t h w -> b c t (h w)")
                x = self.conv_2(x)
                x = rearrange(x, "b c t (h w) -> b c t h w", w=w)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.tf_like:
            x = same_padding(x, x.shape[-2], x.shape[-1],
                             self.stride[-2], self.stride[-1],
                             self.kernel_size[-2], self.kernel_size[-1])
        x = self._forward(x)
        return x

    def _cat_stream_buffer(self, x: Tensor, device: torch.device) -> Tensor:
        if self.activation is None:
            self._setup_activation(x.shape)
        x = torch.cat((self.activation.to(device), x), 2)
        self._save_in_activation(x)
        return x

    def _save_in_activation(self, x: Tensor) -> None:
        assert self.dim_pad > 0
        self.activation = x[:, :, -self.dim_pad:, ...].clone().detach()

    def _setup_activation(self, input_shape: Tuple[float, ...]) -> None:
        assert self.dim_pad > 0
        self.activation = torch.zeros(*input_shape[:2],  # type: ignore
                                      self.dim_pad,
                                      *input_shape[3:])

class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int,  # TODO rename activations
                 activation_2: nn.Module,
                 activation_1: nn.Module,
                 conv_type: str,
                 causal: bool,
                 squeeze_factor: int = 4,
                 bias: bool = True) -> None:
        super().__init__()
        self.causal = causal
        se_multiplier = 2 if causal else 1
        squeeze_channels = _make_divisible(input_channels
                                           // squeeze_factor
                                           * se_multiplier, 8)
        self.temporal_cumualtive_GAvg3D = TemporalCGAvgPool3D()
        self.fc1 = ConvBlock3D(input_channels*se_multiplier,
                               squeeze_channels,
                               kernel_size=(1, 1, 1),
                               padding=0,
                               tf_like=False,
                               causal=causal,
                               conv_type=conv_type,
                               bias=bias)
        self.activation_1 = activation_1()
        self.activation_2 = activation_2()
        self.fc2 = ConvBlock3D(squeeze_channels,
                               input_channels,
                               kernel_size=(1, 1, 1),
                               padding=0,
                               tf_like=False,
                               causal=causal,
                               conv_type=conv_type,
                               bias=bias)

    def _scale(self, input: Tensor) -> Tensor:
        if self.causal:
            x_space = torch.mean(input, dim=[3, 4], keepdim=True)
            scale = self.temporal_cumualtive_GAvg3D(x_space)
            scale = torch.cat((scale, x_space), dim=1)
        else:
            scale = F.adaptive_avg_pool3d(input, 1)
        scale = self.fc1(scale)
        scale = self.activation_1(scale)
        scale = self.fc2(scale)
        return self.activation_2(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


def _make_divisible(v: float,
                    divisor: int,
                    min_value: Optional[int] = None
                    ) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def same_padding(x: Tensor,
                 in_height: int, in_width: int,
                 stride_h: int, stride_w: int,
                 filter_height: int, filter_width: int) -> Tensor:
    if (in_height % stride_h == 0):
        pad_along_height = max(filter_height - stride_h, 0)
    else:
        pad_along_height = max(filter_height - (in_height % stride_h), 0)
    if (in_width % stride_w == 0):
        pad_along_width = max(filter_width - stride_w, 0)
    else:
        pad_along_width = max(filter_width - (in_width % stride_w), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding_pad = (pad_left, pad_right, pad_top, pad_bottom)
    return torch.nn.functional.pad(x, padding_pad)


class tfAvgPool3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avgf = nn.AvgPool3d((1, 3, 3), stride=(1, 2, 2))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != x.shape[-2]:
            raise RuntimeError('only same shape for h and w ' +
                               'are supported by avg with tf_like')
        if x.shape[-1] != x.shape[-2]:
            raise RuntimeError('only same shape for h and w ' +
                               'are supported by avg with tf_like')
        f1 = x.shape[-1] % 2 != 0
        if f1:
            padding_pad = (0, 0, 0, 0)
        else:
            padding_pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, padding_pad)
        if f1:
            x = torch.nn.functional.avg_pool3d(x,
                                               (1, 3, 3),
                                               stride=(1, 2, 2),
                                               count_include_pad=False,
                                               padding=(0, 1, 1))
        else:
            x = self.avgf(x)
            x[..., -1] = x[..., -1] * 9/6
            x[..., -1, :] = x[..., -1, :] * 9/6
        return x

@HEADS.register_module()
class MoViNetHead(BaseHead):
    def __init__(self,
                 cfg,
                 num_classes,
                 loss_cls,
                 dropout_ratio=0.5,
                 topk=(3, 5),
                 **kwargs):
        super().__init__(num_classes, cfg['conv7']['out_channels'], loss_cls=loss_cls, topk=topk, **kwargs)
        self.dropout_ratio = dropout_ratio
        in_channels = cfg['conv7']['out_channels']
        hidden_dim = cfg['dense9']['hidden_dim']
        
        # topk
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

        # dropout
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        # transformer
        self.transformer_s = Transformer(d_model=hidden_dim,
                                         nhead=8,
                                         num_encoder_layers=0,
                                         num_decoder_layers=2,
                                         dim_feedforward=hidden_dim*2,
                                         dropout=0.2,
                                         rm_first_self_attn=False,
                                         rm_self_attn_dec=False
        )
        self.transformer_t = Transformer(d_model=hidden_dim,
                                         nhead=8,
                                         num_encoder_layers=0,
                                         num_decoder_layers=2,
                                         dim_feedforward=hidden_dim*2,
                                         dropout=0.2,
                                         rm_first_self_attn=False,
                                         rm_self_attn_dec=False
        )
        self.pos_enc_module = PositionalEnconding(d_model=hidden_dim, dropout=0.1, max_len=5000, ret_enc=True)
        self.transform = nn.Sequential(
            ConvBlock3D(in_channels,
                        hidden_dim,
                        kernel_size=(1, 1, 1),
                        tf_like=True,
                        causal=False,
                        conv_type='3d',
                        bias=True),
            Swish()
        )
        self.base_cls = nn.Linear(hidden_dim, num_classes)
        self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        self.spatial_pool  = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.fc_cls = nn.Linear(hidden_dim*2, num_classes)



        """ self.classifier = nn.Sequential(
                # dense9
                ConvBlock3D(cfg['conv7']['out_channels'],
                            cfg['dense9']['hidden_dim'],
                            kernel_size=(1, 1, 1),
                            tf_like=tf_like,
                            causal=causal,
                            conv_type=conv_type,
                            bias=True),
                Swish(),
                nn.Dropout(p=0.2, inplace=True),
                # dense10d
                ConvBlock3D(cfg['dense9']['hidden_dim'],
                            num_classes,
                            kernel_size=(1, 1, 1),
                            tf_like=tf_like,
                            causal=causal,
                            conv_type=conv_type,
                            bias=True),
            )
        if causal:
            self.cgap = TemporalCGAvgPool3D()

    def avg(self, x: Tensor) -> Tensor:
        if self.causal:
            avg = F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1))
            avg = self.cgap(avg)[:, :, -1:]
        else:
            avg = F.adaptive_avg_pool3d(x, 1)
        return avg """
    
    def init_weights(self):
        for m in enumerate(self.modules()):
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
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

        cls_score = (score_1 + score_2) / 2
        return cls_score
