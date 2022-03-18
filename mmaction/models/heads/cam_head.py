import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init

from ..builder import HEADS
from .base import BaseHead
from ...core import top_k_accuracy
import math

def obj_loc(score, threshold):
    smax, sdis, sdim = 0, 0, score.size(0)
    minsize = int(math.ceil(sdim * 0.125))  #0.125
    # minsize = 1
    snorm = (score - threshold).sign()
    snormdiff = (snorm[1:] - snorm[:-1]).abs()

    szero = (snormdiff==2).nonzero()
    if len(szero)==0:
       zmin, zmax = int(math.ceil(sdim*0.125)), int(math.ceil(sdim*0.875))
       return zmin, zmax

    if szero[0] > 0:
       lzmin, lzmax = 0, szero[0].item()
       lzdis = lzmax - lzmin
       lsmax, _ = score[lzmin:lzmax].max(0)
       if lsmax > smax:
          smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
       if lsmax == smax:
          if lzdis > sdis:
             smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if szero[-1] < sdim:
       lzmin, lzmax = szero[-1].item(), sdim
       lzdis = lzmax - lzmin
       lsmax, _ = score[lzmin:lzmax].max(0)
       if lsmax > smax:
          smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
       if lsmax == smax:
          if lzdis > sdis:
             smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if len(szero) >= 2:
       for i in range(len(szero)-1):
           lzmin, lzmax = szero[i].item(), szero[i+1].item()
           lzdis = lzmax - lzmin
           lsmax, _ = score[lzmin:lzmax].max(0)
           if lsmax > smax:
              smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis
           if lsmax == smax:
              if lzdis > sdis:
                 smax, zmin, zmax, sdis = lsmax, lzmin, lzmax, lzdis

    if zmax - zmin <= minsize:
        pad = minsize-(zmax-zmin)
        if zmin > int(math.ceil(pad/2.0)) and sdim - zmax > pad:
            zmin = zmin - int(math.ceil(pad/2.0)) + 1
            zmax = zmax + int(math.ceil(pad/2.0))
        if zmin < int(math.ceil(pad/2.0)):
            zmin = 0
            zmax =  minsize
        if sdim - zmax < int(math.ceil(pad/2.0)):
            zmin = sdim - minsize + 1
            zmax = sdim
    
    if zmax - zmin < 1:
        if sdim-zmax>=1:
            zmax = zmax+1
        else:
            zmin = zmin-1
    
    return zmin, zmax


@HEADS.register_module()
class CAMHead(BaseHead):
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
                 topN=6,
                 vis=False,
                 threshold=0.5,
                #  init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.topN = topN
        self.vis = vis
        self.threshold = threshold
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Conv3d(self.in_channels, self.num_classes, 1, bias = True)
        self.fc_local = nn.Sequential(
            nn.Conv3d(self.in_channels, 4 * self.in_channels, 1, bias = True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv3d(4 * self.in_channels, self.num_classes, 1, bias = True),
        )
        

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        kaiming_init(self.fc_cls)
        

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        b, d, t, h, w = x.shape
        # [N, in_channels, 4, 7, 7]   
        gf = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            gf = self.dropout(gf)
        # [N, in_channels, 1, 1, 1]
        
        gs = self.fc_cls(gf)
        # [N, num_classes, 1, 1, 1]
        gs = torch.sigmoid(gs)
        gs = gs.view(x.shape[0], -1)
        # [N, num_classes]

        camscore = self.fc_cls(x.detach())
        camscore = torch.sigmoid(camscore)
        wscore = F.adaptive_max_pool3d(camscore, (1, 1, None)).squeeze(dim=2).squeeze(dim=2)
        hscore = F.adaptive_max_pool3d(camscore, (1, None, 1)).squeeze(dim=4).squeeze(dim=2)
        tscore = F.adaptive_max_pool3d(camscore, (None, 1, 1)).squeeze(dim=3).squeeze(dim=3)
        # print(wscore.shape, hscore.shape, tscore.shape)

        proposals = torch.zeros([b, self.topN, d, t, h, w]).cuda() 
        if self.vis == True:
           region_bboxs = torch.FloatTensor(b, self.topN, 8)
        for i in range(b): 
            gs_inv, gs_ind = gs[i].sort(descending=True)
            for j in range(self.topN):
                xs = wscore[i,gs_ind[j],:].squeeze()
                ys = hscore[i,gs_ind[j],:].squeeze()
                ts = tscore[i,gs_ind[j],:].squeeze()
                if xs.max() == xs.min():
                   xs = xs/xs.max()
                else: 
                   xs = (xs-xs.min())/(xs.max()-xs.min())
                if ys.max() == ys.min():
                   ys = ys/ys.max()
                else:
                   ys = (ys-ys.min())/(ys.max()-ys.min())
                if ts.max() == ts.min():
                   ts = ts/ts.max()
                else:
                   ts = (ts-ts.min())/(ts.max()-ts.min())
                x1, x2 = obj_loc(xs, self.threshold)
                y1, y2 = obj_loc(ys, self.threshold)
                t1, t2 = obj_loc(ts, self.threshold)
                # print(x.shape, x1, x2, y1, y2, t1, t2)
                proposals[i:i+1, j ] = F.interpolate(x[i:i+1, :, t1:t2, y1:y2, x1:x2], size=(t, h, w), mode='trilinear', align_corners=True)
                if self.vis == True:
                   region_bboxs[i,j] = torch.Tensor([t1, t2, x1, x2, y1, y2, gs_ind[j].item(), gs[i, gs_ind[j]].item()])
        
        proposals = proposals.view(b*self.topN, d, t, h, w)
        lf = F.adaptive_max_pool3d(proposals, (1,1,1))
        lf = self.fc_local(lf)
        ls = torch.sigmoid(lf)
        ls = F.adaptive_max_pool2d(ls.reshape(b, self.topN, self.num_classes, 1).permute(0, 3, 1, 2), (1, self.num_classes))
        ls = ls.view(ls.size(0), -1)
        
        if self.vis == True:
            return gs, ls, region_bboxs
        else:
            # print(gs.shape, ls.shape)
            return gs, ls 

        # return cls_score


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
                gs, ls = cls_score
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
                    loss['g_loss'] = self.loss_cls(gs, labels, **kwargs)
                    loss['l_loss'] = self.loss_cls(ls, labels, **kwargs)
                    cls_score = (gs + ls) / 2.

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
