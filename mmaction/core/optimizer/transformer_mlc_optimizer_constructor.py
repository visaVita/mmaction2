import torch
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.utils import SyncBatchNorm, _BatchNorm, _ConvNd


@OPTIMIZER_BUILDERS.register_module()
class transformer_mlc_optimizer_constructor(DefaultOptimizerConstructor):
    def add_params(self, params, model):
        params.append({
            'params': model.backbone.parameters(),
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        params.append({
            'params': model.backbone.parameters(),
            'lr': self.base_lr*10,
            'weight_decay': self.base_wd
        })
