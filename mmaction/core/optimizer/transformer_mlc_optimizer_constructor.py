from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
import torch.nn as nn

@OPTIMIZER_BUILDERS.register_module()
class transformer_mlc_optimizer_constructor(DefaultOptimizerConstructor):
    def add_params(self, params, model):
        for name, param in model.backbone.named_parameters():
            if "bn" in name:
                param.requires_grad = False

        # print([x for x in model.cls_head.modules()])
        # print([x for x in model.named_modules()])
        for layer in model.backbone.modules():
            if isinstance(layer, nn.BatchNorm3d):
                # import ipdb
                # ipdb.set_trace()
                layer.eval()
        
        # model.backbone.eval()

        params.append({
            'params': list(model.backbone.parameters()),
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })

        params.append({
            'params': list(model.cls_head.parameters()),
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })

        """ params.append({
            'params': model.roi_head.parameters(),
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        }) """
