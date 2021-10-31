from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
import torch.nn as nn

@OPTIMIZER_BUILDERS.register_module()
class freeze_bn_constructor(DefaultOptimizerConstructor):
    def add_params(self, params, model):
        backbone_finetune_list = []
        for name, param in model.backbone.named_parameters():
            if "bn" in name:
                param.requires_grad = False
                continue
            backbone_finetune_list.append(param)
            

        # print([x for x in model.cls_head.modules()])
        # print([x for x in model.named_modules()])
        for layer in model.backbone.modules():
            if isinstance(layer, nn.BatchNorm3d):
                # import ipdb
                # ipdb.set_trace()
                layer.eval()

        # model.backbone.eval()
        params.append({
            'params': backbone_finetune_list,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })
        # print(backbone_finetune_list)
        # print(list(model.cls_head.parameters()))
        params.append({
            'params': list(model.cls_head.parameters()),
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })