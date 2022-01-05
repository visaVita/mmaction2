from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from timm.models.registry import model_entrypoint
import torch.nn as nn

@OPTIMIZER_BUILDERS.register_module()
class freeze_backbone_constructor(DefaultOptimizerConstructor):
    def add_params(self, params, model):
        for name, param in model.backbone.named_parameters():
            #if "bn" in name:
            param.requires_grad = False

        # print([x for x in model.cls_head.modules()])
        # print([x for x in model.named_modules()])
        for layer in model.backbone.modules():
            #if isinstance(layer, nn.BatchNorm3d):
                # import ipdb
                # ipdb.set_trace()
            layer.eval()

        model.backbone.eval()


        ParamDict = dict()
        ParamDict['zero_decay'] = list()
        ParamDict['normal_decay'] = list()

        # for name, param in model.cls_head.named_parameters():
        #     if 'pos_enc' in name:
        #         param.requires_grad = False
        #     # elif name.endswith('bias'):
        #     #     ParamDict['zero_decay'].append(param)
        #     else:
        #         ParamDict['normal_decay'].append(param)

        # model.cls_head.pos_enc_module.eval()

        params.append({
            'params': list(model.cls_head.parameters()),
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })

        # params.append({
        #     'params': ParamDict['normal_decay'],
        #     'lr': self.base_lr,
        #     'weight_decay': self.base_wd
        # })

        # params.append({
        #     'params': ParamDict['zero_decay'],
        #     'lr': self.base_lr,
        #     'weight_decay': 0.
        # })

        # params.append({
        #     'params': ParamDict['TranST']['normal_decay'],
        #     'lr': self.base_lr,
        #     'weight_decay': self.base_wd
        # })

        # params.append({
        #     'params': ParamDict['TranST']['zero_decay'],
        #     'lr': self.base_lr,
        #     'weight_decay': 0.
        # })
