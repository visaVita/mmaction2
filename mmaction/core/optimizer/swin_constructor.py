from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor, OPTIMIZERS
import warnings

import torch
from torch.nn import GroupNorm, LayerNorm

from mmcv.utils import _BatchNorm, _InstanceNorm, build_from_cfg, is_list_of
from mmcv.utils.ext_loader import check_ops_exist

@OPTIMIZER_BUILDERS.register_module()
class swin_constructor(DefaultOptimizerConstructor):
    def add_params(self, params, model):
        
        ParamDict = dict()
        ParamDict['lr'] = dict()
        ParamDict['lr']['zero_decay'] = list()
        ParamDict['lr']['normal_decay'] = list()
        ParamDict['lrp'] = dict()
        ParamDict['lrp']['zero_decay'] = list()
        ParamDict['lrp']['normal_decay'] = list()
        ParamDict['TranST'] = dict()
        ParamDict['TranST']['zero_decay'] = list()
        ParamDict['TranST']['normal_decay'] = list()

        for name, param in model.named_parameters():
            if 'backbone' in name:
                if 'absolute_pos_embed' in name:
                    ParamDict['lrp']['zero_decay'].append(param)
                elif 'norm' in name:
                    ParamDict['lrp']['zero_decay'].append(param)
                elif 'relative_position_bias_table' in name:
                    ParamDict['lrp']['zero_decay'].append(param)
                else:
                    ParamDict['lrp']['normal_decay'].append(param)
            
            else:
                if 'TranST' in name:
                    if 'norm' in name:
                        ParamDict['TranST']['zero_decay'].append(param)
                    elif 'pos_enc' in name:
                        ParamDict['TranST']['zero_decay'].append(param)
                    else:
                        ParamDict['TranST']['normal_decay'].append(param)
                else:
                    ParamDict['lr']['normal_decay'].append(param)

        model.cls_head.pos_enc_module.eval()

        param_sum = len(ParamDict['TranST']['zero_decay']) + \
                    len(ParamDict['TranST']['normal_decay']) + \
                    len(ParamDict['lr']['zero_decay']) + \
                    len(ParamDict['lr']['normal_decay']) + \
                    len(ParamDict['lrp']['zero_decay']) + \
                    len(ParamDict['lrp']['normal_decay'])

        print('====  ', len(ParamDict['TranST']['zero_decay']))
        print('====  ', len(ParamDict['TranST']['normal_decay']))
        print('====  ', len(ParamDict['lr']['zero_decay']))
        print('====  ', len(ParamDict['lr']['normal_decay']))
        print('====  ', len(ParamDict['lrp']['zero_decay']))
        print('====  ', len(ParamDict['lrp']['normal_decay']))
        print('====  ', len(list(model.parameters())))
        assert param_sum==len(list(model.parameters()))

        params.append({
            'params': ParamDict['lrp']['normal_decay'],
            'lr': self.base_lr*0.1,
            'weight_decay': self.base_wd
        })

        params.append({
            'params': ParamDict['lrp']['zero_decay'],
            'lr': self.base_lr*0.1,
            'weight_decay': 0.
        })

        params.append({
            'params': ParamDict['lr']['normal_decay'],
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })

        params.append({
            'params': ParamDict['lr']['zero_decay'],
            'lr': self.base_lr,
            'weight_decay': 0.
        })

        params.append({
            'params': ParamDict['TranST']['normal_decay'],
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })

        params.append({
            'params': ParamDict['TranST']['zero_decay'],
            'lr': self.base_lr,
            'weight_decay': 0.
        })
