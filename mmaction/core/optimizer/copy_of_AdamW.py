from mmcv.runner import OPTIMIZERS
from torch.optim import AdamW

@OPTIMIZERS.register_module()
class CopyOfAdamW(AdamW):
    """ A clone of torch.optim.AdamW. """