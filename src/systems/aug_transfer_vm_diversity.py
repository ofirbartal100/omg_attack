import random
import torch.nn as nn
from dabs.src.systems.aug_transfer_vm import ViewmakerTransferSystem


class ViewmakerTransferSystemDiversity(ViewmakerTransferSystem):

    def __init__(self, config):
        super().__init__(config)

    def setup_vm(self,config):
        self.viewmakers = nn.ModuleList([ self.load_viewmaker_from_checkpoint(ckpt,config.viewmaker.config_path) for ckpt in config.vm_ckpt])
        
    def view(self, imgs, with_unnormalized=False):
        if 'Expert' in self.config.system:
            raise RuntimeError('Cannot call self.view() with Expert system')
        unnormalized = random.choice(self.viewmakers)(imgs)
        views = self.normalize(unnormalized)
        # views = self.normalize(unnormalized)
        if with_unnormalized:
            return views, unnormalized
        return views