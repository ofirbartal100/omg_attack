import os.path
import random
from glob import glob

import torch
import torch.nn as nn
from dabs.src.systems.aug_transfer_vm import ViewmakerTransferSystem


class ViewmakerTransferSystemDiversity(ViewmakerTransferSystem):

    def __init__(self, config):
        super().__init__(config)

    def setup_vm(self, config):
        vm_ckpts = glob(config.vm_ckpt + "/epoch*.ckpt")
        self.viewmakers = nn.ModuleList(
            [self.load_viewmaker_from_checkpoint(ckpt, config.viewmaker.config_path) for ckpt in vm_ckpts])

    def view(self, imgs, with_unnormalized=False):
        if 'Expert' in self.config.system:
            raise RuntimeError('Cannot call self.view() with Expert system')
        p=0
        unnormalized = imgs
        views = self.normalize(unnormalized)
        while p <= self.config.viewmaker.reroll_prob:
            unnormalized = random.choice(self.viewmakers)(unnormalized)
            views = self.normalize(unnormalized)
            p = torch.rand(1).item()
        # views = self.normalize(unnormalized)
        if with_unnormalized:
            return views, unnormalized
        return views
