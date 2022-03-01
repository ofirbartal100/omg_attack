from collections import OrderedDict
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import ComposeTransform
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, GaussianBlur

from dabs.src.systems.base_system import BaseSystem
from dabs.src.systems.viewmaker import ViewmakerSystem
from viewmaker.src.datasets.datasets import load_default_transforms
from viewmaker.src.objectives.simclr import SimCLRObjective


class ExpertSystem(ViewmakerSystem):
    '''Implements the i-Mix algorithm on embedded inputs defined in https://arxiv.org/abs/2010.08887.

    Because there aren't predefined augmentations, i-Mix is applied to the original embeddings. The
    algorithm under default parameters can be summarized as

    Algorithm 1:
        lambda ~ Beta(1, 1)
        lambda = max(lambda, 1 - lambda)  # assures mixing coefficient >= 0.5

        embs = embed(*x)
        permuted_idxs = permute(arange(embs))
        permuted_embs = stop_gradient[embs][permuted_idx]
        mixed_embs = lambda * embs + (1 - lambda) * permuted_embs

        logits = mixed_embs @ embs.T
        contrastive_loss = cross_entropy(logits, arange(embs))
        mixed_virtual_loss = cross_entropy(logits, permuted_idxs)

        loss = contrastive_loss + mixed_virtual_loss
    '''

    ALPHA = 1.0
    TEMPERATURE = 0.2

    def __init__(self, config):
        super().__init__(config)
        self.train_transforms = self.expert_transforms()

    def expert_transforms(self):
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur((7, 7), [.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ])
        return train_transforms

    def setup(self, stage):
        super().setup(self)
        #### delete ####
        self.setup2()
        ################
        del self.model.embed_modules

    def training_step(self, batch, batch_idx):
        emb_dict = self.ssl_forward(batch)
        emb_dict["optimizer_idx"] = 0
        encoder_loss, encoder_acc, positive_sim, negative_sim = self.objective(emb_dict)

        metrics = {'encoder_loss': encoder_loss,
                   "train_acc": encoder_acc,
                   "positive_sim": positive_sim,
                   "negative_sim": negative_sim}
        loss = encoder_loss

        return [loss, emb_dict, metrics]

    def ssl_forward(self, batch, validation=False):
        emb_dict = self.make_views(batch)
        normed_imgs = self.normalize(emb_dict["originals"])
        if validation:
            emb_dict["views1"], emb_dict["views2"] = normed_imgs, normed_imgs
            emb_dict["unnormalized_view1"], emb_dict["unnormalized_view1"] = emb_dict["originals"], emb_dict[
                "originals"]
        emb_dict.update({'view1_embs': self.model.forward([emb_dict["views1"]]),
                         'view2_embs': self.model.forward([emb_dict["views2"]]),
                         'orig_embs': self.model.forward([normed_imgs])})
        return emb_dict

    def objective(self, emb_dict):
        loss_fn = SimCLRObjective(emb_dict['view1_embs'], emb_dict['view2_embs'], t=self.config.loss_params.t)
        loss, acc, pos_sim, neg_sim = loss_fn.get_loss_and_acc()
        return loss, acc, pos_sim, neg_sim

    def validation_step(self, batch, batch_idx):
        emb_dict = self.ssl_forward(batch, True)
        labels = batch[-1]
        encoder_loss, encoder_acc, positive_sim, negative_sim = self.objective(emb_dict)
        knn_acc = self.get_nearest_neighbor_label(emb_dict["orig_embs"], labels)

        output = OrderedDict({
            'val_encoder_loss': encoder_loss,
            'val_zero_knn_acc': torch.tensor(knn_acc, dtype=float, device=self.device),
            'val_positive_sim': positive_sim,
            'val_negative_sim': negative_sim,
        })
        return output

    # ########################

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu,
                               using_native_amp, using_lbfgs)

    def configure_optimizers(self):
        enc_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optim.name == 'adam':
            encoder_optim = torch.optim.AdamW(enc_params, lr=self.config.optim.lr,
                                              weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.name == 'sgd':
            encoder_optim = torch.optim.SGD(
                enc_params,
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                momentum=self.config.optim.momentum,
            )
        else:
            raise ValueError(f'{self.config.optim.name} optimizer unrecognized.')

        opt_list = [encoder_optim]

        return opt_list, []

    def create_viewmaker(self, **kwargs):
        return None

    def view(self, imgs, with_unnormalized=False):
        unnormalized = self.train_transforms(imgs)
        views = self.normalize(unnormalized)
        if with_unnormalized:
            return views, unnormalized
        return views