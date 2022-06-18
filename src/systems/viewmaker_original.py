import torch
import torch.nn as nn
from collections import OrderedDict

import PIL.Image as Image
from torchvision.transforms.functional import resize
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid
import wandb
import random
import matplotlib.pyplot as plt
from viewmaker.src.gans.tiny_pix2pix import TinyP2PDiscriminator

from dabs.src.datasets import natural_images
from dabs.src.systems.base_system import BaseSystem, get_model

from original_vm.viewmaker.src.models.viewmaker import Viewmaker
from viewmaker.src.objectives.memory_bank import MemoryBank
from viewmaker.src.systems.image_systems.utils import heatmap_of_view_effect
from viewmaker.src.objectives.adversarial import AdversarialSimCLRLoss
from viewmaker.src.utils import utils

class OriginalViewmakerSystem(BaseSystem):
    '''System for Shuffled Embedding Detection.

    Permutes 15% of embeddings within an example.
    Objective is to predict which embeddings were replaced.
    '''

    def __init__(self, config):
        self.disp_amnt = 10
        self.logging_steps = 100
        super().__init__(config)

    def setup(self, stage):
        super().setup(self)
        self.viewmaker = self.create_viewmaker(name='VM')
        sampled_data_size = len(self.train_dataset_ss) # for parital % of data
        self.memory_bank = MemoryBank(sampled_data_size, 128)
        self.memory_bank_labels = MemoryBank(sampled_data_size, 1, dtype=int)
    
    def forward(self, x, prehead=False):
        x[0] = self.normalize(x[0])
        return self.model.forward(x, prehead=prehead)

    def make_views(self, batch):
        indices, img, labels = batch
        views1, unnormalized_view1 = self.view(img, True)
        views2, unnormalized_view2 = self.view(img, True)
        emb_dict = {
            'indices': indices,
            'originals': img,#.cpu(),
            'views1': views1,
            'unnormalized_view1': unnormalized_view1,#.cpu(),
            'views2': views2,
            'unnormalized_view2': unnormalized_view2,#.cpu(),
            'labels': labels
        }
        return emb_dict

    def ssl_forward(self, batch):
        emb_dict = self.make_views(batch)
        emb_dict.update({'view1_embs': self.model.forward([emb_dict["views1"]]),
                         'view2_embs': self.model.forward([emb_dict["views2"]]), })
        return emb_dict

    def training_step(self, batch, batch_idx, optimizer_idx):
        emb_dict = self.ssl_forward(batch)
        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = self.objective(emb_dict)
        emb_dict['optimizer_idx'] = torch.tensor(optimizer_idx, device=self.device)

        if optimizer_idx == 0:
            metrics = {'encoder_loss': encoder_loss, "train_acc": encoder_acc, "positive_sim": positive_sim,
                       "negative_sim": negative_sim}
            loss = encoder_loss*0
        elif optimizer_idx == 1:
            metrics = {'view_maker_loss': view_maker_loss}
            loss = view_maker_loss
        else:
            loss = None
            metrics = None

        return [loss, emb_dict, metrics]

    def training_step_end(self, train_step_outputs):
        loss, emb_dict, metrics = train_step_outputs
        # reduce distributed results
        loss = loss.mean()
        metrics = {k: v.mean() for k, v in metrics.items()}
        self.wandb_logging(emb_dict)
        self.log_dict(metrics)

        self.add_to_memory_bank(emb_dict)

        return loss

    def objective(self, emb_dict):
        view_maker_loss_weight = self.config.loss_params.view_maker_loss_weight
        loss_function = AdversarialSimCLRLoss(
            embs1=emb_dict['view1_embs'],
            embs2=emb_dict['view2_embs'],
            t=self.config.loss_params.t,
            view_maker_loss_weight=view_maker_loss_weight
        )

        return loss_function.get_loss()

    def validation_step(self, batch, batch_idx):
        emb_dict = self.ssl_forward(batch)
        labels = batch[-1]
        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = self.objective(emb_dict)
        knn_correct, knn_total = self.get_nearest_neighbor_label(emb_dict['view1_embs'], labels)

        output = OrderedDict({
            'val_encoder_loss': encoder_loss,
            'val_view_maker_loss': view_maker_loss,
            'val_knn_acc': torch.tensor(knn_correct/knn_total, dtype=float, device=self.device),
            'val_knn_correct': torch.tensor(knn_correct, dtype=float, device=self.device),
            'val_knn_total': torch.tensor(knn_total, dtype=float, device=self.device),
            'val_positive_sim': positive_sim,
            'val_negative_sim': negative_sim,
        })
        return output

    def validation_step_end(self, output):
        self.log_dict(output)
        return output

    def add_to_memory_bank(self, emb_dict):
        with torch.no_grad():
            new_data_memory = utils.l2_normalize(emb_dict['view1_embs'].detach(), dim=1)
            self.memory_bank.update(emb_dict['indices'], new_data_memory)
            self.memory_bank_labels.update(emb_dict['indices'], emb_dict['labels'].unsqueeze(1))

    def get_nearest_neighbor_label(self, embs, labels):
        '''
        Used for online kNN classifier.
        For each image in validation, find the nearest image in the
        training dataset using the memory bank. Assume its label as
        the predicted label.
        '''

        all_dps = self.memory_bank.get_all_dot_products(embs)
        _, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
        neighbor_idxs = neighbor_idxs.squeeze(1)

        neighbor_labels = self.memory_bank_labels.at_idxs(neighbor_idxs).squeeze(-1)
        num_correct = torch.sum(neighbor_labels.cpu() == labels.cpu()).item()

        return num_correct, embs.size(0)

    def configure_optimizers(self):
        

        view_optim_name = self.config.optim_params.get("viewmaker_optim")
        view_parameters = self.viewmaker.parameters()
        if view_optim_name == 'adam':
            view_optim = torch.optim.Adam(view_parameters,
                                          lr=self.config.optim_params.get("viewmaker_learning_rate", 0.001),
                                          weight_decay=self.config.optim_params.weight_decay)
        elif not view_optim_name or view_optim_name == 'sgd':
            view_optim = torch.optim.SGD(
                view_parameters,
                lr=self.config.optim_params.viewmaker_learning_rate or self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        else:
            raise ValueError(f'Optimizer {view_optim_name} not implemented')


        enc_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(enc_params) == 0:
            encoder_optim = view_optim
        elif self.config.optim.name == 'adam':
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

        opt_list = [encoder_optim, view_optim]
        return opt_list, []

    def get_optimizer_index(self, emb_dict):
        if emb_dict['optimizer_idx'].__class__ == int or emb_dict['optimizer_idx'].dim() == 0:
            optimizer_idx = emb_dict['optimizer_idx']
        else:
            optimizer_idx = emb_dict['optimizer_idx'][0]
        return optimizer_idx

    def wandb_logging(self, emb_dict):
        with torch.no_grad():

            # check optimizer index to log images only once
            # # Handle Tensor (dp) and int (ddp) cases
            optimizer_idx = self.get_optimizer_index(emb_dict)
            if optimizer_idx > 0:
                return

            if self.global_step % self.logging_steps == 0:
                img = emb_dict['originals'][:self.disp_amnt]
                unnormalized_view1 = emb_dict['unnormalized_view1'][:self.disp_amnt]
                unnormalized_view2 = emb_dict['unnormalized_view2'][:self.disp_amnt]

                diff_heatmap = heatmap_of_view_effect(img, unnormalized_view1)
                diff_heatmap2 = heatmap_of_view_effect(img, unnormalized_view2)
                cat = torch.cat(
                    [img,
                     unnormalized_view1,
                     unnormalized_view2,
                     diff_heatmap,
                     diff_heatmap2,
                     (diff_heatmap - diff_heatmap2).abs()])

                if cat.shape[1] > 3:
                    cat = cat.mean(1).unsqueeze(1)

                grid = make_grid(cat, nrow=self.disp_amnt)
                grid = resize(torch.clamp(grid, 0, 1.0), (6*150, 10*150), Image.NEAREST)


                if isinstance(self.logger, WandbLogger):
                    wandb.log({
                        "original_vs_views": wandb.Image(grid, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}"),
                        "mean distortion": (unnormalized_view1 - img).abs().mean(),
                    })
                else:
                    self.logger.experiment.add_image('original_vs_views', grid, self.global_step)

    def create_viewmaker(self, **kwargs):
        view_model = Viewmaker(
            num_channels=self.train_dataset.IN_CHANNELS,
            distortion_budget=self.config.model_params.get("additive_budget", 0.05),
            clamp=False
        )
        return view_model

    def view(self, imgs, with_unnormalized=False):
        if 'Expert' in self.config.system:
            raise RuntimeError('Cannot call self.view() with Expert system')
        unnormalized = self.viewmaker(imgs)
        views = self.normalize(unnormalized)
        if with_unnormalized:
            return views, unnormalized
        return views

    def normalize(self, imgs):
        # These numbers were computed using compute_image_dset_stats.py
        if hasattr(self.train_dataset, "normalize"):
            return self.train_dataset.normalize(imgs)
        elif 'audioMNIST' in self.config.dataset:
            mean = torch.tensor([0.2701, 0.6490, 0.5382], device=imgs.device)
            std = torch.tensor([0.2230, 0.1348, 0.1449], device=imgs.device)
        else:
            raise ValueError(f'Dataset normalizer for {self.config.dataset} not implemented')
        imgs = (imgs - mean[None, :, None, None]) / std[None, :, None, None]
        return imgs

    def unnormalize(self, imgs):
        # These numbers were computed using compute_image_dset_stats.py
        if hasattr(self.train_dataset, "unnormalize"):
            return self.train_dataset.unnormalize(imgs)
        elif 'cifar' in self.config.dataset:
            mean = torch.tensor([0.491, 0.482, 0.446], device=imgs.device)
            std = torch.tensor([0.247, 0.243, 0.261], device=imgs.device)
        elif 'ffhq' in self.config.dataset:
            mean = torch.tensor([0.5202, 0.4252, 0.3803], device=imgs.device)
            std = torch.tensor([0.2496, 0.2238, 0.2210], device=imgs.device)
        elif 'audioMNIST' in self.config.dataset:
            mean = torch.tensor([0.2701, 0.6490, 0.5382], device=imgs.device)
            std = torch.tensor([0.2230, 0.1348, 0.1449], device=imgs.device)
        else:
            raise ValueError(f'Dataset normalizer for {self.config.dataset} not implemented')
        imgs = (imgs * std[None, :, None, None]) + mean[None, :, None, None]
        return imgs

class ViewmakerOriginalSystemDisc(OriginalViewmakerSystem):

    def setup(self, stage):
        super().setup(stage)
        self.disc = TinyP2PDiscriminator(in_channels=self.dataset.spec()[0].in_channels,
                                         wgan=self.config.disc.wgan,
                                         blocks_num=self.config.disc.conv_blocks)

    def training_step(self, batch, batch_idx, optimizer_idx):
        step_output = {'optimizer_idx': torch.tensor(optimizer_idx, device=self.device)}
        step_output.update(self.ssl_forward(batch))

        if optimizer_idx in [0, 1]:
            # step_output.update(self.ssl_forward(batch))
            encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = self.objective(step_output)

        if optimizer_idx in [1, 2]:
            step_output.update(self.gan_forward(batch, step_output, optimizer_idx))
            disc_loss, disc_acc, gen_loss, gen_acc, r1_reg = self.gan_objective(step_output)

        if optimizer_idx == 0:
            loss = encoder_loss
            metrics = {'encoder_loss': encoder_loss, "train_acc": encoder_acc, "positive_sim": positive_sim,
                       "negative_sim": negative_sim}

        elif optimizer_idx == 1:
            loss = self.get_vm_loss_weight() * view_maker_loss + self.config.disc.adv_loss_weight * gen_loss
            metrics = {'view_maker_loss': view_maker_loss,
                       'generator_loss': gen_loss,
                       'view_maker_total_loss': loss}

        elif optimizer_idx == 2:
            loss = disc_loss + self.config.disc.r1_penalty_weight * r1_reg  # ==0
            metrics = {'disc_acc': disc_acc,
                       'disc_loss': disc_loss,
                       'disc_r1_penalty': r1_reg}

        return [loss, step_output, metrics]

    def gan_forward(self, batch, step_output, optimizer_idx=2):
        indices, img, _ = batch
        # if "views1" not in step_output:
        #     views1, unnormalized_view1 = self.view(img, True)
        #     views2, unnormalized_view2 = self.view(img, True)
        #     step_output.update({'indices': indices,
        #                         'originals': img,
        #                         'views1': views1,
        #                         'unnormalized_view1': unnormalized_view1,
        #                         'views2': views2,
        #                         'unnormalized_view2': unnormalized_view2})
        views1, views2 = step_output["views1"], step_output["views2"]

        img.requires_grad = True
        step_output['disc_r1_penalty'] = torch.tensor([0.0], device=self.device)
        if optimizer_idx == 2:
            step_output["real_score"] = self.disc(self.normalize(img))
            if self.disc.wgan:
                try:
                    step_output["disc_r1_penalty"] = self.disc.r1_penalty(step_output["real_score"], img)
                # this fails in validation mode
                except RuntimeError as e:
                    pass
        step_output["fake_score"] = torch.cat([self.disc(views1), self.disc(views2)], dim=0)
        return step_output

    def gan_objective(self, emb_dict):
        real_s = emb_dict.get('real_score')
        fake_s = emb_dict['fake_score']
        loss_n_acc = self.disc.calc_loss_and_acc(real_s, fake_s,
                                                 r1_penalty=emb_dict['disc_r1_penalty'])
        disc_loss = loss_n_acc.get("d_loss")
        disc_acc = loss_n_acc.get("d_acc")
        r1_reg = emb_dict.get('disc_r1_penalty')
        gen_loss = loss_n_acc.get("g_loss")
        gen_acc = loss_n_acc.get("g_acc")
        return disc_loss, disc_acc, gen_loss, gen_acc, r1_reg

    def get_vm_loss_weight(self):
        if self.current_epoch < self.config.disc.gan_warmup:
            return 0
        else:
            return 1

    def validation_step(self, batch, batch_idx):
        step_output = self.ssl_forward(batch)
        step_output.update(self.gan_forward(batch, step_output))
        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = self.objective(step_output)
        disc_loss, disc_acc, gen_loss, gen_acc, r1_reg = self.gan_objective(step_output)
        
        labels = batch[-1]
        knn_correct, knn_total = self.get_nearest_neighbor_label(step_output['view1_embs'], labels)

        output = OrderedDict({
            'val_encoder_loss': encoder_loss,
            'val_view_maker_loss': view_maker_loss,
            'val_knn_acc': torch.tensor(knn_correct/knn_total, dtype=float, device=self.device),
            'val_knn_correct': torch.tensor(knn_correct, dtype=float, device=self.device),
            'val_knn_total': torch.tensor(knn_total, dtype=float, device=self.device),
            'val_positive_sim': positive_sim,
            'val_negative_sim': negative_sim,
            'val_disc_loss': disc_loss,
            'val_disc_acc': disc_acc,
            'val_generator_loss': gen_loss,
            'val_encoder_acc': encoder_acc
        })

        return output

    def on_validation_model_eval(self) -> None:
        """Sets the model to eval during the val loop."""

        self.model.eval()
        self.disc.train()
        self.viewmaker.train()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False)
        if optimizer_idx == 2:
            if batch_idx % (self.config.disc.dis_skip_steps + 1) == 0:
                super(ViewmakerOriginalSystemDisc, self).optimizer_step(epoch, batch_idx, optimizer, optimizer_idx,
                                                            optimizer_closure, on_tpu,
                                                            using_native_amp, using_lbfgs)
            else:
                optimizer_closure()

    def configure_optimizers(self):
        opt_list, [] = super().configure_optimizers()
        disc_optim = torch.optim.Adam(self.disc.parameters(),lr=self.config.disc.lr)
        opt_list.append(disc_optim)
        return opt_list, []