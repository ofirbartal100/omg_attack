import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
import PIL.Image as Image
from torchvision.transforms.functional import resize
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import wandb
import random
import matplotlib.pyplot as plt
from viewmaker.src.gans.tiny_pix2pix import Discriminator, TinyP2PDiscriminator

from dabs.src.datasets import natural_images
from dabs.src.systems.base_system import BaseSystem, get_model

from original_vm.viewmaker.src.models.viewmaker import Viewmaker
from viewmaker.src.objectives.memory_bank import MemoryBank
from viewmaker.src.systems.image_systems.utils import heatmap_of_view_effect
from viewmaker.src.objectives.adversarial import AdversarialSimCLRLoss
from viewmaker.src.utils import utils
from dabs.src.models.conditional_viewmaker import ConditionalViewmaker

from stylegan2_pytorch.model import Discriminator as StyleGan2Disc
from itertools import chain

class OriginalViewmakerSystem(BaseSystem):
    '''System for Shuffled Embedding Detection.

    Permutes 15% of embeddings within an example.
    Objective is to predict which embeddings were replaced.
    '''

    def __init__(self, config):
        self.disp_amnt = 8
        self.logging_steps = 100
        if config.debug:
            self.logging_steps = 10
        super().__init__(config)

    def setup(self, stage):
        super().setup(self)
        self.viewmaker = self.create_viewmaker(name='VM')
        if stage == 'face_mask':
            self.viewmaker.with_mask=True

        sampled_data_size = len(self.train_dataset_ss) # for parital % of data
        self.memory_bank = MemoryBank(sampled_data_size, self.config.embd_dim)
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
        metrics = {k: v.mean().detach().cpu() for k, v in metrics.items()}
        self.wandb_logging(emb_dict)
        self.log_dict(metrics)

        self.add_to_memory_bank(emb_dict)

        return {'loss':loss } #,'encoder_acc':metrics['encoder_acc']}

    def objective(self, emb_dict):
        view_maker_loss_weight = self.config.loss_params.view_maker_loss_weight
        loss_function = AdversarialSimCLRLoss(
            embs1=emb_dict['view1_embs'],
            embs2=emb_dict['original_permutated_embs'],
            t=self.config.loss_params.t,
            view_maker_loss_weight=view_maker_loss_weight
        )

        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = loss_function.get_loss()
        return encoder_loss, encoder_acc, -1*view_maker_loss, positive_sim, negative_sim

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
            clamp=True
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
        loss_n_acc = self.disc.calc_loss_and_acc(real_s, fake_s, r1_penalty=emb_dict['disc_r1_penalty'])
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

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False):
    #     super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False)
    #     if optimizer_idx == 2:
    #         if batch_idx % (self.config.disc.dis_skip_steps + 1) == 0:
    #             super(ViewmakerOriginalSystemDisc, self).optimizer_step(epoch, batch_idx, optimizer, optimizer_idx,
    #                                                         optimizer_closure, on_tpu,
    #                                                         using_native_amp, using_lbfgs)
    #         else:
    #             optimizer_closure()

    def configure_optimizers(self):
        opt_list, [] = super().configure_optimizers()
        disc_optim = torch.optim.Adam(self.disc.parameters(),lr=self.config.disc.lr)
        opt_list.append(disc_optim)
        return opt_list, []


from types import MethodType

class CevaViewmakerSystem(OriginalViewmakerSystem):

    def setup(self, stage):
        OriginalViewmakerSystem.setup(self,'no_face_mask')

        self.disc = StyleGan2Disc(size=256,channel_multiplier=2)
        checkpoint = torch.load("/workspace/stylegan2_pytorch/checkpoint/550000.pt")
        self.disc.load_state_dict(checkpoint["d"],strict=True)
        self.transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    
    def gan_objective(self, emb_dict):
        fake_s = emb_dict['fake_score']
        real_s = emb_dict['real_score']
        gen_loss = F.softplus(-fake_s).mean()
        disc_loss = F.softplus(fake_s).mean() +  F.softplus(-real_s).mean() 
        return disc_loss, gen_loss,  fake_s.mean() ,  real_s.mean()

    def objective(self, emb_dict):
        view_maker_loss_weight = self.config.loss_params.view_maker_loss_weight
        original_embds = self.model.forward([self.normalize(emb_dict["originals"])])

        loss_function2 = AdversarialSimCLRLoss(
            embs1=original_embds,
            embs2=emb_dict['view2_embs'],
            t=self.config.loss_params.t,
            view_maker_loss_weight=view_maker_loss_weight
        ).get_loss()

        loss_function1 = AdversarialSimCLRLoss(
            embs1=original_embds,
            embs2=emb_dict['view1_embs'],
            t=self.config.loss_params.t,
            view_maker_loss_weight=view_maker_loss_weight
        ).get_loss()

        loss_function3 = AdversarialSimCLRLoss(
            embs1=emb_dict['view1_embs'],
            embs2=emb_dict['view2_embs'],
            t=self.config.loss_params.t,
            view_maker_loss_weight=view_maker_loss_weight
        ).get_loss()

        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = ( (i+j+k) / 3 for i,j,k in zip(loss_function1,loss_function2,loss_function3))
        positive_similarity_of_views = loss_function3[3]
        positive_sim = (loss_function1[3] + loss_function2[3])/2

        return encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim , positive_similarity_of_views

    def validation_step(self, batch, batch_idx):
        step_output = self.ssl_forward(batch)
        step_output.update(self.gan_forward(batch, step_output))
        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim,positive_similarity_of_views = self.objective(step_output)
        disc_loss , gen_loss, fake_score , real_score = self.gan_objective(step_output)
        
        output = OrderedDict({
            'val_encoder_loss': encoder_loss,
            'val_view_maker_loss': view_maker_loss,
            'val_positive_sim': positive_sim,
            'val_negative_sim': negative_sim,
            'positive_similarity_of_views': positive_similarity_of_views,
            'val_gen_loss': gen_loss,
            'val_disc_loss': disc_loss,
            'val_fake_score': fake_score,
            'val_real_score': real_score,
            'val_encoder_acc': encoder_acc
        })

        return output

    # def training_step(self, batch, batch_idx):
    def training_step(self, batch, batch_idx, optimizer_idx):
        step_output = {'optimizer_idx': torch.tensor(optimizer_idx, device=self.device)}
        # step_output = {'optimizer_idx': torch.tensor(0, device=self.device)}
        step_output.update(self.ssl_forward(batch))

        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim , positive_similarity_of_views = self.objective(step_output)
        step_output.update(self.gan_forward(batch, step_output, optimizer_idx))
        # step_output.update(self.gan_forward(batch, step_output))
        disc_loss , gen_loss, fake_score , real_score = self.gan_objective(step_output)

        # loss = view_maker_loss + self.config.disc.adv_loss_weight * gen_loss
        if optimizer_idx == 0:
            loss = self.config.disc.enc_loss_weight * view_maker_loss

        elif optimizer_idx == 1:
            loss = self.config.disc.adv_loss_weight_gen * gen_loss

        elif optimizer_idx == 2:
            loss = self.config.disc.adv_loss_weight * disc_loss



        metrics = {'view_maker_loss': view_maker_loss,
                    'generator_loss': gen_loss,
                    'disc_loss': disc_loss,
                    'positive_sim': positive_sim,
                    'negative_sim': negative_sim,
                    'positive_similarity_of_views': positive_similarity_of_views,
                    'fake_score': fake_score,
                    'real_score': real_score,
                    'view_maker_total_loss': loss}


        return [loss, step_output, metrics]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False):
        if optimizer_idx == 0: # dont optimize encoder

            if self.trainer.global_step % self.config.enc_every == 0 and self.trainer.global_step > self.config.start_enc:
                super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False , using_native_amp=False, using_lbfgs=False)
            else:
                optimizer_closure()
            # if self.trainer.global_step < 10000: # start with only the discriminator, to help the budget
            #     optimizer_closure()
            # else:
            #     super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False)
        #     if self.trainer.global_step % 3 == 0: # do less optimization steps on the viewmaker, and focus more on the discriminator
        #         # call the closure by itself to run `training_step` + `backward` without an optimizer step
        #         optimizer_closure()
        #     else:
        #         optimizer_closure()
        #         # super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False)
        else:
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False)
        #super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False)

    def gan_forward(self, batch, step_output, optimizer_idx=1):
    # def gan_forward(self, batch, step_output):
        indices, img, _ = batch
        views1 = self.transform_for_disc(step_output["views1"],isview=True)
        views2 = self.transform_for_disc(step_output["views2"],isview=True)

        # img.requires_grad = True
        step_output["fake_score"] = torch.cat([self.disc(views1), self.disc(views2)], dim=0)
        step_output["real_score"] = self.disc(self.transform_for_disc(img))
        return step_output

    def transform_for_disc(self,img,isview=False):
        if isview:
            x = self.unnormalize(img)
        else:
            x = img

        return self.transform(x)

    def configure_optimizers(self):
        view_optim_name = self.config.optim_params.get("viewmaker_optim")
        if view_optim_name == 'adam':
            view_optim1 = torch.optim.Adam(self.viewmaker.parameters(),
                                          lr=self.config.optim_params.get("viewmaker_learning_rate", 0.001),
                                          weight_decay=self.config.optim_params.weight_decay)
            
            view_optim2 = torch.optim.Adam(self.viewmaker.parameters(),
                                          lr=self.config.optim_params.get("viewmaker_learning_rate", 0.001),
                                          weight_decay=self.config.optim_params.weight_decay)
        elif not view_optim_name or view_optim_name == 'sgd':
            view_optim1 = torch.optim.SGD(
                self.viewmaker.parameters(),
                lr=self.config.optim_params.viewmaker_learning_rate or self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )

            view_optim2 = torch.optim.SGD(
                self.viewmaker.parameters(),
                lr=self.config.optim_params.viewmaker_learning_rate or self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        else:
            raise ValueError(f'Optimizer {view_optim_name} not implemented')

        disc_optim = torch.optim.Adam(self.disc.parameters(),lr=self.config.disc.lr)

        return [view_optim1, view_optim2,disc_optim] , []

class TrafficViewMaker(OriginalViewmakerSystem):


    def setup(self, stage):
        OriginalViewmakerSystem.setup(self,stage)

        self.disc = TinyP2PDiscriminator(in_channels=self.dataset.spec()[0].in_channels,
                                         wgan=self.config.disc.wgan,
                                         blocks_num=self.config.disc.conv_blocks)

    
    def objective(self, emb_dict):
        view_maker_loss_weight = self.config.loss_params.view_maker_loss_weight
        original_embds = self.model.forward([emb_dict["originals"]])

        loss_function2 = AdversarialSimCLRLoss(
            embs1=original_embds,
            embs2=emb_dict['view2_embs'],
            t=self.config.loss_params.t,
            view_maker_loss_weight=view_maker_loss_weight
        ).get_loss()

        loss_function1 = AdversarialSimCLRLoss(
            embs1=original_embds,
            embs2=emb_dict['view1_embs'],
            t=self.config.loss_params.t,
            view_maker_loss_weight=view_maker_loss_weight
        ).get_loss()

        loss_function3 = AdversarialSimCLRLoss(
            embs1=emb_dict['view1_embs'],
            embs2=emb_dict['view2_embs'],
            t=self.config.loss_params.t,
            view_maker_loss_weight=view_maker_loss_weight
        ).get_loss()

        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = ( (i+j+k) / 3 for i,j,k in zip(loss_function1,loss_function2,loss_function3))
        # encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = loss_function1
        positive_similarity_of_views = loss_function3[3]
        positive_sim = (loss_function1[3] + loss_function2[3])/2

        o = self.model.traffic_model.forward_original(emb_dict["originals"]).max(1, keepdim=True)[1]
        v1 = self.model.traffic_model.forward_original(emb_dict["views1"]).max(1, keepdim=True)[1]
        v2 = self.model.traffic_model.forward_original(emb_dict["views2"]).max(1, keepdim=True)[1]

        emb_dict['model_original'] = o
        emb_dict['model_view1'] = v1
        emb_dict['model_view2'] = v2

        encoder_acc = ((o==v1).sum() + (o==v2).sum())/(2*len(o))
        return encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim , positive_similarity_of_views

    def validation_step(self, batch, batch_idx):
        step_output = self.ssl_forward(batch)
        step_output['labels'] = batch[-1]

        step_output.update(self.gan_forward(batch, step_output))
        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim,positive_similarity_of_views = self.objective(step_output)
        disc_loss, disc_acc, gen_loss, gen_acc, r1_reg = self.gan_objective(step_output)
        
        output = OrderedDict({
            # 'val_encoder_loss': encoder_loss,
            'val_view_maker_loss': view_maker_loss,
            'val_positive_sim': positive_sim,
            'val_negative_sim': negative_sim,
            'positive_similarity_of_views': positive_similarity_of_views,
            'val_gen_loss': gen_loss,
            'val_disc_loss': disc_loss,
            'val_gen_acc': gen_acc,
            'val_disc_acc': disc_acc,
            # 'val_fake_score': fake_score,
            # 'val_real_score': real_score,
            'val_encoder_acc': encoder_acc
        })

        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        step_output = {'optimizer_idx': torch.tensor(optimizer_idx, device=self.device)}
        # step_output = {'optimizer_idx': torch.tensor(0, device=self.device)}
        step_output.update(self.ssl_forward(batch))

        step_output['labels'] = batch[-1]

        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim , positive_similarity_of_views = self.objective(step_output)
        step_output.update(self.gan_forward(batch, step_output, optimizer_idx))
        # step_output.update(self.gan_forward(batch, step_output))
        disc_loss, disc_acc, gen_loss, gen_acc, r1_reg = self.gan_objective(step_output)

        # loss = view_maker_loss + self.config.disc.adv_loss_weight * gen_loss
        if optimizer_idx == 0:
            loss = self.config.disc.enc_loss_weight * view_maker_loss

        elif optimizer_idx == 1:
            loss = self.config.disc.adv_loss_weight_gen * gen_loss

        elif optimizer_idx == 2:
            loss = self.config.disc.adv_loss_weight * disc_loss



        metrics = {'view_maker_loss': view_maker_loss,
                    'generator_loss': gen_loss,
                    'gen_acc': gen_acc,
                    'disc_loss': disc_loss,
                    'disc_acc': disc_acc,
                    'encoder_acc': encoder_acc,
                    'positive_sim': positive_sim,
                    'negative_sim': negative_sim,
                    'positive_similarity_of_views': positive_similarity_of_views,
                    # 'fake_score': fake_score,
                    # 'real_score': real_score,
                    'view_maker_total_loss': loss}


        return [loss, step_output, metrics]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False):
        # if optimizer_idx == 0: # dont optimize encoder
        #     # optimizer_closure()
        #     super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False)
        # else:
        #     optimizer_closure()
        #     # super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False)
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False)

    def gan_forward(self, batch, step_output, optimizer_idx=2):
        indices, img, _ = batch
        views1, views2 = step_output["views1"], step_output["views2"]

        img.requires_grad = True
        step_output['disc_r1_penalty'] = torch.tensor([0.0], device=self.device)
        step_output["real_score"] = self.disc(self.normalize(img))
        step_output["fake_score"] = torch.cat([self.disc(views1), self.disc(views2)], dim=0)
        return step_output

    def gan_objective(self, emb_dict):
        real_s = emb_dict.get('real_score')
        fake_s = emb_dict['fake_score']
        loss_n_acc = self.disc.calc_loss_and_acc(real_s, fake_s, r1_penalty=emb_dict['disc_r1_penalty'])
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

    
    def predict(self, x):
        return self.model.traffic_model.forward_original(self.normalize(x)).max(1, keepdim=True)[1]

    def configure_optimizers(self):
        view_optim_name = self.config.optim_params.get("viewmaker_optim")
        if view_optim_name == 'adam':
            view_optim1 = torch.optim.Adam(self.viewmaker.parameters(),
                                          lr=self.config.optim_params.get("viewmaker_learning_rate", 0.001),
                                          weight_decay=self.config.optim_params.weight_decay)
            
            view_optim2 = torch.optim.Adam(self.viewmaker.parameters(),
                                          lr=self.config.optim_params.get("viewmaker_learning_rate", 0.001),
                                          weight_decay=self.config.optim_params.weight_decay)
        elif not view_optim_name or view_optim_name == 'sgd':
            view_optim1 = torch.optim.SGD(
                self.viewmaker.parameters(),
                lr=self.config.optim_params.viewmaker_learning_rate or self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )

            view_optim2 = torch.optim.SGD(
                self.viewmaker.parameters(),
                lr=self.config.optim_params.viewmaker_learning_rate or self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        else:
            raise ValueError(f'Optimizer {view_optim_name} not implemented')

        disc_optim = torch.optim.Adam(self.disc.parameters(),lr=self.config.disc.lr)

        return [view_optim1, view_optim2,disc_optim] , []

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

    
    

class BirdsViewMaker(TrafficViewMaker):
    
    def setup(self, stage):
        OriginalViewmakerSystem.setup(self,stage)
        self.viewmaker = Viewmaker(
            num_channels=self.train_dataset.IN_CHANNELS,
            distortion_budget=self.config.model_params.get("additive_budget", 0.05),
            clamp=False,
            # downsample_to=112
        )

        # self.disc = TinyP2PDiscriminator(in_channels=self.dataset.spec()[0].in_channels,
        #                                  wgan=self.config.disc.wgan,
        #                                  blocks_num=self.config.disc.conv_blocks)

        # self.disc = UNetD(nc=self.dataset.spec()[0].in_channels)
        self.disc = Discriminator(input_shape=(3,224,224))


    def objective(self, emb_dict):
        view_maker_loss_weight = self.config.loss_params.view_maker_loss_weight
        original_embds = self.model.forward([self.normalize(emb_dict["originals"])])

        loss_function2 = AdversarialSimCLRLoss(
            embs1=original_embds,
            embs2=emb_dict['view2_embs'],
            t=self.config.loss_params.t,
            view_maker_loss_weight=view_maker_loss_weight
        ).get_loss()

        loss_function1 = AdversarialSimCLRLoss(
            embs1=original_embds,
            embs2=emb_dict['view1_embs'],
            t=self.config.loss_params.t,
            view_maker_loss_weight=view_maker_loss_weight
        ).get_loss()

        loss_function3 = AdversarialSimCLRLoss(
            embs1=emb_dict['view1_embs'],
            embs2=emb_dict['view2_embs'],
            t=self.config.loss_params.t,
            view_maker_loss_weight=view_maker_loss_weight
        ).get_loss()

        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = ( (i+j+k) / 3 for i,j,k in zip(loss_function1,loss_function2,loss_function3))
        # encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = loss_function1
        positive_similarity_of_views = loss_function3[3]
        positive_sim = (loss_function1[3] + loss_function2[3])/2


        
        # confused_orig = (emb_dict['model_original_pred'].flatten() == emb_dict['labels'].flatten()).flatten()[:self.disp_amnt]

        o,_ = self.model.birds_model(emb_dict["originals"])
        o_prob = F.softmax(o, dim = -1)
        o = o_prob.argmax(1, keepdim = True)

        v1,_ = self.model.birds_model(emb_dict["views1"])
        v1_prob = F.softmax(v1, dim = -1)
        v1 = v1_prob.argmax(1, keepdim = True)

        v2,_ = self.model.birds_model(emb_dict["views2"])
        v2_prob = F.softmax(v2, dim = -1)
        v2 = v2_prob.argmax(1, keepdim = True)

        emb_dict['model_original_pred'] = o
        emb_dict['model_view1_pred'] = v1
        emb_dict['model_view2_pred'] = v2

        encoder_acc = ((o==v1).sum() + (o==v2).sum())/(2*len(o))

        emb_dict['model_acc_top1'] , emb_dict['model_acc_top5'] = BirdsViewMaker.calculate_topk_accuracy(o_prob,emb_dict['labels'])
        emb_dict['view1_acc_top1'] , emb_dict['view1_acc_top5'] = BirdsViewMaker.calculate_topk_accuracy(v1_prob,emb_dict['labels'])
        emb_dict['view2_acc_top1'] , emb_dict['view2_acc_top5'] = BirdsViewMaker.calculate_topk_accuracy(v2_prob,emb_dict['labels'])

        return encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim , positive_similarity_of_views

   
    def predict(self, x):
        pred , _ = self.model.birds_model(self.normalize(x))
        y_prob = F.softmax(pred, dim = -1)
        pred = y_prob.argmax(1, keepdim = True)
        return pred

    @staticmethod
    def calculate_topk_accuracy(y_pred, y, k = 5):
        with torch.no_grad():
            batch_size = y.shape[0]
            _, top_pred = y_pred.topk(k, 1)
            top_pred = top_pred.t()
            correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
            correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
            acc_1 = correct_1 / batch_size
            acc_k = correct_k / batch_size
        return acc_1, acc_k

    @staticmethod
    def paint_border(image, is_similar, pad = 1):
        image[:,:pad] = 0
        image[:,-pad:] = 0
        image[:,:,:pad] = 0
        image[:,:,-pad:] = 0

        if is_similar:
            image[1,:pad] = 1
            image[1,-pad:] = 1
            image[1,:,:pad] = 1
            image[1,:,-pad:] = 1
        else :
            image[0,:pad] = 1
            image[0,-pad:] = 1
            image[0,:,:pad] = 1
            image[0,:,-pad:] = 1

        return image

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

                confused_orig = (emb_dict['model_original_pred'].flatten() == emb_dict['labels'].flatten()).flatten()[:self.disp_amnt]
                confused_v1 = (emb_dict['model_original_pred'] == emb_dict['model_view1_pred']).flatten()[:self.disp_amnt]
                confused_v2 = (emb_dict['model_original_pred'] == emb_dict['model_view2_pred']).flatten()[:self.disp_amnt]

                img_border = torch.cat([ BirdsViewMaker.paint_border(img[c],confused_orig[c],pad=5).unsqueeze(0)  for c in range(self.disp_amnt)],dim=0)
                unnormalized_view1_border = torch.cat([ BirdsViewMaker.paint_border(unnormalized_view1[c],confused_v1[c],pad=5).unsqueeze(0)  for c in range(self.disp_amnt)],dim=0)
                unnormalized_view2_border = torch.cat([ BirdsViewMaker.paint_border(unnormalized_view2[c],confused_v2[c],pad=5).unsqueeze(0)  for c in range(self.disp_amnt)],dim=0)


                cat = torch.cat(
                    [   img_border, #img,
                        # unnormalized_view1,
                        # unnormalized_view2,
                        unnormalized_view1_border,
                        unnormalized_view2_border,
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
                        "model_acc_top1": emb_dict['model_acc_top1'],
                        "model_acc_top5": emb_dict['model_acc_top5'],
                        "view_acc_top1": (emb_dict['view1_acc_top1']+emb_dict['view2_acc_top1'])/2,
                        "view_acc_top5": (emb_dict['view1_acc_top5']+emb_dict['view2_acc_top5'])/2,
                        "gen_every":self.config.gen_every
                    })
                else:
                    self.logger.experiment.add_image('original_vs_views', grid, self.global_step)
                    self.logger.experiment.add_scalar("model_acc_top1", emb_dict['model_acc_top1'], self.global_step)
                    self.logger.experiment.add_scalar("model_acc_top5", emb_dict['model_acc_top5'], self.global_step)
                    self.logger.experiment.add_scalar("view_acc_top1", (emb_dict['view1_acc_top1']+emb_dict['view2_acc_top1'])/2, self.global_step)
                    self.logger.experiment.add_scalar("view_acc_top5", (emb_dict['view1_acc_top5']+emb_dict['view2_acc_top5'])/2, self.global_step)

    def configure_optimizers(self):
        view_optim1 = torch.optim.Adam(self.viewmaker.parameters(),
                                        lr=self.config.optim_params.get("viewmaker_learning_rate", 0.001),
                                        weight_decay=self.config.optim_params.weight_decay)
        
        view_optim2 = torch.optim.Adam(self.viewmaker.parameters(),
                                        lr=self.config.optim_params.get("viewmaker_learning_rate", 0.001),
                                        weight_decay=self.config.optim_params.weight_decay)
        

        disc_optim = torch.optim.Adam(self.disc.parameters(),lr=self.config.disc.lr)

        return [view_optim1, view_optim2,disc_optim] , []

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,using_native_amp=False, using_lbfgs=False):
        if optimizer_idx == 1:
            if self.trainer.global_step % self.config.gen_every == 0 and self.trainer.global_step > self.config.start_enc:
                super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False , using_native_amp=False, using_lbfgs=False)
            else:
                optimizer_closure()

        else:
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False , using_native_amp=False, using_lbfgs=False)

    def training_step_end(self, train_step_outputs):
        loss, emb_dict, metrics = train_step_outputs
        # reduce distributed results
        loss = loss.mean()
        metrics = {k: v.mean().detach().cpu() for k, v in metrics.items()}
        self.wandb_logging(emb_dict)
        self.log_dict(metrics)

        self.add_to_memory_bank(emb_dict)

        return {'loss':loss ,'encoder_acc':metrics['encoder_acc']}

    def training_epoch_end(self, outputs) -> None:
        mean_encoder_acc = np.mean(list(chain(*[ [oo['encoder_acc'].item() for oo in o] for o in outputs])))
        if mean_encoder_acc>0.6:
            self.config.gen_every = 2
        elif mean_encoder_acc<0.3:
            self.config.gen_every = 1
