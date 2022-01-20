import numpy as np
import torch
import torchmetrics
from collections import OrderedDict

from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid
import wandb

from dabs.src.systems.base_system import BaseSystem

from viewmaker.src.models.viewmaker import Viewmaker
from viewmaker.src.systems.image_systems.utils import heatmap_of_view_effect
from viewmaker.src.objectives.adversarial import AdversarialSimCLRLoss

class ViewmakerSystem(BaseSystem):
    '''System for Shuffled Embedding Detection.

    Permutes 15% of embeddings within an example.
    Objective is to predict which embeddings were replaced.
    '''

    def __init__(self, config):
        super().__init__(config)


    def setup(self, stage):
        super().setup(self)
        self.viewmaker = self.create_viewmaker()

    def forward(self, x, prehead=False):
        return self.model.forward(x, prehead=prehead)

    def ssl_forward(self, batch):
        indices, img,  _ = batch

        view1, unnormalized_view1 = self.view(img, True)
        view2, unnormalized_view2 = self.view(img, True)
        emb_dict = {
            'indices': indices,
            'view1_embs': self.model.forward([view1]),
            'view2_embs': self.model.forward([view2]),
            'orig_embs': self.model.forward([self.normalize(img)]),
            'originals': img,
            'views1': unnormalized_view1,
            'views2': unnormalized_view2
        }
        return emb_dict

    def training_step(self, batch, batch_idx, optimizer_idx):
        emb_dict = self.ssl_forward(batch)
        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = self.objective(emb_dict)

        # assuming they fixed taht in new versions
        # # Handle Tensor (dp) and int (ddp) cases
        if optimizer_idx.__class__ == int or optimizer_idx.dim() == 0:
            optimizer_idx = optimizer_idx
        else:
            optimizer_idx = optimizer_idx[0]

        if optimizer_idx == 0:
            metrics = {'encoder_loss': encoder_loss, "train_acc": encoder_acc, "positive_sim": positive_sim, "negative_sim": negative_sim}
            loss = encoder_loss
        elif optimizer_idx == 1:
            metrics = {'view_maker_loss': view_maker_loss}
            loss = view_maker_loss

        self.wandb_logging(emb_dict)
        self.log_dict(metrics)
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
        # num_correct, batch_size = self.get_nearest_neighbor_label(img_embs, labels)

        output = OrderedDict({
            'val_encoder_loss': encoder_loss,
            'val_view_maker_loss': view_maker_loss,
            # 'val_zero_knn_correct': torch.tensor(num_correct, dtype=float, device=self.device),
            # 'val_num_total': torch.tensor(batch_size, dtype=float, device=self.device),
            'val_positive_sim': positive_sim,
            'val_negative_sim': negative_sim,
            # 'val_linear_probe_score': probe_score,
        })
        return output

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # update viwmaker every step
        if optimizer_idx == 0:
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs)
        # freeze_after_epoch = self.config.optim_params.viewmaker_freeze_epoch and self.current_epoch > self.config.optim_params.viewmaker_freeze_epoch
        freeze_after_epoch = False

        if optimizer_idx == 1:
            if freeze_after_epoch:
                # freeze viewmaker after a certain number of epochs
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()
            else:
                super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optim.name == 'adam':
            encoder_optim = torch.optim.AdamW(params, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.name == 'sgd':
            encoder_optim = torch.optim.SGD(
                params,
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                momentum=self.config.optim.momentum,
            )
        else:
            raise ValueError(f'{self.config.optim.name} optimizer unrecognized.')

        view_optim_name = self.config.optim_params.viewmaker_optim
        view_parameters = self.viewmaker.parameters()
        if view_optim_name == 'adam':
            view_optim = torch.optim.Adam(view_parameters, lr=self.config.optim_params.viewmaker_learning_rate or 0.001)
        elif not view_optim_name or view_optim_name == 'sgd':
            view_optim = torch.optim.SGD(
                view_parameters,
                lr=self.config.optim_params.viewmaker_learning_rate or self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        else:
            raise ValueError(f'Optimizer {view_optim_name} not implemented')

        enc_list = [encoder_optim, view_optim]

        return enc_list, []

    def wandb_logging(self, emb_dict):
        logging_steps = 20
        if isinstance(self.logger, WandbLogger):
            logging_steps = 200

        if self.global_step % logging_steps == 0:
            amount_images = 10
            img = emb_dict['originals']
            unnormalized_view1 = emb_dict['views1']
            unnormalized_view2 = emb_dict['views2']

            diff_heatmap = heatmap_of_view_effect(img[:amount_images], unnormalized_view1[:amount_images])
            diff_heatmap2 = heatmap_of_view_effect(img[:amount_images], unnormalized_view2[:amount_images])
            if img.size(1) >3:
                img = img.mean(1, keepdim=True)
                unnormalized_view1 = unnormalized_view1.mean(1, keepdim=True)
                unnormalized_view2 = unnormalized_view2.mean(1, keepdim=True)
                diff_heatmap = diff_heatmap.mean(1, keepdim=True)
                diff_heatmap2 = diff_heatmap2.mean(1, keepdim=True)
            cat = torch.cat([img[:amount_images], unnormalized_view1[:amount_images], unnormalized_view2[:amount_images],
                            diff_heatmap, diff_heatmap2, (diff_heatmap-diff_heatmap2).abs()])
            grid = make_grid(cat, nrow=amount_images)
            grid = torch.clamp(grid, 0, 1.0)
            if isinstance(self.logger, WandbLogger):
                wandb.log({
                    "original_vs_views": wandb.Image(grid, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}"),
                    "mean distortion": (unnormalized_view1 - img).abs().mean(),
                })
            else:
                self.logger.experiment.add_image('original_vs_views', grid, self.global_step)

    def create_viewmaker(self):
        view_model = Viewmaker(
            num_channels=self.train_dataset.IN_CHANNELS,
            activation=self.config.model_params.generator_activation or 'relu',
            clamp=self.config.model_params.clamp_views or False,
            frequency_domain=self.config.model_params.spectral or False,
            downsample_to=self.config.model_params.viewmaker_downsample or False,
            num_res_blocks=self.config.model_params.num_res_blocks or 5,
            use_budget=self.config.model_params.use_budget or True,
            budget_aware=self.config.model_params.budget_aware or False,
            image_dim=(32, 32),
            multiplicative=self.config.model_params.multiplicative or 0,
            multiplicative_budget=self.config.model_params.multiplicative_budget or 0.25,
            additive=self.config.model_params.additive or 1,
            additive_budget=self.config.model_params.additive_budget or 0.05,
            tps=self.config.model_params.tps or 0,
            tps_budget=self.config.model_params.tps_budget or 0.1,
            aug_proba=self.config.model_params.aug_proba or 1,
        )
        return view_model

    def view(self, imgs, with_unnormalized=False):
        if 'Expert' in self.config.system:
            raise RuntimeError('Cannot call self.view() with Expert system')
        # views = self.viewmaker(self.normalize(imgs))
        # unnormalized = self.unnormalize(views)
        unnormalized = self.viewmaker(imgs)
        views = self.normalize(unnormalized)
        if with_unnormalized:
            return views, unnormalized
        return views

    def normalize(self, imgs):
        # These numbers were computed using compute_image_dset_stats.py
        if hasattr(self.train_dataset, "normalize"):
            return self.train_dataset.normalize(imgs)
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