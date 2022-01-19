from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from dotmap import DotMap
from collections import OrderedDict
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid
import wandb
import json
from dabs.src.systems.base_system import BaseSystem
from viewmaker.src.models.viewmaker import Viewmaker
from viewmaker.src.systems.image_systems.utils import heatmap_of_view_effect
from viewmaker.src.objectives.adversarial import AdversarialSimCLRLoss


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    '''Wrapper around BCEWithLogits to cast labels to float before computing loss'''

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target.float())


def get_loss_and_metric_fns(loss, metric, num_classes):
    # Get loss function.
    if loss == 'cross_entropy':
        loss_fn = F.cross_entropy
    elif loss == 'binary_cross_entropy':
        loss_fn = BCEWithLogitsLoss()  # use wrapper module instead of wrapper function so torch can pickle later
    elif loss == 'mse':
        loss_fn = F.mse_loss
    else:
        raise ValueError(f'Loss name {loss} unrecognized.')

    # Get metric function.
    if metric == 'accuracy':
        metric_fn = torchmetrics.functional.accuracy
    elif metric == 'auroc':
        metric_fn = torchmetrics.AUROC(num_classes=num_classes)
    elif metric == 'pearson':
        metric_fn = torchmetrics.functional.pearson_corrcoef
    elif metric == 'spearman':
        metric_fn = torchmetrics.functional.spearman_corrcoef
    else:
        raise ValueError(f'Metric name {metric} unrecognized.')

    # Get post-processing function.
    post_fn = nn.Identity()
    if loss == 'cross_entropy' and metric == 'accuracy':
        post_fn = nn.Softmax(dim=1)
    elif (loss == 'binary_cross_entropy' and metric == 'accuracy') or metric == 'auroc':
        post_fn = torch.sigmoid

    return loss_fn, metric_fn, post_fn


class ViewmakerTransferSystem(BaseSystem):

    def __init__(self, config):
        super().__init__(config)

        # Restore checkpoint if provided.
        if config.ckpt is not None:
            self.load_state_dict(torch.load(config.ckpt)['state_dict'], strict=False)
            for param in self.model.parameters():
                param.requires_grad = False

        # Prepare and initialize linear classifier.
        num_classes = self.dataset.num_classes()
        if num_classes is None:
            num_classes = 1  # maps to 1 output channel for regression
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(self.model.emb_dim, num_classes)

        # Initialize loss and metric functions.
        self.loss_fn, self.metric_fn, self.post_fn = get_loss_and_metric_fns(
            config.dataset.loss,
            config.dataset.metric,
            self.dataset.num_classes(),
        )
        self.is_auroc = (config.dataset.metric == 'auroc')  # this metric should only be computed per epoch

        self.viewmaker = self.load_viewmaker_from_checkpoint(config.vm_ckpt,config.viewmaker.config_path)

    def load_viewmaker_from_checkpoint(self,  system_ckpt, config_path, eval=True):
        config_path =config_path
        with open(config_path, 'r') as fp:
            vm_config = OmegaConf.load(fp.name)

        sd = torch.load(system_ckpt)['state_dict']
        vm_sd = OrderedDict([(i.replace('viewmaker.',''),sd[i]) for i in sd if 'viewmaker' in i])

        viewmaker = Viewmaker(
            num_channels=vm_config.train_dataset.IN_CHANNELS,
            activation=vm_config.model_params.generator_activation or 'relu',
            clamp=vm_config.model_params.clamp_views or False,
            frequency_domain=vm_config.model_params.spectral or False,
            downsample_to=vm_config.model_params.viewmaker_downsample or False,
            num_res_blocks=vm_config.model_params.num_res_blocks or 5,
            use_budget=vm_config.model_params.use_budget or True,
            budget_aware=vm_config.model_params.budget_aware or False,
            image_dim=(32, 32),
            multiplicative=vm_config.model_params.multiplicative or 0,
            multiplicative_budget=vm_config.model_params.multiplicative_budget or 0.25,
            additive=vm_config.model_params.additive or 1,
            additive_budget=vm_config.model_params.additive_budget or 0.05,
            tps=vm_config.model_params.tps or 0,
            tps_budget=vm_config.model_params.tps_budget or 0.1,
            aug_proba=vm_config.model_params.aug_proba or 1,
        )

        viewmaker.load_state_dict(vm_sd,strict=False)
        viewmaker.eval()
        for param in viewmaker.parameters():
            param.requires_grad = False

        return viewmaker

    def objective(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)

    def forward(self, batch):
        embs = self.model.forward(batch, prehead=True)
        preds = self.linear(embs)
        return preds

    def training_step(self, batch, batch_idx):
        batch, labels = batch[1:-1], batch[-1]
        vb, uvb = self.view(batch[0],True)
        preds = self.forward([vb])
        if self.num_classes == 1:
            preds = preds.squeeze(1)

        loss = self.objective(preds, labels)
        self.log('transfer/train_loss', loss.item(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        with torch.no_grad():
            if self.is_auroc:
                self.metric_fn.update(self.post_fn(preds.float()), labels)
            else:
                metric = self.metric_fn(self.post_fn(preds.float()), labels)
                self.log('transfer/train_metric', metric, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        self.wandb_logging(batch[0],uvb)
        return loss

    def on_train_epoch_end(self):
        '''Log auroc at end of epoch here to guarantee presence of every class.'''
        if self.is_auroc:
            self.log(
                'transfer/train_metric',
                self.metric_fn.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.metric_fn.reset()

    def validation_step(self, batch, batch_idx):
        batch, labels = batch[1:-1], batch[-1]
        preds = self.forward(batch)
        if self.num_classes == 1:
            preds = preds.squeeze(1)

        loss = self.objective(preds, labels)
        self.log('transfer/val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if self.is_auroc:
            self.metric_fn.update(self.post_fn(preds.float()), labels)
        else:
            metric = self.metric_fn(self.post_fn(preds.float()), labels)
            self.log('transfer/val_metric', metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        '''Log auroc at end of epoch here to guarantee presence of every class.'''
        if self.is_auroc:
            try:
                self.log(
                    'transfer/val_metric',
                    self.metric_fn.compute(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    prog_bar=True,
                )
            except ValueError as error:
                self.log('transfer/val_metric', 0.0, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                print(f'Logging `0.0` due to {error}. Is this from sanity check?')
            self.metric_fn.reset()

    def view(self, imgs, with_unnormalized=False):
        if 'Expert' in self.config.system:
            raise RuntimeError('Cannot call self.view() with Expert system')
        views = self.viewmaker(self.normalize(imgs))
        unnormalized = self.unnormalize(views)
        # views = self.normalize(unnormalized)
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

    def wandb_logging(self, img,view1):
        logging_steps = 20
        if isinstance(self.logger, WandbLogger):
            logging_steps = 200

        if self.global_step % logging_steps == 0:
            amount_images = 10

            diff_heatmap = heatmap_of_view_effect(img[:amount_images], view1[:amount_images])
            if img.size(1) >3:
                img = img.mean(1, keepdim=True)
                unnormalized_view1 = view1.mean(1, keepdim=True)
                diff_heatmap = diff_heatmap.mean(1, keepdim=True)
            cat = torch.cat([img[:amount_images], view1[:amount_images],diff_heatmap])
            grid = make_grid(cat, nrow=amount_images)
            grid = torch.clamp(grid, 0, 1.0)
            if isinstance(self.logger, WandbLogger):
                wandb.log({
                    "original_vs_views": wandb.Image(grid, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}"),
                    "mean distortion": (view1 - img).abs().mean(),
                })
            else:
                self.logger.experiment.add_image('original_vs_views', grid, self.global_step)

    