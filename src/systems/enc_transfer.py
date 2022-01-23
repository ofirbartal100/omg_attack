import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import wandb

from dabs.src.systems.base_system import BaseSystem


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


class TransferSystem(BaseSystem):

    def __init__(self, config):
        super().__init__(config)

        # Restore checkpoint if provided.
        if config.ckpt is not None:
            model_weight_dict = self.get_model_weights(config.ckpt)
            self.model.load_state_dict(model_weight_dict)
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

    def get_model_weights(self, ckpt):
        system_weights = torch.load(ckpt, map=self.device)['state_dict']
        model_weights = {}
        for name, weight in system_weights.items():
            if "model" in name:
                model_weight_name = name.replace("model.", "")
                model_weights[model_weight_name] = weight
        return model_weights

    def objective(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)

    def forward(self, batch):
        batch[0] = self.normalize(batch[0])
        embs = self.model.forward(batch, prehead=True)
        preds = self.linear(embs)
        return preds

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

    def training_step(self, batch, batch_idx):
        batch, labels = batch[1:-1], batch[-1]
        preds = self.forward(batch)
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
        return loss

    def on_train_epoch_end(self):
        '''Log auroc at end of epoch here to guarantee presence of every class.'''
        if self.is_auroc:
            metric = self.metric_fn.compute()
            self.log(
                'transfer/train_metric',
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            wandb.log({'transfer/train_metric': metric})
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
        wandb.log({'transfer/val_metric': metric, 'transfer/val_loss': loss})


    def on_validation_epoch_end(self):
        '''Log auroc at end of epoch here to guarantee presence of every class.'''
        if self.is_auroc:
            try:
                metric = self.metric_fn.compute()
                self.log(
                    'transfer/val_metric',
                    metric,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    prog_bar=True,
                )
                wandb.log({'transfer/train_metric': metric})
            except ValueError as error:
                self.log('transfer/val_metric', 0.0, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                print(f'Logging `0.0` due to {error}. Is this from sanity check?')
            self.metric_fn.reset()
