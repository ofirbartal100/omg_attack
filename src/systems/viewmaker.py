import cv2
import numpy as np
import torch
import torchmetrics
from collections import OrderedDict

from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid
import wandb

from dabs.src.systems.base_system import BaseSystem
from viewmaker.src.gans.tiny_pix2pix import TinyP2PDiscriminator

from viewmaker.src.models.viewmaker import Viewmaker
from viewmaker.src.objectives.memory_bank import MemoryBank
from viewmaker.src.systems.image_systems.utils import heatmap_of_view_effect
from viewmaker.src.objectives.adversarial import AdversarialSimCLRLoss
from viewmaker.src.utils import utils

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


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
        #### delete ####
        self.setup2()
        ################

    def forward(self, x, prehead=False):
        x[0] = self.normalize(x[0])
        return self.model.forward(x, prehead=prehead)

    def make_views(self, batch):
        indices, img, _ = batch
        views1, unnormalized_view1 = self.view(img, True)
        views2, unnormalized_view2 = self.view(img, True)
        emb_dict = {
            'indices': indices,
            'originals': img,
            'views1': views1,
            'unnormalized_view1': unnormalized_view1,
            'views2': views2,
            'unnormalized_view2': unnormalized_view2
        }
        return emb_dict

    def ssl_forward(self, batch):
        emb_dict = self.make_views(batch)
        emb_dict.update({'view1_embs': self.model.forward([emb_dict["views1"]]),
                         'view2_embs': self.model.forward([emb_dict["views2"]]),
                         'orig_embs': self.model.forward([self.normalize(emb_dict["originals"])])})
        return emb_dict

    def training_step(self, batch, batch_idx, optimizer_idx):
        emb_dict = self.ssl_forward(batch)
        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = self.objective(emb_dict)
        emb_dict['optimizer_idx'] = torch.tensor(optimizer_idx, device=self.device)

        if optimizer_idx == 0:
            metrics = {'encoder_loss': encoder_loss, "train_acc": encoder_acc, "positive_sim": positive_sim,
                       "negative_sim": negative_sim}
            loss = encoder_loss
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

        ##### delete ######
        if "orig_embs" in emb_dict:
            self.add_to_memory_bank(emb_dict["indices"], emb_dict["orig_embs"])
        ###################

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
        knn_acc = self.get_nearest_neighbor_label(emb_dict["orig_embs"], labels)

        output = OrderedDict({
            'val_encoder_loss': encoder_loss,
            'val_view_maker_loss': view_maker_loss,
            'val_zero_knn_acc': torch.tensor(knn_acc, dtype=float, device=self.device),
            'val_positive_sim': positive_sim,
            'val_negative_sim': negative_sim,
        })
        return output

    def validation_step_end(self, output):
        self.log_dict(output)
        return output

    ##### delete this ######
    def setup2(self):
        self.memory_bank = MemoryBank(
            len(self.train_dataset),
            128,
        )

    def add_to_memory_bank(self, indices, img_embs):
        new_data_memory = utils.l2_normalize(img_embs, dim=1)
        self.memory_bank.update(indices, new_data_memory)

    def get_nearest_neighbor_label(self, img_embs, labels):
        '''
        Used for online kNN classifier.
        For each image in validation, find the nearest image in the
        training dataset using the memory bank. Assume its label as
        the predicted label.
        '''
        batch_size = img_embs.size(0)
        all_dps = self.memory_bank.get_all_dot_products(img_embs)
        _, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
        neighbor_idxs = neighbor_idxs.squeeze(1)
        neighbor_idxs = neighbor_idxs.cpu().numpy()

        neighbor_labels = [self.train_dataset[idx][2] for idx in neighbor_idxs]
        neighbor_labels = torch.Tensor(neighbor_labels).long()
        num_correct = torch.sum(neighbor_labels.cpu() == labels.cpu()).item()

        return num_correct / batch_size

    # ########################

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        # update viwmaker every step
        if optimizer_idx == 0:
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu,
                                   using_native_amp, using_lbfgs)
        elif optimizer_idx == 1:
            if self.view_maker_freeze():
                # freeze viewmaker after a certain number of epochs
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()
            else:
                super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu,
                                       using_native_amp, using_lbfgs)

    def view_maker_freeze(self):
        if self.config.optim_params.get("viewmaker_freeze_epoch"):
            return self.current_epoch > self.config.optim_params.viewmaker_freeze_epoch
        else:
            return False

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

        view_optim_name = self.config.optim_params.get("viewmaker_optim")
        view_parameters = self.viewmaker.parameters()
        if view_optim_name == 'adam':
            view_optim = torch.optim.Adam(view_parameters,
                                          lr=self.config.optim_params.get("viewmaker_learning_rate", 0.001))
        elif not view_optim_name or view_optim_name == 'sgd':
            view_optim = torch.optim.SGD(
                view_parameters,
                lr=self.config.optim_params.viewmaker_learning_rate or self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        else:
            raise ValueError(f'Optimizer {view_optim_name} not implemented')

        opt_list = [encoder_optim, view_optim]

        return opt_list, []

    def wandb_logging(self, emb_dict):
        logging_steps = 200
        if isinstance(self.logger, WandbLogger):
            logging_steps = 200

        # check optimizer index to log images only once
        # # Handle Tensor (dp) and int (ddp) cases
        if emb_dict['optimizer_idx'].__class__ == int or emb_dict['optimizer_idx'].dim() == 0:
            optimizer_idx = emb_dict['optimizer_idx']
        else:
            optimizer_idx = emb_dict['optimizer_idx'][0]
        if optimizer_idx > 0:
            return


        if self.global_step % logging_steps == 0:
            amount_images = 10
            img = emb_dict['originals']

            gradcam_viz = []
            for im in img[:amount_images]:
                gradcam_viz.append(self.gradcam(im))
            gradcam_viz = torch.stack(gradcam_viz)

            unnormalized_view1 = emb_dict['unnormalized_view1']
            unnormalized_view2 = emb_dict['unnormalized_view2']

            diff_heatmap = heatmap_of_view_effect(img[:amount_images], unnormalized_view1[:amount_images])
            diff_heatmap2 = heatmap_of_view_effect(img[:amount_images], unnormalized_view2[:amount_images])
            if img.size(1) > 3:
                img = img.mean(1, keepdim=True)
                unnormalized_view1 = unnormalized_view1.mean(1, keepdim=True)
                unnormalized_view2 = unnormalized_view2.mean(1, keepdim=True)
                diff_heatmap = diff_heatmap.mean(1, keepdim=True)
                diff_heatmap2 = diff_heatmap2.mean(1, keepdim=True)
            cat = torch.cat(
                [img[:amount_images],
                 gradcam_viz[:amount_images],
                 unnormalized_view1[:amount_images],
                 unnormalized_view2[:amount_images],
                 diff_heatmap,
                 diff_heatmap2,
                 (diff_heatmap - diff_heatmap2).abs()])
            grid = make_grid(cat, nrow=amount_images)
            grid = resize(torch.clamp(grid, 0, 1.0), (560, 1120), InterpolationMode.NEAREST)
            if isinstance(self.logger, WandbLogger):
                wandb.log({
                    "original_vs_views": wandb.Image(grid,
                                                     caption=f"Epoch: {self.current_epoch}, Step {self.global_step}"),
                    "mean distortion": (unnormalized_view1 - img).abs().mean(),
                })
            else:
                self.logger.experiment.add_image('original_vs_views', grid, self.global_step)

    def create_viewmaker(self):
        view_model = Viewmaker(
            num_channels=self.train_dataset.IN_CHANNELS,
            activation=self.config.model_params.get("generator_activation", 'relu'),
            clamp=self.config.model_params.get("clamp_views", True),
            frequency_domain=self.config.model_params.get("spectral", False),
            downsample_to=self.config.model_params.get("viewmaker_downsample", False),
            num_res_blocks=self.config.model_params.get("num_res_blocks", 5),
            use_budget=self.config.model_params.get("use_budget", True),
            budget_aware=self.config.model_params.get("budget_aware", False),
            image_dim=(32, 32),
            multiplicative=self.config.model_params.get("multiplicative", 0),
            multiplicative_budget=self.config.model_params.get("multiplicative_budget", 0.25),
            additive=self.config.model_params.get("additive", 1),
            additive_budget=self.config.model_params.get("additive_budget", 0.05),
            tps=self.config.model_params.get("tps", 0),
            tps_budget=self.config.model_params.get("tps_budget", 0.1),
            aug_proba=self.config.model_params.get("aug_proba", 1),
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

    def gradcam(self, img):

        class VectorSimilarity:
            def __init__(self, dim):
                self.reference = torch.ones(dim) / np.sqrt(dim)

            def __call__(self, model_output):
                self.reference = self.reference.to(model_output.device)
                sim = torch.nn.CosineSimilarity(0)(model_output, self.reference)
                return sim

        model = self.model.resnet
        target_layers = [model.layer4[-1]]
        if img.ndim == 4:
            input_tensor = img
        else:
            input_tensor = img.unsqueeze(0)
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        targets = [VectorSimilarity(self.config.model.kwargs.out_dim)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        def show_cam_on_image(img: np.ndarray,
                              mask: np.ndarray,
                              use_rgb: bool = False,
                              colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
            """ This function overlays the cam mask on the image as an heatmap.
            By default the heatmap is in BGR format.
            :param img: The base image in RGB or BGR format.
            :param mask: The cam mask.
            :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
            :param colormap: The OpenCV colormap to be used.
            :returns: The default image with the cam overlay.
            """
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
            if use_rgb:
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = np.float32(heatmap) / 255

            if np.max(img) > 1:
                raise Exception(
                    "The input image should np.float32 in the range [0, 1]")

            cam = heatmap + img
            cam = cam / np.max(cam)
            return np.uint8(255 * cam)

        res = show_cam_on_image(img.detach().cpu().permute(1,2,0).numpy(), grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET) / 255
        return torch.from_numpy(res).to(input_tensor.device).permute(2, 0, 1)


class ViewmakerSystemDisc(ViewmakerSystem):

    def setup(self, stage):
        super().setup(stage)
        self.disc = TinyP2PDiscriminator(in_channels=self.dataset.spec()[0].in_channels,
                                         wgan=self.config.disc.wgan,
                                         blocks_num=self.config.disc.conv_blocks)

    def training_step(self, batch, batch_idx, optimizer_idx):
        step_output = {'optimizer_idx': torch.tensor(optimizer_idx, device=self.device)}

        if optimizer_idx in [0, 1]:
            step_output.update(self.ssl_forward(batch))
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
        if "views1" not in step_output:
            views1, unnormalized_view1 = self.view(img, True)
            views2, unnormalized_view2 = self.view(img, True)
            step_output.update({'indices': indices,
                                'originals': img,
                                'views1': views1,
                                'unnormalized_view1': unnormalized_view1,
                                'views2': views2,
                                'unnormalized_view2': unnormalized_view2})
        views1, views2 = step_output["views1"], step_output["views2"]

        img.requires_grad = True
        # self.disc = self.disc.to(self.device)
        step_output['disc_r1_penalty'] = 0.0
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
        knn_acc = self.get_nearest_neighbor_label(step_output["orig_embs"], batch[-1])

        output = OrderedDict({
            'val_encoder_loss': encoder_loss,
            'val_view_maker_loss': view_maker_loss,
            'val_zero_knn_acc': torch.tensor(knn_acc, dtype=float, device=self.device),
            'val_positive_sim': positive_sim,
            'val_negative_sim': negative_sim,
            'val_generator_loss': gen_loss,
            'val_disc_loss': disc_loss,
            'val_disc_acc': disc_acc,
            'val_encoder_acc': encoder_acc
        })
        return output

    def on_validation_model_eval(self) -> None:
        """Sets the model to eval during the val loop."""
        self.trainer.model.eval()
        self.trainer.model.module.module.disc.train()
        self.trainer.model.module.module.viewmaker.train()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False,
                               using_native_amp=False, using_lbfgs=False)
        if optimizer_idx == 2:
            if self.view_maker_freeze():
                # freeze viewmaker after a certain number of epochs
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()
            elif batch_idx % (self.config.disc.dis_skip_steps + 1) == 0:
                super(ViewmakerSystem, self).optimizer_step(epoch, batch_idx, optimizer, optimizer_idx,
                                                            optimizer_closure, on_tpu,
                                                            using_native_amp, using_lbfgs)
            else:
                optimizer_closure()

    def configure_optimizers(self):
        opt_list, [] = super().configure_optimizers()
        disc_optim = torch.optim.Adam(self.disc.parameters(),
                                      lr=self.config.disc.lr)
        opt_list.append(disc_optim)
        return opt_list, []


class DoubleViewmakerSystem(ViewmakerSystemDisc):

    def setup(self, stage):
        super().setup(self)
        self.viewmaker2 = self.create_viewmaker()

    def configure_optimizers(self):
        opt_list, [] = super().configure_optimizers()
        opt_list[1].add_param_group({'params': self.viewmaker2.parameters()})
        return opt_list, []

    def view(self, imgs, with_unnormalized=False):
        if 'Expert' in self.config.system:
            raise RuntimeError('Cannot call self.view() with Expert system')
        unnormalized_disc = self.viewmaker(imgs)
        unnormalized = self.viewmaker2(imgs, return_view_func=True)(unnormalized_disc)
        # normalize
        views_disc = self.normalize(unnormalized_disc)
        views = self.normalize(unnormalized)
        if with_unnormalized:
            return views, unnormalized, views_disc, unnormalized_disc
        return views, views_disc

    def make_views(self, batch):
        indices, img, _ = batch
        views1, unnormalized_view1, views1_disc, unnormalized_disc_view1 = self.view(img, True)
        views2, unnormalized_view2, views2_disc, unnormalized_disc_view2 = self.view(img, True)
        emb_dict = {
            'indices': indices,
            'originals': img,
            'views1': views1,
            'unnormalized_view1': unnormalized_view1,
            'views1_disc': views1_disc,
            'unnormalized_disc_view1': unnormalized_disc_view1,
            'views2': views2,
            'unnormalized_view2': unnormalized_view2,
            'views2_disc': views2_disc,
            'unnormalizeddisc__view2': unnormalized_disc_view2,
        }
        return emb_dict

    def ssl_forward(self, batch):
        emb_dict = self.make_views(batch)
        emb_dict.update({'view1_embs': self.model.forward([emb_dict["views1"]]),
                         'view2_embs': self.model.forward([emb_dict["views2"]]),
                         'view1_disc_embs': self.model.forward([emb_dict["views1_disc"]]),
                         'view2_disc_embs': self.model.forward([emb_dict["views2_disc"]]),
                         'orig_embs': self.model.forward([self.normalize(emb_dict["originals"])])})
        return emb_dict

    # def objective(self, emb_dict):
    #     view_maker_loss_weight = self.config.loss_params.view_maker_loss_weight
    #     loss_function_disc = AdversarialSimCLRLoss(
    #         embs1=emb_dict['view1_disc_embs'],
    #         embs2=emb_dict['view2_disc_embs'],
    #         t=self.config.loss_params.t,
    #         view_maker_loss_weight=view_maker_loss_weight
    #     )
    #     loss_function_double = AdversarialSimCLRLoss(
    #         embs1=emb_dict['view1_embs'],
    #         embs2=emb_dict['view2_embs'],
    #         t=self.config.loss_params.t,
    #         view_maker_loss_weight=view_maker_loss_weight
    #     )
    #     encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = loss_function_disc.get_loss()
    #     encoder_loss2, encoder_acc2, view_maker_loss2, positive_sim2, negative_sim2 = loss_function_double.get_loss()
    #     encoder_loss = (encoder_loss + encoder_loss2)/2
    #     encoder_acc = (encoder_acc + encoder_acc2)/2
    #     view_maker_loss = (view_maker_loss + view_maker_loss2)/2
    #     positive_sim = (positive_sim + positive_sim2)/2
    #     negative_sim = (negative_sim + negative_sim2)/2
    #     return encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim

    def gan_forward(self, batch, step_output, optimizer_idx=2):
        indices, img, _ = batch
        if "views1" not in step_output:
            step_output.update(self.make_views(batch))
        views1, views2 = step_output["views1_disc"], step_output["views2_disc"]

        img.requires_grad = True
        # self.disc = self.disc.to(self.device)
        step_output['disc_r1_penalty'] = 0.0
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


class ViewmakerCoopSystem(ViewmakerSystem):

    def training_step(self, batch, batch_idx, optimizer_idx):
        emb_dict = self.ssl_forward(batch)
        encoder_loss, encoder_acc, view_maker_loss, positive_sim, negative_sim = self.objective(emb_dict)
        emb_dict['optimizer_idx'] = torch.tensor(optimizer_idx, device=self.device)

        if optimizer_idx == 0:
            metrics = {'encoder_loss': encoder_loss, "train_acc": encoder_acc, "positive_sim": positive_sim,
                       "negative_sim": negative_sim}
            loss = encoder_loss
        elif optimizer_idx == 1:
            diff_loss = 1 / (((emb_dict['views1'] - emb_dict['views2']) ** 2).mean() + 1e-4)
            metrics = {'view_maker_loss': encoder_loss + diff_loss,
                       'diff_loss': diff_loss}
            loss = encoder_loss + diff_loss
            if positive_sim * (1 - negative_sim) > 0.95:
                self.viewmaker.additive_budget += 0.01
        else:
            loss = None
            metrics = None

        return [loss, emb_dict, metrics]
