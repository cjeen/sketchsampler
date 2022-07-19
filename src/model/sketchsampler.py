from typing import Any, List, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.metric import Metric
from common.networks import define_G, get_norm_layer, init_net, UnitBlock, ResMLP
from common.schedulers import get_decay_scheduler
from common.utils import get_classdict
from omegaconf import DictConfig
from torch.optim import Optimizer


class SketchTranslator(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = define_G(input_nc=1, output_nc=64 + 1, ngf=64, netG='resnet_9blocks', norm='instance',
                                  use_dropout=False, init_type='normal', init_gain=0.02)

    def forward(self, sketch):
        feature1 = self.parameter.model[:22](sketch)
        feature2 = self.parameter.model[22:25](feature1)
        feature3 = self.parameter.model[25:28](feature2)
        feature4 = self.parameter.model[28:](feature3)
        return [feature1, feature2, feature3, feature4]


class DensityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = UnitBlock(512 + 256 + 128 + 64, 256)
        self.conv = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.block2 = UnitBlock(64, 1)

    def forward(self, features):
        _, _, h_out, w_out = features[-1].shape
        for i in range(len(features) - 1):
            features[i] = F.interpolate(features[i], (h_out, w_out))
        input_feature = torch.cat(features, dim=1)
        m = self.block1(input_feature)
        m = self.relu(self.conv(m))
        m = self.block2(m)
        m = m / torch.sum(m, dim=(2, 3), keepdim=True)
        return m


class MapSampler(nn.Module):
    def __init__(self, mode='stable'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['stable', 'normal']

    def forward(self, density_map, pt_count):
        if self.mode == 'stable':
            return self.sample_stable(density_map, pt_count)
        elif self.mode == 'normal':
            return self.sample_normal(density_map, pt_count)
        else:
            raise NotImplementedError

    def sample_stable(self, density_map, pt_count):
        density_map = density_map * pt_count
        H, W = density_map.shape[-2:]
        density_map = torch.round(density_map).long()
        points = [torch.where(img[0] >= 1 - 1e-3) for img in density_map]
        density = [b_map[0][b_point] for b_map, b_point in zip(density_map, points)]
        aux_norm = torch.tensor([(W - 1), (H - 1)], device=density_map.device).view(1, 2)
        pointclouds_normed = [torch.stack(point, dim=-1).flip(-1).float() * 2 / aux_norm - 1
                              for point in points]
        xys = [torch.repeat_interleave(pointcloud_normed, b_density, dim=0) for pointcloud_normed, b_density in
               zip(pointclouds_normed, density)]
        random_prior = [torch.rand(size=(xy.shape[0], 1), device=density_map.device) for xy
                        in
                        xys]
        return xys, random_prior

    def sample_normal(self, density_map, pt_count):
        H, W = density_map.shape[-2:]
        indices = torch.multinomial(density_map[:, 0, :, :].flatten(1), pt_count, replacement=True)
        xs = indices % W
        ys = indices // W
        xys = torch.stack((xs, ys), -1)
        aux_norm = torch.tensor([(W - 1), (H - 1)], device=density_map.device).view(1, 2)
        xys = xys.float() * 2 / aux_norm - 1
        random_prior = torch.rand(size=(xys.shape[0], xys.shape[1], 1), device=density_map.device)
        return xys, random_prior


class DepthHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_encode1 = ResMLP(1, 16, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.z_encode2 = ResMLP(16, 32, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.z_encode3 = ResMLP(32, 64, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())

        self.z_decode1 = ResMLP(512 + 256 + 128 + 64 + 64, 64, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.z_decode2 = ResMLP(64, 32, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.z_decode3 = ResMLP(32, 16, norm_layer=get_norm_layer('instance1D'),
                                activation=nn.ReLU())
        self.gen_z = nn.Linear(16, 1)

    def unproj(self, WH, Z):
        w, h = WH.T
        z = Z.squeeze(1)
        Y = -h * 0.75
        X = w * 0.75
        res = torch.stack([X, Y, z], dim=1)
        return res

    def forward(self, features, WH, random_prior):
        batchsize = features[0].shape[0]
        pointclouds = []
        for i in range(batchsize):
            feats_i = [feat[i].unsqueeze(0) for feat in features]
            wh_i = WH[i].unsqueeze(0)
            random_prior_i = random_prior[i].unsqueeze(0)
            z_i = self.z_encode1(random_prior_i)
            z_i = self.z_encode2(z_i)
            z_i = self.z_encode3(z_i)

            feats_sampled_i = [F.grid_sample(
                feat_i, wh_i.unsqueeze(1),
                mode='bilinear', align_corners=False)[:, :, 0, :] \
                                   .permute(0, 2, 1) for feat_i in feats_i]
            feats_i = torch.cat(feats_sampled_i, dim=-1)
            z_i = self.z_decode1(torch.cat([feats_i, z_i], dim=-1))
            z_i = self.z_decode2(z_i)
            z_i = self.z_decode3(z_i)
            z_i = self.gen_z(z_i)
            z_i -= z_i.mean(dim=1, keepdim=True)
            pointclouds.append(self.unproj(wh_i.squeeze(0), z_i.squeeze(0)))
        return pointclouds


class SketchSampler(pl.LightningModule):
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.save_hyperparameters()

        self.depth_head = init_net(DepthHead())
        self.density_head = init_net(DensityHead())
        self.sketch_translator = SketchTranslator()
        self.map_sampler = MapSampler()

        self.n_points = self.cfg.train.n_points
        self.lambda1 = self.cfg.train.lambda1
        self.lambda2 = self.cfg.train.lambda2

        self.class_dict = get_classdict()
        self.train_metric = Metric(self.class_dict, 'train')
        self.val_metric = Metric(self.class_dict, 'val')
        self.test_metric = Metric(self.class_dict, 'test')

    def forward(self, sketch, density_map, use_predicted_map):
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        """
        features = self.sketch_translator(sketch)
        predicted_map = self.density_head(features)

        if use_predicted_map or density_map is None:
            WH, depth_prior = self.map_sampler(predicted_map, self.n_points)
        else:
            WH, depth_prior = self.map_sampler(density_map, self.n_points)
        predicted_points = self.depth_head(features, WH, depth_prior)
        return predicted_map, predicted_points

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        sketch, pointclouds, density_map, metadata = batch
        predicted_map, predicted_points = self(sketch, density_map, use_predicted_map=False)
        points_loss = self.train_metric.loss_points(predicted_points, pointclouds)
        density_loss = self.train_metric.loss_density(predicted_map, density_map)
        train_loss = self.lambda1 * points_loss + self.lambda2 * density_loss
        self.log_dict(
            {"train_loss": train_loss,
             "density_loss": density_loss,
             "points_loss": points_loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        return train_loss

    def validation_step(self, batch: Any, batch_idx: int):
        sketch, pointclouds, density_map, metadata = batch
        predicted_map, predicted_points = self(sketch, density_map, use_predicted_map=True)
        self.val_metric.evaluate_chamfer_and_f1(predicted_points, pointclouds, metadata)
        return

    def test_step(self, batch: Any, batch_idx: int):
        sketch, pointclouds, density_map, metadata = batch
        predicted_map, predicted_points = self(sketch, density_map, use_predicted_map=True)
        self.test_metric.evaluate_chamfer_and_f1(predicted_points, pointclouds, metadata)
        return

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self.val_metric.get_dict())
        self.val_metric.reset_state()

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self.test_metric.get_dict())
        self.test_metric.reset_state()

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(
            self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.cfg.optim.optimizer,
            params=self.parameters()
        )

        if self.cfg.optim.use_lr_scheduler:
            scheduler = [{
                'scheduler': get_decay_scheduler(opt, self.num_training_steps(), 0.9),
                'interval': 'step',
            }]
            return [opt], scheduler

        return opt
