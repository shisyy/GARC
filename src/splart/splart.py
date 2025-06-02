from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import torch
from gsplat.strategy.ops import duplicate
from kornia.filters import sobel
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.trainer import Trainer
from nerfstudio.model_components.lib_bilagrid import color_correct, total_variation_loss
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat
from nerfstudio.utils import colormaps
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_apply, quaternion_raw_multiply
from torch.nn import Parameter
from torch.nn.functional import l1_loss, normalize, relu
from torch.optim import Adam
from tqdm import tqdm
from vis_utils.utils import (
    angle,
    ch_cam_pose_spec,
    compute_trans_diff,
    homogenize_transforms,
    homogenize_vecs,
    put_texts,
    to_np_color,
    to_pt_color,
)

from splart.articulation_params import ArticulationParams, ArticulationType
from splart.rasterization import rasterization
from splart.splart_strategy import SplartStrategy


@dataclass
class SplartModelConfig(SplatfactoModelConfig):
    """SplArt Model Config"""

    _target: Type = field(default_factory=lambda: SplartModel)
    strategy: Literal["default"] = "default"
    """only default strategy is supported"""
    articulation_type: ArticulationType = ArticulationType.UNDEFINED
    """articulation type of the object"""
    lambda_opacity_reg: float = 1e-5
    """weight of the opacity regularization"""
    enable_depth_consistency: bool = True
    """whether to enable depth consistency loss"""
    lambda_depth_consistency_loss: float = 10
    """weight of the depth consistency loss"""
    depth_grad_gamma: float = 30
    """gamma of the depth gradient (used in computing the depth consistency loss)"""
    depth_tolerance: float = 0.05
    """tolerance of the depth error (used in computing the depth consistency loss)"""
    lambda_mobility_reg_photo: float = 5e-3
    """weight of the mobility regularization for the cross-static rendering loss"""
    lambda_mobility_reg_geom: float = 1e-4
    """weight of the mobility regularization for the cross-static geometric consistency loss"""
    optim_seed: int = 0
    """seed for the randomization of initial articulation params"""
    n_optim_tries: int = 3
    """number of optimization tries for estimating articulation params; for both mobile (m) and cross-mobile (cm) geometric consistency formualtions"""
    n_max_optim_iters: int = 10000
    """maximum number of iterations for the optimization of articulation params"""
    cs_after: int = 10000
    """step after which the cross-static formulation takes effect (for mobilities optimization)"""
    cm_after: int = 15000
    """step after which the cross-mobile formulation takes effect (for joint articulation and mobilities optimization)"""
    stop_split_at: int = 20000
    """stop splitting at this step"""


class SplartModel(SplatfactoModel):
    """SplArt Model

    Args:
        config: SplArt configuration to instantiate model
    """

    config: SplartModelConfig

    def populate_modules(self):
        super().populate_modules()
        self.gauss_params.update(
            {
                "states": Parameter(torch.randint(2, (self.num_points, 1)), requires_grad=False),
                "mobilities": Parameter(torch.full((self.num_points, 1), torch.nan)),
            }
        )
        self.gt_articulation_params = ArticulationParams.get_empty()
        self.articulation_params = ArticulationParams({"type": self.config.articulation_type})
        self.strategy.__class__ = SplartStrategy
        self.last_step = None
        # for visualizing part segmentation
        self.static_color = torch.tensor((0, 0, 1))
        self.mobile_color = torch.tensor((1, 0, 0))

    @property
    def states(self):
        return self.gauss_params["states"]

    @property
    def mobilities(self):
        return self.gauss_params["mobilities"]

    def load_state_dict(self, dict, **kwargs):
        for name, param in self.gauss_params.items():
            param.data = torch.empty_like(dict[f"gauss_params.{name}"], device=self.device)
        super(SplatfactoModel, self).load_state_dict(dict, **kwargs)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = super().get_training_callbacks(training_callback_attributes)
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb_extra,
                args=[training_callback_attributes.trainer],
            )
        )
        return cbs

    def step_cb_extra(self, trainer: Trainer, step):
        if self.last_step is None:
            self.last_step = step + trainer.config.max_num_iterations - 1
        if step == self.config.cs_after:
            n = self.num_points
            duplicate(
                self.gauss_params, self.optimizers, self.strategy_state, mask=torch.full((n,), True, device=self.device)
            )
            self.mobilities.data[-n:] = 0
            self.optimize_mobilities(mode="cs")
        if step == self.config.cm_after:
            self.optimize_mobilities_and_articulation(seed=self.config.optim_seed)
            for param_name in ("opacities", "mobilities"):
                del self.optimizers[param_name].state[self.gauss_params[param_name]]
            self.optimizers["mobilities"].param_groups[0].update({"lr": 3e-3})

    def step_post_backward(self, step):
        super().step_post_backward(step)
        if step == self.last_step and step >= self.config.cm_after:
            self.optimize_mobilities()

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {**super().get_gaussian_param_groups(), "mobilities": [self.gauss_params["mobilities"]]}

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["axis"] = [self.articulation_params.axis]
        if self.articulation_params.articulation_type != ArticulationType.REVOLUTE:
            param_groups["dist"] = [self.articulation_params.dist]
        if self.articulation_params.articulation_type != ArticulationType.PRISMATIC:
            param_groups["pivot"] = [self.articulation_params.pivot]
            param_groups["angle"] = [self.articulation_params.angle]
        return param_groups

    def get_outputs(
        self, camera: Cameras, mobility=None, mode: Literal["train", "val", "test"] = "test", gen_part_seg=False
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        assert camera.shape[0] == 1, "Only one camera at a time"
        if self.training:
            mode = "train"
        has_articulation_state = camera.metadata is not None and "articulation_state" in camera.metadata
        if mode == "train":
            c2w = self.camera_optimizer.apply_to_camera(camera)
            self.info = []
        else:
            c2w = camera.camera_to_worlds
        if mode != "test":
            assert (
                mobility is None and has_articulation_state and camera.metadata["articulation_state"].item() in {0, 1}
            )

        states = self.states.squeeze(-1)
        opacities = self.opacities.sigmoid().squeeze(-1)
        means = self.means
        scales = self.scales.exp()
        quats = normalize(self.quats)
        colors = torch.cat((self.features_dc[:, None], self.features_rest), dim=1)
        mobilities = self.mobilities.sigmoid().squeeze(-1)
        if self.crop_box is None or mode == "train":
            mask_crop = torch.full((self.num_points,), True, device=self.device)
        else:
            mask_crop = self.crop_box.within(means)  # Note: articulation is not considered
        mask_ref = mobilities.isnan()
        mask_tgt = ~mask_ref
        mask_0 = mask_crop & states == 0
        mask_1 = mask_crop & states == 1
        mask_ref_0 = mask_ref & mask_0
        mask_ref_1 = mask_ref & mask_1
        mask_tgt_0 = mask_tgt & mask_0
        mask_tgt_1 = mask_tgt & mask_1
        if gen_part_seg:
            static_color = self.static_color.to(colors)
            mobile_color = self.mobile_color.to(colors)
        outputs = {}

        if mode != "test" or self.step < self.config.cs_after:
            assert has_articulation_state
            state = camera.metadata["articulation_state"].item()
            assert state in {0, 1}
            mask_ref_cur = locals()[f"mask_ref_{state}"]
            if mobility is not None or gen_part_seg and self.step < self.config.cs_after:
                print("Mobilities are not estimated yet, mobility and gen_part_seg have no effect!")
            if mode == "train":
                self.info.append({"use_ids": torch.where(mask_ref_cur)[0]})
            self.update_outputs(
                outputs,
                camera,
                c2w,
                opacities[mask_ref_cur],
                means[mask_ref_cur],
                scales[mask_ref_cur],
                quats[mask_ref_cur],
                colors[mask_ref_cur],
            )
        if self.config.cs_after <= self.step < self.config.cm_after:
            assert has_articulation_state
            state = camera.metadata["articulation_state"].item()
            assert state in {0, 1}
            mask_tgt_cur = locals()[f"mask_tgt_{state}"]
            mask_tgt_other = locals()[f"mask_tgt_{1 - state}"]
            opacities_static = torch.cat(
                (
                    opacities[mask_tgt_0] * (1 - mobilities[mask_tgt_0]),
                    opacities[mask_tgt_1] * (1 - mobilities[mask_tgt_1]),
                )
            )
            opacities_mobile = opacities[mask_tgt_cur] * mobilities[mask_tgt_cur]
            means_static = torch.cat((means[mask_tgt_0], means[mask_tgt_1]))
            means_mobile = means[mask_tgt_cur]
            scales_static = torch.cat((scales[mask_tgt_0], scales[mask_tgt_1]))
            scales_mobile = scales[mask_tgt_cur]
            quats_static = torch.cat((quats[mask_tgt_0], quats[mask_tgt_1]))
            quats_mobile = quats[mask_tgt_cur]
            colors_static = torch.cat((colors[mask_tgt_0], colors[mask_tgt_1]))
            colors_mobile = colors[mask_tgt_cur]
            if gen_part_seg:
                colors_part_seg_static = static_color.repeat(len(colors_static), 1)
                colors_part_seg_mobile = mobile_color.repeat(len(colors_mobile), 1)
            if mobility is None:
                if mode == "test":
                    self.update_outputs(
                        outputs,
                        camera,
                        c2w,
                        torch.cat((opacities_static, opacities_mobile)),
                        torch.cat((means_static, means_mobile)),
                        torch.cat((scales_static, scales_mobile)),
                        torch.cat((quats_static, quats_mobile)),
                        torch.cat((colors_static, colors_mobile)),
                        colors_part_seg=(
                            torch.cat((colors_part_seg_static, colors_part_seg_mobile)) if gen_part_seg else None
                        ),
                    )
                else:
                    opacities_cur = opacities[mask_tgt_cur]
                    opacities_other = opacities[mask_tgt_other] * (1 - mobilities[mask_tgt_other])
                    means_cur = means[mask_tgt_cur]
                    means_other = means[mask_tgt_other]
                    scales_cur = scales[mask_tgt_cur]
                    scales_other = scales[mask_tgt_other]
                    quats_cur = quats[mask_tgt_cur]
                    quats_other = quats[mask_tgt_other]
                    colors_cur = colors[mask_tgt_cur]
                    colors_other = colors[mask_tgt_other]
                    if mode == "train":
                        self.info.append(
                            {"use_ids": torch.cat((torch.where(mask_tgt_cur)[0], torch.where(mask_tgt_other)[0]))}
                        )
                        outputs["mobilities_other"] = mobilities[mask_tgt_other]
                    self.update_outputs(
                        outputs,
                        camera,
                        c2w,
                        torch.cat((opacities_cur, opacities_other)),
                        torch.cat((means_cur, means_other)),
                        torch.cat((scales_cur, scales_other)),
                        torch.cat((quats_cur, quats_other)),
                        torch.cat((colors_cur, colors_other)),
                        suffix="_csr",
                    )
            if mobility == 0 or mode == "val":
                self.update_outputs(
                    outputs,
                    camera,
                    c2w,
                    opacities_static,
                    means_static,
                    scales_static,
                    quats_static,
                    colors_static,
                    colors_part_seg=colors_part_seg_static if gen_part_seg else None,
                    suffix="" if mode == "test" else "_s",
                )
            if mobility == 1 or mode == "val":
                self.update_outputs(
                    outputs,
                    camera,
                    c2w,
                    opacities_mobile,
                    means_mobile,
                    scales_mobile,
                    quats_mobile,
                    colors_mobile,
                    colors_part_seg=colors_part_seg_mobile if gen_part_seg else None,
                    suffix="" if mode == "test" else "_m",
                )
        if self.step >= self.config.cm_after:
            t = camera.metadata["articulation_state"].item() if has_articulation_state else 0.5
            means_mobile_0t = means[mask_tgt_0]
            means_mobile_1t = means[mask_tgt_1]
            quats_mobile_0t = quats[mask_tgt_0]
            quats_mobile_1t = quats[mask_tgt_1]
            axis = normalize(self.articulation_params.axis, dim=0)
            if self.articulation_params.articulation_type.item() in {
                ArticulationType.REVOLUTE,
                ArticulationType.CYLINDRICAL,
            }:
                pivot = self.articulation_params.pivot
                angle_0t = self.articulation_params.angle * t
                angle_1t = self.articulation_params.angle * (t - 1)
                rot_0t = axis_angle_to_quaternion(axis * angle_0t)
                rot_1t = axis_angle_to_quaternion(axis * angle_1t)
                means_mobile_0t = quaternion_apply(rot_0t, means_mobile_0t - pivot) + pivot
                means_mobile_1t = quaternion_apply(rot_1t, means_mobile_1t - pivot) + pivot
                quats_mobile_0t = quaternion_raw_multiply(rot_0t, quats_mobile_0t)
                quats_mobile_1t = quaternion_raw_multiply(rot_1t, quats_mobile_1t)
                motion_quats = torch.stack((rot_0t, rot_1t))
            else:
                motion_quats = None
            if self.articulation_params.articulation_type.item() in {
                ArticulationType.PRISMATIC,
                ArticulationType.CYLINDRICAL,
            }:
                dist_0t = self.articulation_params.dist * t
                dist_1t = self.articulation_params.dist * (t - 1)
                means_mobile_0t = means_mobile_0t + axis * dist_0t
                means_mobile_1t = means_mobile_1t + axis * dist_1t
            opacities_static = torch.cat(
                (
                    opacities[mask_tgt_0] * (1 - mobilities[mask_tgt_0]),
                    opacities[mask_tgt_1] * (1 - mobilities[mask_tgt_1]),
                )
            )
            opacities_mobile = torch.cat(
                (opacities[mask_tgt_0] * mobilities[mask_tgt_0], opacities[mask_tgt_1] * mobilities[mask_tgt_1])
            )
            means_static = torch.cat((means[mask_tgt_0], means[mask_tgt_1]))
            means_mobile = torch.cat((means_mobile_0t, means_mobile_1t))
            scales_static = torch.cat((scales[mask_tgt_0], scales[mask_tgt_1]))
            scales_mobile = torch.cat((scales[mask_tgt_0], scales[mask_tgt_1]))
            quats_static = torch.cat((quats[mask_tgt_0], quats[mask_tgt_1]))
            quats_mobile = torch.cat((quats_mobile_0t, quats_mobile_1t))
            colors_static = torch.cat((colors[mask_tgt_0], colors[mask_tgt_1]))
            colors_mobile = torch.cat((colors[mask_tgt_0], colors[mask_tgt_1]))
            motion_ids_static = torch.full((len(opacities_static),), -1, device=self.device)
            motion_ids_mobile = torch.cat(
                (
                    torch.full((mask_tgt_0.sum(),), 0, device=self.device),
                    torch.full((mask_tgt_1.sum(),), 1, device=self.device),
                )
            )
            if gen_part_seg:
                colors_part_seg_static = static_color.repeat(len(colors_static), 1)
                colors_part_seg_mobile = mobile_color.repeat(len(colors_mobile), 1)
            if mobility is None:
                if mode == "train":
                    use_ids_tgt_0 = torch.where(mask_tgt_0)[0]
                    use_ids_tgt_1 = torch.where(mask_tgt_1)[0]
                    self.info.append(
                        {"use_ids": torch.cat((use_ids_tgt_0, use_ids_tgt_1, use_ids_tgt_0, use_ids_tgt_1))}
                    )
                self.update_outputs(
                    outputs,
                    camera,
                    c2w,
                    torch.cat((opacities_static, opacities_mobile)),
                    torch.cat((means_static, means_mobile)),
                    torch.cat((scales_static, scales_mobile)),
                    torch.cat((quats_static, quats_mobile)),
                    torch.cat((colors_static, colors_mobile)),
                    motion_quats=motion_quats,
                    motion_ids=torch.cat((motion_ids_static, motion_ids_mobile)),
                    colors_part_seg=(
                        torch.cat((colors_part_seg_static, colors_part_seg_mobile)) if gen_part_seg else None
                    ),
                    suffix="" if mode == "test" else "_cmr",
                )
            if mobility == 0 or mode == "val":
                self.update_outputs(
                    outputs,
                    camera,
                    c2w,
                    opacities_static,
                    means_static,
                    scales_static,
                    quats_static,
                    colors_static,
                    motion_quats=motion_quats,
                    motion_ids=motion_ids_static,
                    colors_part_seg=colors_part_seg_static if gen_part_seg else None,
                    suffix="" if mode == "test" else "_s",
                )
            if mobility == 1 or mode == "val":
                self.update_outputs(
                    outputs,
                    camera,
                    c2w,
                    opacities_mobile,
                    means_mobile,
                    scales_mobile,
                    quats_mobile,
                    colors_mobile,
                    motion_quats=motion_quats,
                    motion_ids=motion_ids_mobile,
                    colors_part_seg=colors_part_seg_mobile if gen_part_seg else None,
                    suffix="" if mode == "test" else "_m",
                )
        return outputs

    def update_outputs(
        self,
        outputs,
        cam,
        c2w,
        opacities,
        means,
        scales,
        quats,
        colors,
        motion_quats=None,
        motion_ids=None,
        colors_part_seg=None,
        suffix="",
    ):
        """motion_quats & motion_ids: some of the input gaussians may result from transforming the original ones. If such transformations involve rotation, the direction with which to query their spherical harmonics need to be updated accordingly. motion_quats[i] is the quaternion representing the rotation component of the ith transformation. motion_ids[j] indicates which transformation the jth gaussian takes. If no transformation is involved for the jth gaussian, motion_ids[j] should be -1."""
        # for compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in {"antialiased", "classic"}:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        camera_scale_fac = self._get_downscale_factor()
        cam.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(c2w)
        K = cam.get_intrinsics_matrices().cuda()
        W, H = int(cam.width.item()), int(cam.height.item())
        if self.training:
            self.last_size = (H, W)
        cam.rescale_output_resolution(camera_scale_fac)  # type: ignore

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors = torch.sigmoid(colors).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, info = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmat,  # [1, 4, 4]
            K,  # [1, 3, 3]
            W,
            H,
            sh_degree=sh_degree_to_use,
            packed=False,
            render_mode="RGB+D",
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
            motion_quats=motion_quats,
            motion_ids=motion_ids,
        )
        if self.training:
            self.strategy.step_pre_backward(self.gauss_params, self.optimizers, self.strategy_state, self.step, info)
            self.info[-1].update(info)

        if "background" not in outputs:
            outputs["background"] = self._get_background_color()
        background = outputs["background"]
        rgb = render[..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)
        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training and cam.metadata is not None and "cam_idx" in cam.metadata:
            rgb = self._apply_bilateral_grid(rgb, cam.metadata["cam_idx"], H, W)
        depth = render[..., 3:]

        outputs.update(
            {
                f"rgb{suffix}": rgb.squeeze(0),  # type: ignore
                f"depth{suffix}": depth.squeeze(0),  # type: ignore
                f"accumulation{suffix}": alpha.squeeze(0),  # type: ignore
            }
        )  # type: ignore

        if colors_part_seg is not None:
            # currently only intended for visualization or evaluation
            with torch.inference_mode():
                outputs[f"part-seg{suffix}"] = rasterization(
                    means,
                    quats,
                    scales,
                    opacities,
                    colors_part_seg,
                    viewmat,  # [1, 4, 4]
                    K,  # [1, 3, 3]
                    W,
                    H,
                    packed=False,
                    render_mode="RGB",
                    rasterize_mode=self.config.rasterize_mode,
                    # set some threshold to disregrad small gaussians for faster rendering.
                    # radius_clip=3.0,
                    motion_quats=motion_quats,
                    motion_ids=motion_ids,
                )[0].squeeze(0)

    @torch.inference_mode()
    def get_outputs_for_camera(
        self,
        camera: Cameras,
        obb_box: Optional[OrientedBox] = None,
        mobility=None,
        mode: Literal["val", "test"] = "test",
        gen_part_seg=False,
    ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device), mobility=mobility, mode=mode, gen_part_seg=gen_part_seg)
        return outs  # type: ignore

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        mask_ref = self.mobilities.isnan()
        mask_tgt = ~mask_ref
        mask_0 = self.states == 0
        mask_1 = self.states == 1
        metrics_dict["n_ref_0"] = (mask_ref & mask_0).sum()
        metrics_dict["n_ref_1"] = (mask_ref & mask_1).sum()
        metrics_dict["n_tgt_0"] = (mask_tgt & mask_0).sum()
        metrics_dict["n_tgt_1"] = (mask_tgt & mask_1).sum()
        metrics_dict["mean_mobility"] = self.mobilities[mask_tgt].sigmoid().mean()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.get_gt_img(batch["image"])
        gt_img = self.composite_with_background(gt_img, outputs["background"])
        pred_img = outputs["rgb"]
        pred_depth = outputs["depth"]
        if "rgb_csr" in outputs:
            cr = "csr"
        elif "rgb_cmr" in outputs:
            cr = "cmr"
        else:
            cr = None
        if cr:
            pred_img_cr = outputs[f"rgb_{cr}"]
            pred_depth_cr = outputs[f"depth_{cr}"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch['mask'] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask
            pred_depth = pred_depth * mask
            if cr:
                pred_img_cr = pred_img_cr * mask
                pred_depth_cr = pred_depth_cr * mask

        loss_dict = {
            "main_loss": l1_loss(pred_img, gt_img) * (1 - self.config.ssim_lambda)
            + (1 - self.ssim(pred_img.permute(2, 0, 1)[None], gt_img.permute(2, 0, 1)[None])) * self.config.ssim_lambda,
            "opacity_reg": self.opacities.sigmoid().mean() * self.config.lambda_opacity_reg,
        }
        if cr:
            loss_dict[f"main_loss_{cr}"] = (
                l1_loss(pred_img_cr, gt_img) * (1 - self.config.ssim_lambda)
                + (1 - self.ssim(pred_img_cr.permute(2, 0, 1)[None], gt_img.permute(2, 0, 1)[None]))
                * self.config.ssim_lambda
            )
            if cr == "csr":
                loss_dict["mobility_reg"] = outputs["mobilities_other"].mean() * self.config.lambda_mobility_reg_photo
                if self.config.enable_depth_consistency:
                    pred_depth = pred_depth.detach()
                    pred_depth_grad = sobel(pred_depth.permute(2, 0, 1)[None])[0].permute(1, 2, 0)
                    loss_dict[f"depth_loss_{cr}"] = (
                        relu(
                            (pred_depth - pred_depth_cr).abs() * (-pred_depth_grad * self.config.depth_grad_gamma).exp()
                            - self.config.depth_tolerance
                        )
                        ** 2
                    ).mean() * self.config.lambda_depth_consistency_loss

        if self.config.use_scale_regularization:
            if self.step % 10 == 0:
                scale_exp = torch.exp(self.scales)
                scale_reg = (
                    torch.maximum(
                        scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(self.config.max_gauss_ratio)
                    )
                    - self.config.max_gauss_ratio
                )
                scale_reg = 0.1 * scale_reg.mean()
            else:
                scale_reg = torch.tensor(0.0).to(self.device)
            loss_dict["scale_reg"] = scale_reg

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict

    def _optimize(
        self,
        optim,
        opacities=None,
        means=None,
        mobilities=None,
        masks=None,
        m=10,
        r_th=1e-3,
        mode: Literal["cm", "cs", "m"] = "cm",
        vis_cb=None,
        desc="",
    ):
        avg_loss = None
        pbar = tqdm()
        imgs = []
        i = 0
        while True:
            optim.zero_grad()
            loss = self.compute_geometric_loss(
                opacities=opacities, means=means, mobilities=mobilities, masks=masks, mode=mode
            )
            if mode == "cs":
                loss = loss + self.mobilities.sigmoid().nanmean() * self.config.lambda_mobility_reg_geom
            loss.backward()
            optim.step()
            if avg_loss is None:
                r = 1
                avg_loss = loss
            else:
                r = 1 - loss / avg_loss
                avg_loss = ((m - 1) * avg_loss + loss) / m
            pbar.set_description_str(f"{desc} loss: {loss:.4g}, r: {r:.3g}")
            pbar.update()
            if not (i % 10 or vis_cb is None):
                imgs.append(vis_cb())
            i += 1
            if abs(r) < r_th or i == self.config.n_max_optim_iters:
                break
        pbar.close()
        return loss, imgs

    def optimize_mobilities(self, mode: Literal["cm", "cs", "m"] = "cm", vis_cb=None):
        opacities = self.opacities.detach().sigmoid().squeeze(-1)
        mask_ref = self.mobilities.isnan().squeeze(-1)
        mask_tgt = ~mask_ref
        states = self.states.squeeze(-1)
        mask_0 = states == 0
        mask_1 = states == 1
        masks = {
            "ref0": mask_ref & mask_0,
            "ref1": mask_ref & mask_1,
            "tgt0": mask_tgt & mask_0,
            "tgt1": mask_tgt & mask_1,
        }
        pgs = self.get_param_groups()
        optim = Adam(pgs["mobilities"], lr=3 if mode == "cs" else 3e-1)
        return self._optimize(
            optim,
            opacities=opacities,
            masks=masks,
            mode=mode,
            vis_cb=vis_cb,
            desc=("Estimating" if mode == "cs" else "Refining") + f" mobilities only.",
        )[1]

    def optimize_mobilities_and_articulation(
        self, seed=None, n_m_tries=None, n_cm_tries=None, regularize_angle=True, vis_cb=None
    ):
        opacities = self.opacities.detach().sigmoid().squeeze(-1)
        mask_ref = self.mobilities.isnan().squeeze(-1)
        mask_tgt = ~mask_ref
        states = self.states.squeeze(-1)
        mask_0 = states == 0
        mask_1 = states == 1
        masks = {
            "ref0": mask_ref & mask_0,
            "ref1": mask_ref & mask_1,
            "tgt0": mask_tgt & mask_0,
            "tgt1": mask_tgt & mask_1,
        }
        pgs = self.get_param_groups()
        lrs = {"axis": 3e-2, "pivot": 3e-2, "dist": 1e-2, "angle": 1e-1}
        loss_lst = []
        imgs_lst = []
        best_loss = None
        best_articulation_params = ArticulationParams.get_empty()
        if seed is not None:
            torch.manual_seed(seed)
        if n_m_tries is None:
            n_m_tries = self.config.n_optim_tries
        if n_cm_tries is None:
            n_cm_tries = self.config.n_optim_tries
        assert n_m_tries + n_cm_tries > 0
        for i_try in range(n_m_tries + n_cm_tries + bool(n_m_tries)):
            if i_try < n_m_tries:
                mode = "m"
            else:
                mode = "cm"
            desc = f'Try {i_try}: estimating articulation params only in "{mode}gc" mode.'
            f = i_try == n_m_tries and n_m_tries
            if f:
                self.articulation_params.update(best_articulation_params)
                optim = Adam([{"params": pgs[k], "lr": lrs[k] / 10} for k in pgs if k in lrs])
                desc += f" Continued from try {best_try}."
                best_loss = None
            else:
                self.articulation_params.randomize()
                optim = Adam([{"params": pgs[k], "lr": lrs[k]} for k in pgs if k in lrs])
            loss, imgs = self._optimize(optim, opacities=opacities, masks=masks, mode=mode, vis_cb=vis_cb, desc=desc)
            loss_lst.append(loss.item())
            imgs_lst.append((imgs_lst[best_try] if f else []) + imgs)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_try = i_try
                best_optim = optim
                best_articulation_params.update(self.articulation_params)
                print(f"New best!")
        self.articulation_params.update(best_articulation_params)
        optim = best_optim
        optim.add_param_group({"params": pgs["mobilities"], "lr": 1})
        imgs = self._optimize(
            optim,
            opacities=opacities,
            masks=masks,
            r_th=1e-4,
            vis_cb=vis_cb,
            desc=f"Jointly estimating articulation params and mobilities. Continued from try {best_try}.",
        )[1]
        imgs_lst[best_try].extend(imgs)
        axis = self.articulation_params.axis
        axis.data /= axis.norm()
        if self.articulation_params.articulation_type != ArticulationType.PRISMATIC and regularize_angle:
            angle = self.articulation_params.angle
            angle.data %= np.pi * 2
            if angle > np.pi:
                angle.data = np.pi * 2 - angle
                axis.data *= -1
        return loss_lst, imgs_lst, best_try

    def compute_geometric_loss(
        self, opacities=None, means=None, mobilities=None, masks=None, mode: Literal["cm", "cs", "m"] = "cm"
    ):
        """cm: cross-mobile geometric consistency; cs: cross-static geometric consistency; m: mobile-only geometric consistency"""

        def weighted_chamfer_distance(means1, weights1, means2, weights2, tau=0.1):
            dist1 = chamfer_distance(
                means1[None],
                means2[weights2 > tau][None],
                batch_reduction=None,
                point_reduction=None,
                single_directional=True,
            )[0]
            dist2 = chamfer_distance(
                means2[None],
                means1[weights1 > tau][None],
                batch_reduction=None,
                point_reduction=None,
                single_directional=True,
            )[0]
            return (dist1 * weights1).sum() / weights1.sum() + (dist2 * weights2).sum() / weights2.sum()

        if opacities is None:
            opacities = self.opacities.sigmoid().squeeze(-1)
        if means is None:
            means = self.means
        if mobilities is None:
            mobilities = self.mobilities.sigmoid().squeeze(-1)
        if masks is None:
            mask_ref = mobilities.isnan()
            mask_tgt = ~mask_ref
            states = self.states.squeeze(-1)
            mask_0 = states == 0
            mask_1 = states == 1
            masks = {
                "ref0": mask_ref & mask_0,
                "ref1": mask_ref & mask_1,
                "tgt0": mask_tgt & mask_0,
                "tgt1": mask_tgt & mask_1,
            }
        if mode in {"cm", "m"}:
            means_mobile_01 = means[masks["tgt0"]]  # 01 denotes from state 0 to 1
            means_mobile_10 = means[masks["tgt1"]]
            axis = normalize(self.articulation_params.axis, dim=0)
            if self.articulation_params.articulation_type.item() in {
                ArticulationType.REVOLUTE,
                ArticulationType.CYLINDRICAL,
            }:
                pivot = self.articulation_params.pivot
                angle_01 = self.articulation_params.angle
                rot_01 = axis_angle_to_quaternion(axis * angle_01)
                means_mobile_01 = quaternion_apply(rot_01, means_mobile_01 - pivot) + pivot
                angle_10 = -self.articulation_params.angle
                rot_10 = axis_angle_to_quaternion(axis * angle_10)
                means_mobile_10 = quaternion_apply(rot_10, means_mobile_10 - pivot) + pivot
            if self.articulation_params.articulation_type.item() in {
                ArticulationType.PRISMATIC,
                ArticulationType.CYLINDRICAL,
            }:
                dist_01 = self.articulation_params.dist
                means_mobile_01 = means_mobile_01 + axis * dist_01
                dist_10 = -self.articulation_params.dist
                means_mobile_10 = means_mobile_10 + axis * dist_10
            opacities_m0 = opacities[masks["tgt0"]] * mobilities[masks["tgt0"]]
            opacities_m1 = opacities[masks["tgt1"]] * mobilities[masks["tgt1"]]
        if mode == "m":
            return weighted_chamfer_distance(
                means[masks["tgt0"]], opacities_m0, means_mobile_10, opacities_m1, tau=self.config.cull_alpha_thresh
            )
        means_tgt0 = torch.cat((means[masks["tgt0"]], means[masks["tgt1"]]))
        opacities_tgt0 = torch.cat(
            (opacities[masks["tgt0"]], opacities[masks["tgt1"]] * (1 - mobilities[masks["tgt1"]]))
        )
        means_tgt1 = torch.cat((means[masks["tgt1"]], means[masks["tgt0"]]))
        opacities_tgt1 = torch.cat(
            (opacities[masks["tgt1"]], opacities[masks["tgt0"]] * (1 - mobilities[masks["tgt0"]]))
        )
        if mode == "cm":
            means_tgt0 = torch.cat((means_tgt0, means_mobile_10))
            opacities_tgt0 = torch.cat((opacities_tgt0, opacities_m1))
            means_tgt1 = torch.cat((means_tgt1, means_mobile_01))
            opacities_tgt1 = torch.cat((opacities_tgt1, opacities_m0))
        return weighted_chamfer_distance(
            means[masks["ref0"]],
            opacities[masks["ref0"]],
            means_tgt0,
            opacities_tgt0,
            tau=self.config.cull_alpha_thresh,
        ) + weighted_chamfer_distance(
            means[masks["ref1"]],
            opacities[masks["ref1"]],
            means_tgt1,
            opacities_tgt1,
            tau=self.config.cull_alpha_thresh,
        )

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], camera: Cameras
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        K = camera.get_intrinsics_matrices()[0]
        c2w = camera.camera_to_worlds[0]
        gt_rgb_vis = self.vis_articulation(gt_rgb, K, c2w, self.gt_articulation_params, color=(0, 255, 0), text="gt")
        if "rgb_csr" in outputs:
            cr = "csr"
        elif "rgb_cmr" in outputs:
            cr = "cmr"
        else:
            cr = None
        rgb = outputs["rgb"]
        depth = outputs["depth"]
        if cr == "cmr":
            self.regularize_pivot()
        articulation_params = self.articulation_params if cr == "cmr" else None
        rgb_vis = self.vis_articulation(rgb, K, c2w, articulation_params, color=(0, 128, 255), text="ref")
        combined_rgb = torch.cat((gt_rgb_vis, rgb_vis))

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None]
        rgb = torch.moveaxis(rgb, -1, 0)[None]
        psnr = self.psnr(gt_rgb, rgb)
        ssim = self.ssim(gt_rgb, rgb)
        lpips = self.lpips(gt_rgb, rgb)
        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            cc_ssim = self.ssim(gt_rgb, cc_rgb)
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        if cr:
            rgb_cr = outputs[f"rgb_{cr}"]
            rgb_cr_vis = self.vis_articulation(rgb_cr, K, c2w, articulation_params, color=(0, 128, 255), text=cr)
            rgb_cr_err = self.vis_articulation(
                (rgb[0].permute(1, 2, 0) - rgb_cr).abs(),
                K,
                c2w,
                articulation_params,
                color=(0, 128, 255),
                text=f"{cr}_err",
            )
            rgb_s = self.vis_articulation(outputs["rgb_s"], K, c2w, articulation_params, color=(0, 128, 255), text="s")
            rgb_m = self.vis_articulation(outputs["rgb_m"], K, c2w, articulation_params, color=(0, 128, 255), text="m")
            combined_rgb = torch.cat(
                (combined_rgb, torch.cat((rgb_cr_vis, rgb_cr_err)), torch.cat((rgb_s, rgb_m))), dim=1
            )

            depth_grad = sobel(depth.permute(2, 0, 1)[None])[0].permute(1, 2, 0)
            depth_grad_weight = (-depth_grad * self.config.depth_grad_gamma).exp()
            depth_diff = (depth - outputs[f"depth_{cr}"]).abs() * depth_grad_weight
            depth = torch.cat(
                (
                    torch.cat((depth, depth_grad)),
                    torch.cat((outputs[f"depth_{cr}"], depth_diff)),
                    torch.cat((outputs["depth_s"], outputs["depth_m"])),
                ),
                dim=1,
            )

            rgb_cr = torch.moveaxis(rgb_cr, -1, 0)[None]
            psnr_cr = self.psnr(gt_rgb, rgb_cr)
            ssim_cr = self.ssim(gt_rgb, rgb_cr)
            lpips_cr = self.lpips(gt_rgb, rgb_cr)
            metrics_dict.update(
                {f"psnr_{cr}": float(psnr_cr.item()), f"ssim_{cr}": float(ssim_cr), f"lpips_{cr}": float(lpips_cr)}
            )

        depth_vis = to_np_color(colormaps.apply_depth_colormap(depth))
        put_texts(depth_vis, "ref", color=(0, 128, 255))
        if cr:
            h, w = gt_rgb.shape[-2:]
            put_texts(depth_vis[h:], "grad", color=(0, 128, 255))
            put_texts(depth_vis[:, w:], cr, color=(0, 128, 255))
            put_texts(depth_vis[h:, w:], "diff", color=(0, 128, 255))
            put_texts(depth_vis[:, w * 2 :], "s", color=(0, 128, 255))
            put_texts(depth_vis[h:, w * 2 :], "m", color=(0, 128, 255))
        depth_vis = to_pt_color(depth_vis, device=self.device)

        images_dict = {"img": combined_rgb, "depth": depth_vis}

        return metrics_dict, images_dict

    @torch.inference_mode()
    def vis_articulation(self, img, K, c2w, articulation_params, color=(255, 0, 0), text="", angle_mult=0.1):
        img = to_np_color(img)
        put_texts(img, text, color=color)
        if articulation_params and articulation_params.is_valid:
            K = K.numpy()
            c2w = homogenize_transforms(c2w).cpu().numpy()
            p1 = articulation_params.pivot
            p2 = p1 + normalize(articulation_params.axis, dim=0) * (
                articulation_params.dist
                if articulation_params.articulation_type.item()
                in {ArticulationType.PRISMATIC, ArticulationType.CYLINDRICAL}
                else articulation_params.angle * angle_mult
            )
            pts = homogenize_vecs(torch.stack((p1, p2))).cpu().numpy()
            w2c = np.linalg.inv(ch_cam_pose_spec(c2w, 2, 1))
            pts_h = K @ (w2c @ pts)[:, :3]
            pts_i = np.round(pts_h[:, :2] / pts_h[:, [2]]).astype(int).squeeze(-1)
            cv2.arrowedLine(img, pts_i[0], pts_i[1], color, thickness=2)
        return to_pt_color(img, device=self.device)

    @torch.inference_mode()
    def regularize_pivot(self):
        """Since the pivot can move freely along the axis, we regularize the pivot to be the closest point to the ground-truth."""
        # For the prismatic case, the pivot is already set to either the ground-truth (if available) or the origin (otherwise).
        if self.articulation_params.articulation_type != ArticulationType.PRISMATIC:
            axis = normalize(self.articulation_params.axis, dim=0)
            self.articulation_params.pivot += (
                (self.gt_articulation_params.pivot - self.articulation_params.pivot) @ axis * axis
            )

    @torch.inference_mode()
    def evaluate_articulation(self, params=None, gt_params=None):
        """Evaluates the articulation parameters against the ground-truth."""
        if params is None:
            params = self.articulation_params
        assert params.is_valid
        if gt_params is None:
            gt_params = self.gt_articulation_params
        assert gt_params and gt_params.is_valid and params.articulation_type == gt_params.articulation_type
        axis_err = angle(params.axis, gt_params.axis)
        if axis_err > torch.pi / 2:
            axis_err = torch.pi - axis_err
        metrics = {"axis": axis_err.rad2deg().item()}
        T_pred = params.to_matrix().cpu().numpy()
        T_gt = gt_params.to_matrix().cpu().numpy()
        r, t = compute_trans_diff(T_pred, T_gt)
        if gt_params.articulation_type != ArticulationType.PRISMATIC:
            perp_dir = normalize(torch.linalg.cross(params.axis, gt_params.axis), dim=0)
            metrics["pivot"] = torch.dot(gt_params.pivot - params.pivot, perp_dir).abs().item()
            metrics["r"] = r.item()
        if gt_params.articulation_type != ArticulationType.REVOLUTE:
            metrics["t"] = t.item()
        return metrics
