from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal, Optional, Type

import torch
import torchvision.utils as vutils
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch.cuda.amp.grad_scaler import GradScaler

from splart.articulation_params import ArticulationParams, ArticulationType


@dataclass
class SplartPipelineConfig(VanillaPipelineConfig):
    """Configuration for the SplArt pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SplartPipeline)
    """target class to instantiate"""


class SplartPipeline(VanillaPipeline):
    """Pipeline class for the SplArt setup."""

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        if "articulation_params" in self.datamanager.train_dataset.metadata:
            self.model.gt_articulation_params.update(
                ArticulationParams(self.datamanager.train_dataset.metadata["articulation_params"])
            )
            if self.model.articulation_params.articulation_type == ArticulationType.UNDEFINED:
                self.model.articulation_params.articulation_type = self.model.gt_articulation_params.articulation_type
        else:
            assert self.model.articulation_params.articulation_type != ArticulationType.UNDEFINED
        if (
            not self.model.gt_articulation_params.is_valid
            or "pivot" not in self.datamanager.train_dataset.metadata["articulation_params"]
        ):
            self.model.gt_articulation_params.pivot = torch.zeros(3, device=device)
        if self.model.articulation_params.articulation_type == ArticulationType.PRISMATIC:
            self.model.articulation_params.pivot.data = self.model.gt_articulation_params.pivot

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera, mode="val")
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch, camera)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_image_metrics(
        self,
        data_loader,
        image_prefix: str,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the dataset and get the average.

        Args:
            data_loader: the data loader to iterate over
            image_prefix: prefix to use for the saved image filenames
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all images...", total=num_images)
            idx = 0
            for camera, batch in data_loader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera, mode="val")
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch, camera)
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C] order
                        vutils.save_image(
                            image.permute(2, 0, 1).cpu(), output_path / f"{image_prefix}_{key}_{idx:04d}.png"
                        )

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )

        self.train()
        return metrics_dict
