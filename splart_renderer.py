import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import torch
import tyro
from imageio.v3 import imread, imwrite
from nerfstudio.cameras.cameras import Cameras
from tqdm import tqdm
from tqdm.contrib import tzip
from vis_utils.utils import (
    gen_spherical_spiral_poses,
    get_cur_timestamp,
    homogenize_transforms,
    put_texts,
    reset_dir,
    to_np_color,
    to_np_depth,
)

from splart.splart import SplartModelConfig
from splart.splart_dataparser import SplartDataParserConfig


class SplartRenderer:
    def __init__(self, ckpt_dir, load_step=None, data_dir=None, background="random", device="cuda") -> None:
        if load_step is None or not (ckpt_path := ckpt_dir / f"step-{load_step:09d}.ckpt").exists():
            # load the latest checkpoint
            ckpt_path = max(ckpt_dir.iterdir())
            load_step = int(ckpt_path.stem[5:])
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = {key[7:]: val for key, val in state["pipeline"].items() if key.startswith("_model.")}
        self.model = SplartModelConfig(background_color=background).setup(scene_box=None, num_train_data=0).to(device)
        self.model.load_state_dict(state)
        self.model.step = load_step
        self.model.eval()
        print(f"loaded checkpoint at {ckpt_path}")
        with (ckpt_dir.parent / "dataparser_transforms.json").open() as f:
            meta = json.load(f)
        self.ns_s_raw = meta["scale"]
        ns_S_raw = torch.diag(torch.tensor([self.ns_s_raw] * 3 + [1], device=device))
        self.raw_S_ns = ns_S_raw.inverse()
        self.ns_T_raw = ns_S_raw @ homogenize_transforms(torch.tensor(meta["transform"], device=device))
        self.should_undistort = False
        if data_dir is not None:
            dataparser = SplartDataParserConfig(data=data_dir, downscale_factor=1).setup()
            self.train_dataparser_outputs = dataparser.get_dataparser_outputs()
            self.val_dataparser_outputs = dataparser.get_dataparser_outputs(split="val")
            self.test_dataparser_outputs = dataparser.get_dataparser_outputs(split="test")
            cam = self.train_dataparser_outputs.cameras[0]
            distortion_params = cam.distortion_params.numpy()
            if np.any(distortion_params):
                self.should_undistort = True
                K = np.array(
                    ((cam.fx.item(), 0, cam.cx.item() - 0.5), (0, cam.fy.item(), cam.cy.item() - 0.5), (0, 0, 1))
                )
                distortion_params = np.concatenate((distortion_params[[0, 1, 4, 5, 2, 3]], (0, 0)))
                K_opt, (self.x, self.y, self.w, self.h) = cv2.getOptimalNewCameraMatrix(
                    K, distortion_params, (cam.width.item(), cam.height.item()), 0
                )
                self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                    K, distortion_params, None, K_opt, (self.w, self.h), cv2.CV_32FC1
                )
                self.cam_info = {
                    "fx": K_opt[0, 0],
                    "fy": K_opt[1, 1],
                    "cx": K_opt[0, 2] + 0.5,
                    "cy": K_opt[1, 2] + 0.5,
                    "width": self.w,
                    "height": self.h,
                }
            else:
                self.cam_info = {
                    "fx": cam.fx,
                    "fy": cam.fy,
                    "cx": cam.cx,
                    "cy": cam.cy,
                    "width": cam.width,
                    "height": cam.height,
                }
        self.device = device

    def load_img(self, img):
        if isinstance(img, Path):
            img = imread(img)
        if self.should_undistort:
            img = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)
            return img[self.y : self.y + self.h, self.x : self.x + self.w]
        return img

    def render_view(
        self,
        cam,
        mobility=None,
        articulation_state=None,
        vis_gt_articulation=False,
        vis_articulation=True,
        pose_type="cam2raw",
        color_only=True,
        gen_part_seg=False,
    ):
        """If color_only, gen_part_seg determines whether to return part segmentation or rgb.
        If not color_only, gen_part_seg determines whether to include part segmentation in the outputs."""
        if isinstance(cam, Cameras):
            cam = deepcopy(cam)
            if cam.distortion_params is not None:
                distortion_params = cam.distortion_params.numpy()
                if np.any(distortion_params):
                    K = np.array(
                        ((cam.fx.item(), 0, cam.cx.item() - 0.5), (0, cam.fy.item(), cam.cy.item() - 0.5), (0, 0, 1))
                    )
                    distortion_params = np.concatenate((distortion_params[[0, 1, 4, 5, 2, 3]], (0, 0)))
                    K_opt, (_, _, w, h) = cv2.getOptimalNewCameraMatrix(
                        K, distortion_params, (cam.width.item(), cam.height.item()), 0
                    )
                    cam.fx[0] = K_opt[0, 0]
                    cam.fy[0] = K_opt[1, 1]
                    cam.cx[0] = K_opt[0, 2] + 0.5
                    cam.cy[0] = K_opt[1, 2] + 0.5
                    cam.width[0] = w
                    cam.height[0] = h
        else:
            cam = Cameras(torch.as_tensor(cam), **self.cam_info)
        if articulation_state is not None:
            if cam.metadata is None:
                cam.metadata = {}
            cam.metadata["articulation_state"] = torch.full((*cam.shape, 1), articulation_state)
        cam = cam.to(self.device)
        if not cam.ndim:
            cam = cam.reshape(1)
        if pose_type == "cam2raw":
            cam.camera_to_worlds = self.ns_T_raw @ homogenize_transforms(cam.camera_to_worlds) @ self.raw_S_ns
        else:
            assert pose_type == "cam2ns", f"invalid pose_type: {pose_type}"
        outputs = self.model.get_outputs(cam, mobility=mobility, gen_part_seg=gen_part_seg)
        if vis_gt_articulation:
            outputs["rgb"] = self.model.vis_articulation(
                outputs["rgb"],
                cam.get_intrinsics_matrices()[0],
                cam.camera_to_worlds[0],
                self.model.gt_articulation_params,
                color=(0, 255, 0),
            )
        if vis_articulation:
            outputs["rgb"] = self.model.vis_articulation(
                outputs["rgb"],
                cam.get_intrinsics_matrices()[0],
                cam.camera_to_worlds[0],
                self.model.articulation_params,
            )
        if color_only:
            if gen_part_seg:
                return outputs["part-seg"]
            return outputs["rgb"]
        outputs["depth"] /= self.ns_s_raw
        return outputs

    @torch.inference_mode()
    def render_views(
        self,
        cams,
        output_dir,
        name=None,
        img_ids=None,
        show_img_ids=False,
        mobility=None,
        articulation_states=None,
        vis_articulation=True,
        pose_type="cam2raw",
        color_only=True,
        gen_part_seg=False,
        animate=0,
        return_results=False,
    ):
        if name is None:
            name = get_cur_timestamp()
        match mobility:
            case 0:
                name += "_static"
            case 1:
                name += "_mobile"
        output_dir /= name
        color_type = "part-seg" if color_only and gen_part_seg else "color"
        if animate:
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            reset_dir(output_dir / color_type)
            if not color_only:
                reset_dir(output_dir / "depth")
                if gen_part_seg:
                    reset_dir(output_dir / "part-seg")
        if isinstance(cams, Cameras) and not cams.ndim:
            cams = cams.reshape(1)
        if hasattr(articulation_states, "__len__"):
            assert len(articulation_states) == len(cams)
        else:
            articulation_states = [articulation_states] * len(cams)
        if img_ids is None:
            img_ids = [f"{i:04d}" for i in range(len(cams))]
        else:
            assert hasattr(img_ids, "__len__") and len(img_ids) == len(cams)
        if animate or return_results:
            results = defaultdict(list)
        for img_id, cam, articulation_state in tzip(img_ids, cams, articulation_states):
            outputs = self.render_view(
                cam,
                mobility=mobility,
                articulation_state=articulation_state,
                vis_articulation=vis_articulation,
                pose_type=pose_type,
                color_only=color_only,
                gen_part_seg=gen_part_seg,
            )
            if color_only:
                color = to_np_color(outputs)
            else:
                color = to_np_color(outputs["rgb"])
                depth = to_np_depth(outputs["depth"])
                if gen_part_seg:
                    part_seg = to_np_color(outputs["part-seg"])
            if show_img_ids:
                put_texts(color, img_id, color=(0, 120, 240))
            if animate or return_results:
                results[color_type].append(color)
                if not color_only:
                    results["depth"].append(depth)
                    if gen_part_seg:
                        results["part-seg"].append(part_seg)
            if not animate:
                imwrite(output_dir / f"{color_type}/{img_id}.jpg", color, quality=100)
                if not color_only:
                    imwrite(output_dir / f"depth/{img_id}.png", depth)
                    if gen_part_seg:
                        imwrite(output_dir / f"part-seg/{img_id}.png", part_seg)
        if animate:
            imwrite(output_dir / f"{color_type}.mp4", results[color_type], fps=animate, quality=10)
            if not color_only:
                imwrite(output_dir / "depth.mp4", results["depth"], fps=animate, quality=10)
                if gen_part_seg:
                    imwrite(output_dir / "part-seg.mp4", results["part-seg"], fps=animate, quality=10)
        if return_results:
            return results

    def render_split(
        self,
        split: Literal["train", "val", "test"],
        output_dir,
        name=None,
        show_img_ids=True,
        mobility=None,
        vis_articulation=True,
        color_only=True,
        gen_part_seg=False,
        animate=0,
        return_results=False,
    ):
        if name is None:
            name = get_cur_timestamp()
        name += f"_{split}"
        dataparser_outputs = getattr(self, f"{split}_dataparser_outputs")
        return self.render_views(
            dataparser_outputs.cameras,
            output_dir,
            name=name,
            img_ids=[img_path.stem for img_path in dataparser_outputs.image_filenames],
            show_img_ids=show_img_ids,
            mobility=mobility,
            vis_articulation=vis_articulation,
            pose_type="cam2ns",
            color_only=color_only,
            gen_part_seg=gen_part_seg,
            animate=animate,
            return_results=return_results,
        )

    def render_articulation(
        self,
        cam,
        output_dir,
        name=None,
        min_state=0,
        max_state=1,
        n_states=60,
        vis_articulation=True,
        pose_type="cam2raw",
        color_only=True,
        gen_part_seg=False,
        animate=0,
        return_results=False,
    ):
        if name is None:
            name = f"{get_cur_timestamp()}_articulation"
        return self.render_views(
            [cam] * n_states,
            output_dir,
            name=name,
            articulation_states=np.linspace(min_state, max_state, n_states),
            vis_articulation=vis_articulation,
            pose_type=pose_type,
            color_only=color_only,
            gen_part_seg=gen_part_seg,
            animate=animate,
            return_results=return_results,
        )


def main(
    dataset: str = "splart-pms",
    scene: Optional[str] = None,
    fps: int = 30,
    duration: int = 20,
    timestamps: Optional[list[str]] = None,
    timestamp_min: Optional[str] = None,
    timestamp_max: Optional[str] = None,
    dataset_dir: Path = Path("datasets/splart"),
    output_dir: Path = Path("outputs/renders"),
):
    model_ckpts_dir = Path(f"model_ckpts/{dataset}")
    assert model_ckpts_dir.exists()
    scenes = [scene_dir.name for scene_dir in model_ckpts_dir] if scene is None else [scene]
    lst = []
    for scene in scenes:
        model_ckpt_dir = model_ckpts_dir / scene
        assert model_ckpt_dir.exists()
        if timestamps is None:
            if timestamp_min is None and timestamp_max is None:
                timestamps = [p.name for p in model_ckpt_dir.iterdir() if p.is_dir()]
                assert timestamps
                lst.append((scene, max(timestamps)))
            else:
                for p in model_ckpt_dir.iterdir():
                    timestamp = p.name
                    if (timestamp_min is None or timestamp_min <= timestamp) and (
                        timestamp_max is None or timestamp <= timestamp_max
                    ):
                        lst.append((scene, timestamp))
        else:
            for timestamp in timestamps:
                if (model_ckpt_dir / timestamp).is_dir():
                    lst.append((scene, timestamp))
    dataset_dir /= dataset
    output_dir /= dataset
    n = fps * duration
    cams = np.array(gen_spherical_spiral_poses(1, 0, np.pi / 3, -0.25, 1.25, n=n))
    cams = Cameras(torch.from_numpy(cams), 800.0, 800.0, 500.0, 500.0)
    articulation_states = (np.sin(np.arange(n) / (n - 1) * np.pi * 15 - np.pi / 2) + 1) / 2
    for scene, timestamp in tqdm(lst):
        SplartRenderer(
            max((Path(f"model_ckpts/{dataset}/{scene}/{timestamp}") / "splart").iterdir()) / "nerfstudio_models",
            data_dir=dataset_dir / scene,
        ).render_views(
            cams,
            output_dir / scene,
            name=timestamp,
            articulation_states=articulation_states,
            pose_type="cam2ns",
            color_only=False,
            gen_part_seg=True,
            animate=fps,
        )


if __name__ == "__main__":
    tyro.cli(main)
