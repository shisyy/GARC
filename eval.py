import json
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import tyro
from imageio.v3 import imread
from pytorch3d.io import IO
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from torch.nn.functional import one_hot
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import trange
from tqdm.contrib import tzip
from vis_utils.utils import (
    ch_cam_pose_spec,
    gen_spherical_poses,
    homogenize_transforms,
    mesh_denoise,
    reset_dir,
    to_np_color,
    to_np_depth,
    to_pt_color,
    to_pt_depth,
)

from splart_renderer import SplartRenderer


def eval_mesh(pred_path, gt_path, n_samples=10000):
    io = IO()
    pred_mesh = io.load_mesh(pred_path)
    gt_mesh = io.load_mesh(gt_path)
    pred_pts = sample_points_from_meshes(pred_mesh, num_samples=n_samples)
    gt_pts = sample_points_from_meshes(gt_mesh, num_samples=n_samples)
    return chamfer_distance(pred_pts, gt_pts)[0].item()


@torch.inference_mode()
def main(dataset: str, timestamp: str, load_step: Optional[int] = None, use_cached: bool = True):
    dataset_dir = Path("datasets/splart") / dataset
    spec = f"{timestamp}_{load_step}"
    output_dir = Path("outputs")
    psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()
    for scene_dir in dataset_dir.iterdir():
        scene = scene_dir.name
        metrics_save_path = output_dir / "metrics" / scene / f"{spec}.json"
        if use_cached and metrics_save_path.exists():
            with metrics_save_path.open() as f:
                metrics = json.load(f)
        else:
            model_ckpt_dir = Path(f"model_ckpts/{dataset}/{scene}/{timestamp}")
            if not model_ckpt_dir.exists():
                continue
            model_ckpt_dir = max((model_ckpt_dir / "splart").iterdir()) / "nerfstudio_models"
            splart_renderer = SplartRenderer(model_ckpt_dir, load_step=load_step, data_dir=dataset_dir / scene)
            metrics = {}

            articulation_metrics = splart_renderer.model.evaluate_articulation()
            for k in {"pivot", "t"}:
                if k in articulation_metrics:
                    articulation_metrics[k] /= splart_renderer.ns_s_raw
            metrics["articulation"] = articulation_metrics

            mesh_metrics = {}
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                splart_renderer.cam_info["width"],
                splart_renderer.cam_info["height"],
                np.array(
                    (
                        (splart_renderer.cam_info["fx"].item(), 0, splart_renderer.cam_info["cx"].item()),
                        (0, splart_renderer.cam_info["fy"].item(), splart_renderer.cam_info["cy"].item()),
                        (0, 0, 1),
                    )
                ),
            )
            cams = gen_spherical_poses(3, -np.pi * 2 / 5, np.pi * 2 / 5, m=5, n=10)
            extrinsics = np.linalg.inv(ch_cam_pose_spec(homogenize_transforms(np.array(cams)), 2, 1))
            pred_mesh_dir = output_dir / "mesh" / dataset / scene / timestamp
            reset_dir(pred_mesh_dir)
            mobility2str = {0: "static", 1: "mobile", None: "whole"}
            for mobility in {0, 1, None}:
                results = splart_renderer.render_views(
                    cams,
                    output_dir / "renders" / dataset / scene / timestamp,
                    mobility=mobility,
                    vis_articulation=False,
                    color_only=False,
                    return_results=True,
                )
                volume = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=3e-3, sdf_trunc=1.5e-2, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                )
                for i in trange(len(cams)):
                    color = o3d.geometry.Image(to_np_color(results["color"][i]))
                    depth = o3d.geometry.Image(to_np_depth(results["depth"][i]))
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color, depth, depth_trunc=5.0, convert_rgb_to_intensity=False
                    )
                    volume.integrate(rgbd, intrinsics, extrinsics[i])
                mesh = volume.extract_triangle_mesh()
                mesh_denoise(mesh)
                mobility_str = mobility2str[mobility]
                pred_mesh_path = pred_mesh_dir / f"{mobility_str}.ply"
                o3d.io.write_triangle_mesh(str(pred_mesh_path), mesh)
                gt_mesh_path = dataset_dir / scene / f"mesh/{mobility_str}.ply"
                if gt_mesh_path.exists():
                    mesh_metrics[mobility_str] = eval_mesh(pred_mesh_path, gt_mesh_path)
            metrics["mesh"] = mesh_metrics

            synthesis_metrics = {"psnr": [], "ssim": [], "lpips": [], "depth_mae": [], "part_seg_ious": []}
            results = splart_renderer.render_split(
                "test",
                output_dir / "renders" / dataset / scene / timestamp,
                name=spec,
                show_img_ids=False,
                vis_articulation=False,
                color_only=False,
                gen_part_seg=True,
                return_results=True,
            )
            background_color = splart_renderer.model._get_background_color()
            for img_path, pred_color, pred_depth, pred_seg in tzip(
                splart_renderer.test_dataparser_outputs.image_filenames,
                results["color"],
                results["depth"],
                results["part-seg"],
            ):
                pred_color_pt = to_pt_color(pred_color, channel_first=True)[None]
                gt_color = imread(img_path)
                gt_color_pt = splart_renderer.model.composite_with_background(
                    to_pt_color(gt_color), background_color
                ).permute(2, 0, 1)[None]
                synthesis_metrics["psnr"].append(psnr(pred_color_pt, gt_color_pt).item())
                synthesis_metrics["ssim"].append(ssim(pred_color_pt, gt_color_pt).item())
                synthesis_metrics["lpips"].append(lpips(pred_color_pt, gt_color_pt).item())
                p = img_path.parent.parent
                gt_depth = imread(p.with_name("depth") / img_path.relative_to(p))
                synthesis_metrics["depth_mae"].append(
                    (to_pt_depth(gt_depth) - to_pt_depth(pred_depth))[gt_depth > 0].abs().mean().item()
                )
                gt_seg = imread(p.with_name("part-seg") / img_path.relative_to(p))
                # 0: static, 1: mobile, 2: background
                gt_seg[gt_seg == 255] = 2
                gt_seg = one_hot(torch.from_numpy(gt_seg).to(device="cuda", dtype=int))
                pred_seg = to_pt_color(pred_seg)
                # blue: static, red: mobile, remainder: background
                pred_seg = torch.dstack((pred_seg[..., 2], pred_seg[..., 0], 1 - pred_seg.sum(dim=-1)))
                pred_seg = one_hot(torch.argmax(pred_seg, dim=-1))
                ious = torch.nan_to_num(
                    torch.min(gt_seg, pred_seg).sum(dim=(0, 1)) / torch.max(gt_seg, pred_seg).sum(dim=(0, 1)), 1
                )
                synthesis_metrics["part_seg_ious"].append(ious.tolist())
            synthesis_metrics = [
                {
                    "img_id": img_path.stem,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips,
                    "depth_mae": depth_mae,
                    "part_seg_ious": part_seg_ious,
                }
                for img_path, psnr, ssim, lpips, depth_mae, part_seg_ious in zip(
                    splart_renderer.test_dataparser_outputs.image_filenames,
                    synthesis_metrics["psnr"],
                    synthesis_metrics["ssim"],
                    synthesis_metrics["lpips"],
                    synthesis_metrics["depth_mae"],
                    synthesis_metrics["part_seg_ious"],
                )
            ]
            metrics["novel_synthesis"] = synthesis_metrics
            metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_save_path.unlink(missing_ok=True)
            with metrics_save_path.open("w") as f:
                json.dump(metrics, f, indent=2)
        print(f"\n{scene}")
        for k, v in metrics["articulation"].items():
            print(k, f"{v:.3e}")
        for k, v in metrics["mesh"].items():
            print(k, f"{v:.3e}")
        synthesis_metrics = metrics["novel_synthesis"]
        for metric in synthesis_metrics[0]:
            if metric == "img_id":
                continue
            v = np.mean([img[metric] for img in synthesis_metrics], axis=0)
            match metric:
                case "psnr":
                    print(metric, f"{v:.4g}")
                case k if k in {"ssim", "lpips", "depth_mae"}:
                    print(metric, f"{v:.4f}")
                case "part_seg_ious":
                    print(metric, [f"{x:.4f}" for x in v])
                case _:
                    print(metric, f"{v:.3e}")


if __name__ == "__main__":
    tyro.cli(main)
