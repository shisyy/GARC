import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

import cv2
import imageio
import numpy as np
import tyro
from nerfstudio.utils.scripts import run_command
from tqdm import tqdm
from vis_utils.utils import reset_dir
from vis_utils.visualizer import Visualizer


def main(
    dataset_dir: Path,
    output_dir: Optional[Path] = None,
    vid_ids: Optional[set[str]] = None,
    vid_exts: Optional[set[str]] = None,
    fps: Optional[float] = None,
    downsample: Optional[int] = None,
    n_frames: Optional[int] = None,
    max_dim: Optional[int] = None,
    sfm_tool: Literal["hloc", "colmap"] = "hloc",
    extract_images: bool = False,
    run_sfm: bool = False,
    vis: bool = False,
):
    """Prepare datasets of calibrated RGB images from video(s)

    Args:
        dataset_dir: the directory containing source videos
        output_dir: the directory to save the prepared datasets; if None, will use dataset-dir
        vid_ids: the video ids to process; files starting with the same vid-id will be jointly treated; if None, will try all
        vid_exts: the video extensions to process; if None, will try all
        fps: the frame rate at which images are extracted; at most one of fps, downsample and n-frames should be specified; if all None, will use the original frame rate
        downsample: the factor at which videos are downsampled; at most one of fps, downsample and n-frames should be specified; if all None, will not downsample
        n_frames: the number of frames to extract; at most one of fps, downsample and n-frames should be specified; if all None, will use all frames
        sfm_tool: the SfM tool to use
        extract_images: if True, will extract images from videos
        run_sfm: if True, will run SfM
        vis: if True, will visualize the SfM results
    """
    if output_dir is None:
        output_dir = dataset_dir
    if vid_ids is None:
        vid_ids = {f.stem for f in dataset_dir.iterdir() if f.is_file()}
    if vid_exts is not None:
        vid_exts = {"." + re.fullmatch(r"\.?(.*)", vid_ext)[1].lower() for vid_ext in vid_exts}
    if extract_images:
        assert (
            sum((fps is None, downsample is None, n_frames is None)) >= 2
        ), "at most one of fps, downsample and n-frames should be specified"
        vids = defaultdict(set)
        for vid_id in vid_ids:
            for f in dataset_dir.iterdir():
                if not f.stem.startswith(vid_id) or vid_exts and f.suffix.lower() not in vid_exts:
                    continue
                try:
                    assert "nframes" in imageio.get_reader(f).get_meta_data()
                except Exception:
                    continue
                vids[vid_id].add(f)
    else:
        vids = set()
        for vid_id in vid_ids:
            if (output_dir / vid_id).exists():
                vids.add(vid_id)

    if extract_images:
        for vid_id in vids:
            cur_output_dir = output_dir / "images" / vid_id
            reset_dir(cur_output_dir)
            for f in vids[vid_id]:
                vid = imageio.get_reader(f)
                vid_info = vid.get_meta_data()
                vid_fps = vid_info["fps"]
                vid_duration = vid_info["duration"]
                vid_n_frames = round(vid_fps * vid_duration)
                if fps is not None:
                    img_ids = (
                        np.linspace(0, vid_n_frames - 1, num=round(min(fps, vid_fps) * vid_duration))
                        .round()
                        .astype(int)
                    )
                elif downsample is not None:
                    img_ids = np.arange(vid_n_frames, step=max(downsample, 1))
                elif n_frames is not None:
                    img_ids = np.linspace(0, vid_n_frames - 1, num=min(n_frames, vid_n_frames)).round().astype(int)
                else:
                    img_ids = np.arange(vid_n_frames)
                if max_dim is not None:
                    s = max_dim / max(vid_info["size"])
                i = 0
                for j, img in enumerate(tqdm(vid, total=vid_n_frames, leave=False)):
                    if j == img_ids[i]:
                        if max_dim is not None:
                            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                        imageio.v3.imwrite(str(cur_output_dir / f"{i:04d}.jpg"), img, quality=100)
                        i += 1
                        if i == len(img_ids):
                            break

    if run_sfm:
        sfm_cmd = (
            "ns-process-data images",
            f"--data {output_dir}/backgrounds",
            f"--output-dir {output_dir}/sfm",
            "--matching-method sequential",
            f"--sfm-tool {sfm_tool}",
            "--feature-type superpoint_inloc",
            "--num-downscales 0",
        )
        run_command(" ".join(sfm_cmd), verbose=True)

    if vis:
        with (output_dir / "sfm/transforms.json").open() as f:
            transforms = json.load(f)
        poses = np.array([frame["transform_matrix"] for frame in transforms["frames"]], dtype=np.float32)
        visualizer = Visualizer(frame_scale=True, pt_size=3)
        visualizer.add_trajectory(poses, cam_size=0.3, color=(0, 0, 0.8))
        visualizer.add_point_cloud(str(output_dir / "sfm/sparse_pc.ply"))
        visualizer.show()


if __name__ == "__main__":
    tyro.cli(main)
