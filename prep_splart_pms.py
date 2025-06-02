import json
from pathlib import Path

import numpy as np
import sapien.core as sapien
from imageio.v3 import imwrite
from tqdm import tqdm
from vis_utils.utils import ch_cam_pose_spec, gen_spherical_random_poses, homogenize_transforms, reset_dir, to_np_color

from splart.articulation_params import ArticulationType


class TreeNode:
    def __init__(self, id):
        self.id = id
        self.children = []

    def get_descendants(self):
        descendants = [self.id]
        for child in self.children:
            descendants.extend(child.get_descendants())
        return descendants


class PartnetMobilityProcessor:
    def __init__(self, pm_dir, pms_dir):
        self.pm_dir = pm_dir
        self.pms_dir = pms_dir
        engine = sapien.Engine()
        renderer = sapien.SapienRenderer(offscreen_only=True)
        engine.set_renderer(renderer)

        scene_config = sapien.SceneConfig()
        scene_config.gravity = np.zeros(3)  # Disable gravity
        self.scene = engine.create_scene(scene_config)
        self.scene.set_ambient_light((0.5, 0.5, 0.5))
        self.scene.add_directional_light((0, 1, -1), (0.5, 0.5, 0.5))

        self.loader = self.scene.create_urdf_loader()
        self.color_palette = np.array(((0, 0, 1), (1, 0, 0), (0, 0, 0)))  # static  # mobile  # background
        width, height = 1000, 1000
        self.camera = self.scene.add_camera("cam", width, height, np.pi / 4, 1e-2, 1e1)
        K = self.camera.get_intrinsic_matrix().astype(float)
        self.meta = {"fl_x": K[0, 0], "fl_y": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "w": width, "h": height}

    def process(self, obj_id, joint_id=0, joint_limits=None, seed=0):
        if seed is not None:
            np.random.seed(seed)
        with (self.pm_dir / obj_id / "meta.json").open() as f:
            obj_cls = json.load(f)["model_cat"]
        print(f"Processing {obj_id}-{obj_cls}...")
        output_dir = self.pms_dir / f"{obj_id}-{obj_cls}"
        reset_dir(output_dir)
        urdf_path = self.pm_dir / obj_id / "mobility.urdf"
        asset = self.loader.load(str(urdf_path))
        assert asset, "failed to load URDF."
        self.scene.step()
        joints = asset.get_active_joints()
        links = {}
        for joint in joints:
            p = joint.get_parent_link().id
            c = joint.get_child_link().id
            if p not in links:
                links[p] = TreeNode(p)
            if c not in links:
                links[c] = TreeNode(c)
            links[p].children.append(links[c])
        joint = joints[joint_id]
        mobile_ids = links[joint.get_child_link().id].get_descendants()
        pose = joint.get_global_pose().to_transformation_matrix()
        axis = pose[:3, 0]
        pivot = pose[:3, 3]
        if joint_limits is None:
            joint_limits = joint.get_limits()[0].astype(float)
            print(f"using default joint limits: {joint_limits}")
        meta = self.meta.copy()
        meta["articulation"] = {
            "pivot": pivot.tolist(),
            "axis": axis.tolist(),
            "type": getattr(ArticulationType, joint.type.upper()),
        }
        if joint.type == "revolute":
            meta["articulation"]["angle"] = joint_limits[1] - joint_limits[0]
        else:
            meta["articulation"]["dist"] = joint_limits[1] - joint_limits[0]
        meta["frames"] = []
        n_samples_per_split = {"train": 200, "val": 20, "test": 100}
        for split, n_samples in n_samples_per_split.items():
            n_samples_half = -(-n_samples // 2)
            meta[f"{split}_filenames"] = []
            (output_dir / f"color/{split}").mkdir(parents=True, exist_ok=True)
            (output_dir / f"depth/{split}").mkdir(parents=True, exist_ok=True)
            (output_dir / f"part-seg/{split}").mkdir(parents=True, exist_ok=True)
            (output_dir / f"part-seg-vis/{split}").mkdir(parents=True, exist_ok=True)
            r_lo, r_hi = (2.5, 4) if split == "train" else (2, 5)
            c2ws = homogenize_transforms(
                np.array(
                    gen_spherical_random_poses(
                        r_lo,
                        r_hi,
                        -np.pi / 6,
                        np.pi / 3,
                        -np.pi,
                        np.pi,
                        -np.pi / 6,
                        np.pi / 6,
                        np.eye(4),
                        np.ones(3),
                        n_samples,
                        pose_spec=3,
                    )
                )
            )
            for i, c2w in enumerate(tqdm(c2ws)):
                if split == "test":
                    state = np.random.uniform(-0.1, 1.1)
                    qpos = asset.get_qpos()
                    qpos[joint_id] = joint_limits[0] + state * (joint_limits[1] - joint_limits[0])
                    asset.set_qpos(qpos)
                elif not i % n_samples_half:
                    state = i // n_samples_half
                    qpos = asset.get_qpos()
                    qpos[joint_id] = joint_limits[state]
                    asset.set_qpos(qpos)
                filepath = f"{split}/{i:04d}.png"
                meta["frames"].append(
                    {
                        "file_path": f"color/{filepath}",
                        "transform_matrix": ch_cam_pose_spec(c2w, 3, 2).tolist(),
                        "state": state,
                    }
                )
                meta[f"{split}_filenames"].append(f"color/{filepath}")
                self.camera.set_pose(sapien.Pose(c2w))
                self.scene.update_render()
                self.camera.take_picture()
                rgba = self.camera.get_color_rgba()
                depth = -self.camera.get_position_rgba()[..., 2]
                segmentation = self.camera.get_visual_actor_segmentation()[..., 1]
                rgba[depth == 0, 3] = 0
                rgba = to_np_color(rgba)
                depth = (depth * 1000).astype(np.uint16)
                condlist = [np.logical_not(segmentation), np.isin(segmentation, mobile_ids)]
                condlist.append(~np.logical_or(condlist[0], condlist[1]))
                segmentation = np.select(condlist, (-1, 1, 0))
                segmentation_vis = to_np_color(self.color_palette[segmentation])
                imwrite(output_dir / f"color/{filepath}", rgba)
                imwrite(output_dir / f"depth/{filepath}", depth)
                imwrite(output_dir / f"part-seg/{filepath}", segmentation.astype(np.uint8))
                imwrite(output_dir / f"part-seg-vis/{filepath}", segmentation_vis)
        with (output_dir / "transforms.json").open("w") as f:
            json.dump(meta, f, indent=2)
        self.scene.remove_articulation(asset)


if __name__ == "__main__":
    pmp = PartnetMobilityProcessor(Path("datasets/partnet-mobility"), Path("datasets/splart/splart-pms"))
    pmp.process("2230", joint_id=16, joint_limits=(0, np.pi / 2))  # Chair
    pmp.process("3558")  # Bottle
    pmp.process("5477", joint_limits=(0, np.pi / 2))  # Display
    pmp.process("7054", joint_limits=(0, np.pi * 3 / 4))  # Clock
    pmp.process("11951", joint_limits=(-np.pi / 3, np.pi / 4))  # TrashCan
    pmp.process("12085")  # DishWasher
    pmp.process("27189")  # Table
    pmp.process("100247", joint_limits=(-np.pi / 6, np.pi / 2))  # Box
    pmp.process("100248")  # Suitcase
    pmp.process("100460", joint_limits=(0, np.pi / 2))  # Bucket
    pmp.process("100756", joint_limits=(0, np.pi * 2 / 3))  # Globe
    pmp.process("100794", joint_id=1)  # Globe
    pmp.process("100882")  # Switch
    pmp.process("101542", joint_id=1, joint_limits=(0, np.pi * 2 / 3))  # Dispenser
    pmp.process("101713")  # Pen
    pmp.process("102016")  # USB
    pmp.process("102400")  # Knife
    pmp.process("102812")  # Switch
    pmp.process("103031")  # CoffeeMachine
    pmp.process("103042")  # Window
    pmp.process("103549", joint_id=3)  # Toaster
    pmp.process("103941", joint_id=33)  # Phone
