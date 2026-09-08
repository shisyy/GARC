#!/usr/bin/env python3
"""Source-only ten-object neutral-render/AlexNet feasibility for SMARC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.models import alexnet

from splart.smarc import opaque_object_group, opaque_row_key, swap_invariant_pair


VIEWS = ((0., 0.), (0., 90.), (0., 180.), (0., 270.), (35., 45.), (-35., 225.))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        root = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError("unsafe archive member")
        handle.extractall(destination, filter="data")


def rpy_matrix(text: str | None) -> torch.Tensor:
    roll, pitch, yaw = (0., 0., 0.) if not text else tuple(float(v) for v in text.split())
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rx = torch.tensor([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
    ry = torch.tensor([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
    rz = torch.tensor([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    return (rz @ ry @ rx).float()


def origin_transform(node: ET.Element | None) -> tuple[torch.Tensor, torch.Tensor]:
    if node is None:
        return torch.eye(3), torch.zeros(3)
    return rpy_matrix(node.attrib.get("rpy")), torch.tensor([float(x) for x in node.attrib.get("xyz", "0 0 0").split()])


def read_obj(path: Path) -> torch.Tensor:
    vertices = []
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append(tuple(float(value) for value in line.split()[1:4]))
    if len(vertices) < 2:
        raise ValueError("mesh has fewer than two vertices")
    return torch.tensor(vertices, dtype=torch.float32)


def resolve_mesh(root: Path, filename: str) -> Path:
    clean = filename.removeprefix("package://").lstrip("/\\")
    direct = root / clean
    if direct.is_file():
        return direct
    matches = list(root.rglob(Path(clean).name))
    if len(matches) != 1:
        raise FileNotFoundError("mesh resolution is not unique")
    return matches[0]


def primitive_points(kind: str, node: ET.Element) -> torch.Tensor:
    if kind == "box":
        sx, sy, sz = (float(v) / 2 for v in node.attrib["size"].split())
        return torch.tensor([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    if kind == "cylinder":
        radius, length = float(node.attrib["radius"]), float(node.attrib["length"])
        return torch.tensor([(radius * math.cos(2*math.pi*i/32), radius * math.sin(2*math.pi*i/32), z)
                             for z in (-length/2, 0., length/2) for i in range(32)])
    radius = float(node.attrib["radius"])
    return torch.tensor([(radius*math.cos(a)*math.cos(b), radius*math.cos(a)*math.sin(b), radius*math.sin(a))
                         for a in torch.linspace(-math.pi/2, math.pi/2, 9) for b in torch.linspace(0, 2*math.pi, 32)])


def link_points(link: ET.Element, root: Path) -> torch.Tensor:
    clouds = []
    for visual in link.findall("visual"):
        geometry = visual.find("geometry")
        if geometry is None:
            continue
        cloud = None
        for kind in ("box", "cylinder", "sphere"):
            node = geometry.find(kind)
            if node is not None:
                cloud = primitive_points(kind, node)
                break
        mesh = geometry.find("mesh")
        if cloud is None and mesh is not None:
            cloud = read_obj(resolve_mesh(root, mesh.attrib["filename"]))
            cloud *= torch.tensor([float(v) for v in mesh.attrib.get("scale", "1 1 1").split()])
        if cloud is None:
            continue
        rotation, translation = origin_transform(visual.find("origin"))
        clouds.append(cloud @ rotation.T + translation)
    if not clouds:
        raise ValueError("link has no visual geometry")
    return torch.cat(clouds)


def parse_articraft(root: Path) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]]:
    urdfs = list(root.rglob("model.urdf"))
    if len(urdfs) != 1:
        raise ValueError("archive must contain exactly one model.urdf")
    robot = ET.parse(urdfs[0]).getroot()
    links = {link.attrib["name"]: link for link in robot.findall("link")}
    samples = []
    for joint in (x for x in robot.findall("joint") if x.attrib.get("type") == "revolute"):
        try:
            limit = joint.find("limit")
            lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
            if not math.isfinite(lower + upper) or upper <= lower:
                continue
            static = link_points(links[joint.find("parent").attrib["link"]], urdfs[0].parent)
            child = link_points(links[joint.find("child").attrib["link"]], urdfs[0].parent)
            joint_rotation, pivot = origin_transform(joint.find("origin"))
            axis_node = joint.find("axis")
            axis = joint_rotation @ torch.tensor([float(v) for v in ("1 0 0" if axis_node is None else axis_node.attrib.get("xyz", "1 0 0")).split()])
            samples.append((static, child @ joint_rotation.T + pivot, axis / axis.norm(), pivot, lower, upper))
        except (AttributeError, KeyError, FileNotFoundError, ValueError):
            continue
    if not samples:
        raise ValueError("no usable finite-limit revolute joint")
    return samples


def rotate(points: torch.Tensor, axis: torch.Tensor, pivot: torch.Tensor, angle: float) -> torch.Tensor:
    centered = points - pivot
    a = torch.tensor(angle, dtype=points.dtype)
    return (centered * torch.cos(a) + torch.cross(axis.expand_as(centered), centered, dim=-1) * torch.sin(a)
            + axis * (centered @ axis)[..., None] * (1 - torch.cos(a))) + pivot


def convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    points = sorted(set(points))
    if len(points) <= 2:
        return points
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def view_rotation(elevation: float, azimuth: float) -> torch.Tensor:
    e, a = math.radians(elevation), math.radians(azimuth)
    ry = torch.tensor([[math.cos(a), 0., math.sin(a)], [0., 1., 0.], [-math.sin(a), 0., math.cos(a)]])
    rx = torch.tensor([[1., 0., 0.], [0., math.cos(e), -math.sin(e)], [0., math.sin(e), math.cos(e)]])
    return rx @ ry


def render_pair(static: torch.Tensor, mobile_states: list[torch.Tensor], size: int = 224) -> torch.Tensor:
    combined = torch.cat([static, *mobile_states])
    center = (combined.amin(0) + combined.amax(0)) * .5
    scale = (combined - center).abs().max().clamp_min(1e-6)
    images = []
    for mobile in mobile_states:
        state_views = []
        for elevation, azimuth in VIEWS:
            rotation = view_rotation(elevation, azimuth)
            projected = [((cloud-center)/scale @ rotation.T)[:, :2] for cloud in (static, mobile)]
            image = Image.new("RGB", (size, size), (13, 13, 13))
            draw = ImageDraw.Draw(image)
            for cloud, color in zip(projected, ((116, 116, 116), (224, 224, 224))):
                pixels = [tuple(int(round((float(v)+1)*.45*(size-1)+.05*(size-1))) for v in point) for point in cloud]
                hull = convex_hull(pixels)
                if len(hull) >= 3:
                    draw.polygon(hull, fill=color)
                else:
                    for x, y in hull:
                        draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
            state_views.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
        images.append(torch.stack(state_views))
    return torch.stack(images)  # [state,view,3,H,W], uint8


def alexnet_encoder(checkpoint: Path, device: torch.device):
    if sha256_file(checkpoint) != "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02":
        raise ValueError("AlexNet checkpoint hash mismatch")
    model = alexnet(weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten(), *list(model.classifier.children())[:-1]).eval().to(device)


def encode_views(model, images: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, float]:
    batch = images.float().to(device) / 255.
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    state_features, repeat_error = [], 0.
    with torch.no_grad():
        for state in range(2):
            normalized = (batch[state] - mean) / std
            first = torch.nn.functional.normalize(model(normalized), dim=-1)
            second = torch.nn.functional.normalize(model(normalized), dim=-1)
            repeat_error = max(repeat_error, float((first-second).abs().max()))
            state_features.append(torch.nn.functional.normalize(first.mean(0), dim=0).cpu())
    return swap_invariant_pair(*state_features), repeat_error


def smoke_retrieval(rows: list[dict], labels: dict[str, float]) -> dict:
    predictions = {"semantic": [], "mechanical": [], "global": []}
    targets = []
    for held in rows:
        train = [row for row in rows if row["object_group_id"] != held["object_group_id"]]
        if not train:
            raise ValueError("object-disjoint retrieval has empty train set")
        target = labels[held["joint_id"]]
        targets.append(target)
        sem = torch.stack([row["semantic"] for row in train])
        mech = torch.stack([row["mechanical"] for row in train])
        sem_distance = 1 - torch.nn.functional.cosine_similarity(held["semantic"][None], sem)
        scale = mech.std(0, unbiased=False).clamp_min(1e-6)
        mech_distance = (((mech-held["mechanical"])/scale).square().mean(-1)).sqrt()
        predictions["semantic"].append(labels[train[int(sem_distance.argmin())]["joint_id"]])
        predictions["mechanical"].append(labels[train[int(mech_distance.argmin())]["joint_id"]])
        predictions["global"].append(float(torch.tensor([labels[row["joint_id"]] for row in train]).median()))
    target_tensor = torch.tensor(targets)
    return {name: {"joint_macro_mare": float(((torch.tensor(value)-target_tensor).abs()/target_tensor).mean())}
            for name, value in predictions.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--pilc-data", type=Path, required=True)
    parser.add_argument("--alexnet-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objects", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = json.loads(args.selection.read_text())["selection"]["endpoint_pretrain"]
    source_inputs = torch.load(args.pilc_data / "inputs.pt", map_location="cpu", weights_only=False)
    source_labels = torch.load(args.pilc_data / "labels.pt", map_location="cpu", weights_only=False)
    inputs_by_key = {row["key"]: row for row in source_inputs}
    labels_by_key = {row["key"]: row for row in source_labels}
    device = torch.device(args.device)
    encoder = alexnet_encoder(args.alexnet_checkpoint, device)
    rows, labels, render_bank, accepted_groups, rejections, attempted_assets = [], {}, {}, [], [], []
    max_repeat_error = 0.
    for archive_name in selection:
        if len(accepted_groups) >= args.objects:
            break
        group = opaque_object_group(archive_name)
        archive_sha256 = sha256_file(args.archive_root / archive_name)
        attempted_assets.append({"object_group_id": group, "archive_sha256": archive_sha256})
        try:
            with tempfile.TemporaryDirectory(prefix="smarc-source-") as temporary:
                extracted = Path(temporary)
                safe_extract(args.archive_root / archive_name, extracted)
                joints = parse_articraft(extracted)
                object_rows = []
                for joint_index, (static, mobile, axis, pivot, lower, upper) in enumerate(joints):
                    key = opaque_row_key(archive_name, joint_index, .25, .75, 1, "forward")
                    if key not in inputs_by_key or key not in labels_by_key:
                        raise ValueError("replayed node4 join key missing")
                    source = inputs_by_key[key]
                    if source["split"] != "endpoint_pretrain":
                        raise ValueError("validation object leaked into smoke index")
                    mechanical = torch.cat((swap_invariant_pair(source["state0_features"], source["state1_features"]),
                                            torch.tensor([abs(source["observed_displacement"])])))
                    physical_range = float(labels_by_key[key]["physical_range"])
                    if abs(physical_range - (upper-lower)) > 1e-5:
                        raise ValueError("parsed and sidecar physical range disagree")
                    if not abs(float(source["observed_displacement"])) <= physical_range <= 2 * math.pi + 1e-6:
                        raise ValueError("revolute range outside [|delta|,2pi]")
                    mobile_states = [rotate(mobile, axis, pivot, lower + fraction*(upper-lower)) for fraction in (.25, .75)]
                    images = render_pair(static, mobile_states)
                    semantic, repeat_error = encode_views(encoder, images, device)
                    max_repeat_error = max(max_repeat_error, repeat_error)
                    joint_id = hashlib.sha256(("splart-smarc-joint-v1:"+key).encode()).hexdigest()
                    object_rows.append({"joint_id": joint_id, "object_group_id": group,
                                        "mechanical": mechanical.cpu(), "semantic": semantic.cpu(),
                                        "observed_displacement": abs(float(source["observed_displacement"]))})
                    labels[joint_id] = physical_range
                    render_bank[joint_id] = images
            rows.extend(object_rows)
            accepted_groups.append(group)
        except Exception as error:
            rejections.append({"object_group_id": group, "archive_sha256": archive_sha256,
                               "error_type": type(error).__name__, "reason": str(error)})
    if len(accepted_groups) != args.objects or not rows:
        raise RuntimeError(f"only {len(accepted_groups)} smoke objects accepted")
    if len(set(accepted_groups)) != args.objects or len({row['joint_id'] for row in rows}) != len(rows):
        raise RuntimeError("opaque ID collision")
    if max_repeat_error != 0 or not all(torch.isfinite(row["semantic"]).all() for row in rows):
        raise RuntimeError("encoder determinism/finite audit failed")
    mechanical_stack = torch.stack([row["mechanical"] for row in rows])
    varying = mechanical_stack.std(0, unbiased=False) > 1e-8
    if not varying.any():
        raise RuntimeError("all mechanical columns are constant")
    removed_constant_columns = int((~varying).sum())
    for row in rows:
        row["mechanical"] = row["mechanical"][varying]
    args.output.mkdir(parents=True)
    torch.save(rows, args.output / "inputs.pt")
    torch.save(labels, args.output / "labels.pt")
    torch.save(render_bank, args.output / "neutral_render_bank.pt")
    metrics = smoke_retrieval(rows, labels)
    manifest = {"schema": "splart-smarc-alexnet-smoke/v1", "objects": len(accepted_groups),
                "joints": len(rows), "input_fields": sorted(rows[0]), "label_fields": ["physical_range"],
                "encoder": "AlexNet-ImageNet-smoke-only", "encoder_sha256": sha256_file(args.alexnet_checkpoint),
                "embedding_dim": int(rows[0]["semantic"].numel()), "mechanical_dim": int(rows[0]["mechanical"].numel()),
                "removed_constant_mechanical_columns": removed_constant_columns,
                "joint_type": "revolute",
                "views_per_state": len(VIEWS), "repeat_max_error": max_repeat_error,
                "object_disjoint_smoke": metrics, "rejections": rejections,
                "attempted_assets": attempted_assets,
                "accepted_assets": [row for row in attempted_assets if row["object_group_id"] in accepted_groups],
                "source_hashes": {"selection": sha256_file(args.selection),
                                  "inputs.pt": sha256_file(args.pilc_data / "inputs.pt"),
                                  "labels.pt": sha256_file(args.pilc_data / "labels.pt")},
                "box_labels_read": [], "protected_splits_read": []}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
