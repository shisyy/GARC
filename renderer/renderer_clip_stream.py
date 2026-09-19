"""Hash-bound streaming renderer for the C-CLIP-LD counterfactual cache.

The plugin intentionally has no dependency on SplArt training code.  It reads
only the asset and pose fields carried by the validated render plan, renders a
fixed six-view neutral scene, and returns ephemeral uint8 mini-batches.  No RGB
trajectory is written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import tarfile
import tempfile
from typing import Any, Iterator, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import Tensor


PLAN_SCHEMA = "splart-c-clip-ld-render-plan/v2"
VIEWS = ((0, 0), (90, 0), (180, 0), (270, 0), (45, 35), (225, -35))
IMAGE_SIZE = 224
ROW_KEYS = {
    "asset", "axis_orientation", "episode_id", "joint_selector", "object_id", "split",
    "state0_fraction", "state1_fraction", "state_order",
}
FORBIDDEN_KEYS = {"true_endpoint", "true_endpoints", "target_endpoint", "target_endpoints", "labels", "label"}
STATIC_RGB = (148, 148, 148)
MOBILE_RGB = (230, 118, 34)
BACKGROUND_RGB = (255, 255, 255)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution_sha256(name: str) -> str:
    """Bind the actual installed distribution files, not only its version tag."""

    distribution = metadata.distribution(name)
    if not distribution.files:
        raise RuntimeError(f"{name} distribution has no auditable file manifest")
    manifest = []
    for relative in sorted(distribution.files, key=str):
        path = Path(distribution.locate_file(relative))
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{name} distribution file is missing or symlinked: {relative}")
        manifest.append({"path": str(relative).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return _canonical_sha256(manifest)


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in FORBIDDEN_KEYS or _contains_forbidden_key(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _safe_relative(root: Path, relative: str, *, directory: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("asset path must be a safe relative path")
    unresolved = root / relative
    if unresolved.is_symlink():
        raise ValueError("asset path is missing, symlinked, or has the wrong type")
    candidate = unresolved.resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError("asset path escapes the configured root")
    if candidate.is_symlink() or (not candidate.is_dir() if directory else not candidate.is_file()):
        raise ValueError("asset path is missing, symlinked, or has the wrong type")
    return candidate


@dataclass(frozen=True)
class TriangleMesh:
    vertices: Tensor
    faces: Tensor


@dataclass(frozen=True)
class ArticulatedAsset:
    static: TriangleMesh
    mobile_zero: TriangleMesh
    axis: Tensor
    pivot: Tensor
    lower: float
    upper: float
    frame_center: Tensor
    frame_scale: float
    bundle_sha256: str


def _rpy_matrix(text: str | None) -> Tensor:
    roll, pitch, yaw = (0.0, 0.0, 0.0) if not text else tuple(float(value) for value in text.split())
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = torch.tensor(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    ry = torch.tensor(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rz = torch.tensor(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    return (rz @ ry @ rx).float()


def _origin_transform(node: ET.Element | None) -> tuple[Tensor, Tensor]:
    if node is None:
        return torch.eye(3), torch.zeros(3)
    rotation = _rpy_matrix(node.attrib.get("rpy"))
    xyz = tuple(float(value) for value in node.attrib.get("xyz", "0 0 0").split())
    if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
        raise ValueError("invalid URDF origin")
    return rotation, torch.tensor(xyz, dtype=torch.float32)


def _read_obj_mesh(path: Path) -> TriangleMesh:
    vertices: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                vertex = tuple(float(value) for value in line.split()[1:4])
                if len(vertex) != 3 or not all(math.isfinite(value) for value in vertex):
                    raise ValueError("OBJ contains an invalid vertex")
                vertices.append(vertex)
            elif line.startswith("f "):
                indices = []
                for token in line.split()[1:]:
                    raw = int(token.split("/")[0])
                    index = raw - 1 if raw > 0 else len(vertices) + raw
                    if index < 0 or index >= len(vertices):
                        raise ValueError("OBJ face index is out of range")
                    indices.append(index)
                if len(indices) >= 3:
                    polygons.append(indices)
    faces = [(polygon[0], polygon[index], polygon[index + 1]) for polygon in polygons for index in range(1, len(polygon) - 1)]
    if len(vertices) < 3 or not faces:
        raise ValueError(f"mesh has no usable triangles: {path.name}")
    return TriangleMesh(torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64))


def _box_mesh(size: Sequence[float]) -> TriangleMesh:
    sx, sy, sz = (float(value) / 2.0 for value in size)
    vertices = torch.tensor([(x, y, z) for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    faces = torch.tensor([
        (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
        (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
    ])
    return TriangleMesh(vertices.float(), faces.long())


def _cylinder_mesh(radius: float, length: float, segments: int = 24) -> TriangleMesh:
    vertices = [(0.0, 0.0, -length / 2), (0.0, 0.0, length / 2)]
    for z_value in (-length / 2, length / 2):
        vertices.extend(
            (radius * math.cos(2 * math.pi * index / segments), radius * math.sin(2 * math.pi * index / segments), z_value)
            for index in range(segments)
        )
    faces = []
    for index in range(segments):
        following = (index + 1) % segments
        faces.extend(((0, 2 + index, 2 + following), (1, 2 + segments + following, 2 + segments + index)))
        faces.extend(((2 + index, 2 + segments + index, 2 + segments + following), (2 + index, 2 + segments + following, 2 + following)))
    return TriangleMesh(torch.tensor(vertices).float(), torch.tensor(faces).long())


def _sphere_mesh(radius: float, latitude: int = 12, longitude: int = 24) -> TriangleMesh:
    vertices = [(0.0, 0.0, radius), (0.0, 0.0, -radius)]
    for ring in range(1, latitude):
        phi = math.pi * ring / latitude
        vertices.extend(
            (radius * math.sin(phi) * math.cos(2 * math.pi * index / longitude),
             radius * math.sin(phi) * math.sin(2 * math.pi * index / longitude), radius * math.cos(phi))
            for index in range(longitude)
        )
    faces = []
    first, last = 2, 2 + (latitude - 2) * longitude
    for index in range(longitude):
        faces.append((0, first + index, first + (index + 1) % longitude))
        faces.append((1, last + (index + 1) % longitude, last + index))
    for ring in range(latitude - 2):
        upper, lower = 2 + ring * longitude, 2 + (ring + 1) * longitude
        for index in range(longitude):
            following = (index + 1) % longitude
            faces.extend(((upper + index, lower + index, lower + following), (upper + index, lower + following, upper + following)))
    return TriangleMesh(torch.tensor(vertices).float(), torch.tensor(faces).long())


def _resolve_mesh(root: Path, filename: str) -> Path:
    clean = filename.removeprefix("package://").lstrip("/\\")
    unresolved = root / clean
    direct = unresolved.resolve()
    if (root == direct or root in direct.parents) and unresolved.is_file() and not unresolved.is_symlink():
        return direct
    matches = [path for path in root.rglob(Path(clean).name) if path.is_file() and not path.is_symlink()]
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot uniquely resolve mesh {filename}")
    return matches[0]


def _validate_njc_mesh_references(root: Path, urdf: Path, authorized: set[Path]) -> None:
    """Reject unbound external geometry before any NJC render is produced."""

    robot = ET.parse(urdf).getroot()
    used = []
    for mesh in robot.findall(".//visual/geometry/mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            raise ValueError("NJC visual mesh has no filename")
        resolved = _resolve_mesh(root, filename)
        if resolved not in authorized:
            raise ValueError("NJC URDF references an unauthorized mesh outside the exact two-file bundle")
        used.append(resolved)
    if set(used) != authorized:
        raise ValueError("NJC URDF must use both base_final.obj and lid_final.obj")


def _transform_mesh(mesh: TriangleMesh, rotation: Tensor, translation: Tensor, scale: Tensor | None = None) -> TriangleMesh:
    vertices = mesh.vertices if scale is None else mesh.vertices * scale
    return TriangleMesh(vertices @ rotation.T + translation, mesh.faces)


def _concat_meshes(meshes: Sequence[TriangleMesh]) -> TriangleMesh:
    if not meshes:
        raise ValueError("link set has no visual geometry")
    vertices, faces, offset = [], [], 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    return TriangleMesh(torch.cat(vertices), torch.cat(faces))


def _compose_transform(parent_r: Tensor, parent_t: Tensor, local_r: Tensor, local_t: Tensor) -> tuple[Tensor, Tensor]:
    return parent_r @ local_r, parent_r @ local_t + parent_t


def _link_mesh(link: ET.Element, root: Path) -> TriangleMesh:
    meshes = []
    for visual in link.findall("visual"):
        geometry = visual.find("geometry")
        if geometry is None:
            continue
        if (node := geometry.find("box")) is not None:
            mesh, scale = _box_mesh([float(value) for value in node.attrib["size"].split()]), None
        elif (node := geometry.find("cylinder")) is not None:
            mesh, scale = _cylinder_mesh(float(node.attrib["radius"]), float(node.attrib["length"])), None
        elif (node := geometry.find("sphere")) is not None:
            mesh, scale = _sphere_mesh(float(node.attrib["radius"])), None
        elif (node := geometry.find("mesh")) is not None:
            mesh = _read_obj_mesh(_resolve_mesh(root, node.attrib["filename"]))
            scale = torch.tensor(tuple(float(value) for value in node.attrib.get("scale", "1 1 1").split()))
        else:
            continue
        rotation, translation = _origin_transform(visual.find("origin"))
        meshes.append(_transform_mesh(mesh, rotation, translation, scale))
    return _concat_meshes(meshes)


def _parse_urdf(root: Path, urdf: Path, selector: dict[str, Any], bundle_sha256: str) -> ArticulatedAsset:
    robot = ET.parse(urdf).getroot()
    links = {link.attrib["name"]: link for link in robot.findall("link")}
    joints = robot.findall("joint")
    if not links or not joints:
        raise ValueError("URDF has no links or joints")
    selected: ET.Element | None = None
    if selector == {"kind": "first_bounded_revolute_document_order"}:
        for joint in joints:
            limit = joint.find("limit")
            if joint.attrib.get("type") == "revolute" and limit is not None:
                lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
                if math.isfinite(lower + upper) and upper > lower:
                    selected = joint
                    break
    elif selector == {"kind": "exact_name", "name": "lid_hinge"}:
        selected = next((joint for joint in joints if joint.attrib.get("name") == "lid_hinge"), None)
        if selected is not None and selected.attrib.get("type") != "revolute":
            raise ValueError("lid_hinge is not revolute")
    else:
        raise ValueError("joint selector is not authorized")
    if selected is None:
        raise ValueError("required bounded revolute joint was not found")
    limit = selected.find("limit")
    if limit is None:
        raise ValueError("selected joint has no finite bounds")
    lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
    if not math.isfinite(lower + upper) or upper <= lower:
        raise ValueError("selected joint has invalid bounds")

    child_to_joint: dict[str, ET.Element] = {}
    children: dict[str, list[str]] = {name: [] for name in links}
    for joint in joints:
        parent_node, child_node = joint.find("parent"), joint.find("child")
        if parent_node is None or child_node is None:
            raise ValueError("joint without parent or child")
        parent, child = parent_node.attrib["link"], child_node.attrib["link"]
        if parent not in links or child not in links or child in child_to_joint:
            raise ValueError("invalid URDF kinematic tree")
        child_to_joint[child] = joint
        children[parent].append(child)
    roots = sorted(set(links) - set(child_to_joint))
    if not roots:
        raise ValueError("URDF has no kinematic root")
    world: dict[str, tuple[Tensor, Tensor]] = {name: (torch.eye(3), torch.zeros(3)) for name in roots}
    queue = list(roots)
    while queue:
        parent = queue.pop(0)
        parent_r, parent_t = world[parent]
        for child in children[parent]:
            local_r, local_t = _origin_transform(child_to_joint[child].find("origin"))
            world[child] = _compose_transform(parent_r, parent_t, local_r, local_t)
            queue.append(child)
    if set(world) != set(links):
        raise ValueError("URDF graph is cyclic or disconnected")
    world_meshes = {}
    for name, link in links.items():
        try:
            local = _link_mesh(link, root)
        except ValueError as error:
            if "no visual geometry" in str(error):
                continue
            raise
        world_meshes[name] = _transform_mesh(local, *world[name])

    parent_name = selected.find("parent").attrib["link"]
    child_name = selected.find("child").attrib["link"]
    subtree, pending = set(), [child_name]
    while pending:
        name = pending.pop()
        if name in subtree:
            raise ValueError("cycle in mobile subtree")
        subtree.add(name)
        pending.extend(children[name])
    mobile_names = sorted(subtree & set(world_meshes))
    static_names = sorted(set(world_meshes) - subtree)
    if not mobile_names or not static_names:
        raise ValueError("selected joint does not partition rendered geometry")
    static = _concat_meshes([world_meshes[name] for name in static_names])
    mobile = _concat_meshes([world_meshes[name] for name in mobile_names])
    local_r, local_t = _origin_transform(selected.find("origin"))
    joint_r, pivot = _compose_transform(*world[parent_name], local_r, local_t)
    axis_node = selected.find("axis")
    axis_text = "1 0 0" if axis_node is None else axis_node.attrib.get("xyz", "1 0 0")
    axis = joint_r @ torch.tensor(tuple(float(value) for value in axis_text.split()))
    if not torch.isfinite(axis).all() or float(torch.linalg.vector_norm(axis)) <= 0:
        raise ValueError("selected joint has an invalid axis")
    union = torch.cat((static.vertices, mobile.vertices))
    center = (union.amin(0) + union.amax(0)) * 0.5
    scale = float(torch.linalg.vector_norm(union.amax(0) - union.amin(0)))
    if not math.isfinite(scale) or scale <= 1e-8:
        raise ValueError("asset has degenerate geometry")
    return ArticulatedAsset(static, mobile, axis, pivot, lower, upper, center, scale, bundle_sha256)


def _rotate(mesh: TriangleMesh, axis: Tensor, pivot: Tensor, angle: float) -> TriangleMesh:
    axis = axis / torch.linalg.vector_norm(axis)
    offset = mesh.vertices - pivot
    theta = torch.tensor(angle, dtype=mesh.vertices.dtype)
    cross = torch.linalg.cross(axis.expand_as(offset), offset, dim=-1)
    projection = (offset * axis).sum(-1, keepdim=True) * axis
    vertices = pivot + offset * torch.cos(theta) + cross * torch.sin(theta) + projection * (1 - torch.cos(theta))
    return TriangleMesh(vertices, mesh.faces)


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        root = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if (root != target and root not in target.parents) or member.issym() or member.islnk():
                raise ValueError("unsafe archive member")
        handle.extractall(destination, filter="data")


def _bundle_digest(files: Sequence[tuple[str, Path]]) -> str:
    manifest = [{"path": name, "sha256": _sha256_file(path), "bytes": path.stat().st_size} for name, path in files]
    return _canonical_sha256(manifest)


class StreamingCounterfactualRenderer:
    def __init__(self, asset_root: Path, domain: str, plan: dict[str, Any], *, device: str = "cpu") -> None:
        self.asset_root = asset_root.resolve()
        self.domain = domain
        self.device = torch.device(device)
        if domain not in {"articraft", "njc"} or plan.get("schema") != PLAN_SCHEMA or plan.get("domain") != domain:
            raise ValueError("renderer plan schema/domain mismatch")
        if plan.get("target_labels_used") is not False:
            raise ValueError("renderer requires an explicitly label-independent render plan")
        if _contains_forbidden_key(plan):
            raise ValueError("renderer plan contains target endpoint or label metadata")
        if tuple(tuple(value) for value in plan.get("views", ())) != VIEWS:
            raise ValueError("renderer requires the exact fixed six-view schedule")
        self._canonical_q = plan.get("canonical_q")
        if not isinstance(self._canonical_q, dict) or set(self._canonical_q) != {"side0", "side1"}:
            raise ValueError("renderer plan has no canonical candidate coordinates")
        if not all(isinstance(value, list) and value for value in self._canonical_q.values()):
            raise ValueError("renderer plan has invalid canonical candidate coordinates")
        side_lengths = {len(value) for value in self._canonical_q.values()}
        expected_shape = [2, next(iter(side_lengths)) if len(side_lengths) == 1 else -1, 6, 3, IMAGE_SIZE, IMAGE_SIZE]
        if plan.get("required_render_shape") != expected_shape:
            raise ValueError("renderer requires the hash-bound [2,N,6,3,224,224] render shape")
        self.backend = "pytorch3d-face-zbuffer" if self.device.type == "cuda" else "software-face-zbuffer"
        if self.backend.startswith("pytorch3d"):
            try:
                import pytorch3d
            except ImportError as error:
                raise RuntimeError("CUDA materialization requires the staged PyTorch3D renderer dependency") from error
            if not torch.cuda.is_available():
                raise RuntimeError("formal CUDA renderer requested but CUDA is unavailable")
            pytorch3d_version = getattr(pytorch3d, "__version__", metadata.version("pytorch3d"))
            if pytorch3d_version != "0.7.8":
                raise RuntimeError("formal renderer requires pytorch3d==0.7.8")
            pytorch3d_distribution_sha256: str | None = _distribution_sha256("pytorch3d")
        else:
            pytorch3d_version = None
            pytorch3d_distribution_sha256 = None
        self.runtime = {
            "device_type": self.device.type,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "pytorch3d_version": pytorch3d_version,
            "pytorch3d_distribution_sha256": pytorch3d_distribution_sha256,
        }
        config = {
            "schema": "splart-c-clip-ld-render-config/v1",
            "backend": self.backend,
            "views_azimuth_elevation_degrees": VIEWS,
            "image_size": IMAGE_SIZE,
            "camera": {"projection": "perspective" if self.device.type == "cuda" else "orthographic", "distance": 2.4, "fov_degrees": 35.0},
            "background_rgb": BACKGROUND_RGB,
            "static_rgb": STATIC_RGB,
            "mobile_rgb": MOBILE_RGB,
            "depth_shading": {"far_factor": 0.72, "near_factor": 1.0},
            "normalization": "fixed zero-pose union bbox center and diagonal",
            "runtime": self.runtime,
        }
        self.render_config_sha256 = _canonical_sha256(config)
        self._assets: dict[tuple[str, str], ArticulatedAsset] = {}
        self._audited: set[tuple[str, str]] = set()
        self._views_checked = 0

    def _validate_row(self, row: dict[str, Any]) -> None:
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError("renderer row has missing, extra, or target-only metadata")
        if _contains_forbidden_key(row):
            raise ValueError("renderer row contains true endpoints or labels")
        if not all(isinstance(row[key], str) and row[key] for key in ("object_id", "episode_id")):
            raise ValueError("renderer row has invalid identity")
        if row["axis_orientation"] not in (-1, 1) or row["state_order"] not in ("forward", "reverse"):
            raise ValueError("renderer row has invalid explicit pose metadata")
        if not all(isinstance(row[key], (int, float)) and math.isfinite(float(row[key])) for key in ("state0_fraction", "state1_fraction")):
            raise ValueError("renderer row has invalid state fractions")

    def _load_asset(self, row: dict[str, Any]) -> ArticulatedAsset:
        self._validate_row(row)
        key = (row["object_id"], _canonical_sha256(row["asset"]))
        if key in self._assets:
            return self._assets[key]
        asset = row["asset"]
        if self.domain == "articraft":
            if set(asset) != {"kind", "relative_path"} or asset["kind"] != "articraft_archive":
                raise ValueError("invalid Articraft asset mapping")
            archive = _safe_relative(self.asset_root, asset["relative_path"])
            if not archive.name.endswith(".gz") or Path(archive.name).stem != row["object_id"]:
                raise ValueError("Articraft object_id/archive mapping mismatch")
            bundle = _bundle_digest([(asset["relative_path"], archive)])
            with tempfile.TemporaryDirectory(prefix="splart-clip-render-") as temporary:
                extracted = Path(temporary)
                _safe_extract(archive, extracted)
                urdfs = list(extracted.rglob("model.urdf"))
                if len(urdfs) != 1 or urdfs[0].is_symlink():
                    raise ValueError("Articraft archive must contain exactly one regular model.urdf")
                parsed = _parse_urdf(extracted, urdfs[0], row["joint_selector"], bundle)
        else:
            required = ["object.urdf", "base_final.obj", "lid_final.obj", "qc.json"]
            if set(asset) != {"kind", "relative_path", "required_files"} or asset["kind"] != "njc_directory" or asset["required_files"] != required:
                raise ValueError("invalid NJC asset mapping")
            if asset["relative_path"] != row["object_id"]:
                raise ValueError("NJC object_id/directory mapping mismatch")
            directory = _safe_relative(self.asset_root, asset["relative_path"], directory=True)
            files = [(name, _safe_relative(directory, name)) for name in required]
            bundle = _bundle_digest(files)
            by_name = dict(files)
            _validate_njc_mesh_references(
                directory, by_name["object.urdf"], {by_name["base_final.obj"], by_name["lid_final.obj"]}
            )
            parsed = _parse_urdf(directory, by_name["object.urdf"], row["joint_selector"], bundle)
        self._assets[key] = parsed
        return parsed

    def asset_bundle_sha256(self, row: dict[str, Any]) -> str:
        return self._load_asset(row).bundle_sha256

    def _angles(self, row: dict[str, Any], canonical_q: Sequence[float], asset: ArticulatedAsset) -> list[float]:
        start, end = float(row["state0_fraction"]), float(row["state1_fraction"])
        span = asset.upper - asset.lower
        angle0 = asset.lower + start * span
        angle1 = asset.lower + end * span
        # This is true linear continuation from the two public observed states.
        # It is deliberately never clamped back to the URDF limits.
        return [angle0 + float(q_value) * (angle1 - angle0) for q_value in canonical_q]

    def _render_pytorch3d(self, asset: ArticulatedAsset, angles: Sequence[float]) -> Tensor:
        from pytorch3d.renderer import FoVPerspectiveCameras, MeshRasterizer, RasterizationSettings, look_at_view_transform
        from pytorch3d.structures import Meshes

        vertices_list, faces_list = [], []
        face_count = len(asset.static.faces) + len(asset.mobile_zero.faces)
        face_parts = torch.cat((torch.zeros(len(asset.static.faces), dtype=torch.long), torch.ones(len(asset.mobile_zero.faces), dtype=torch.long))).to(self.device)
        for angle in angles:
            mobile = _rotate(asset.mobile_zero, asset.axis, asset.pivot, angle)
            vertices = torch.cat((asset.static.vertices, mobile.vertices))
            vertices = (vertices - asset.frame_center) / asset.frame_scale
            faces = torch.cat((asset.static.faces, mobile.faces + len(asset.static.vertices)))
            for _ in VIEWS:
                vertices_list.append(vertices.to(self.device))
                faces_list.append(faces.to(self.device))
        meshes = Meshes(verts=vertices_list, faces=faces_list)
        azimuth = torch.tensor([view[0] for _ in angles for view in VIEWS], device=self.device, dtype=torch.float32)
        elevation = torch.tensor([view[1] for _ in angles for view in VIEWS], device=self.device, dtype=torch.float32)
        rotation, translation = look_at_view_transform(dist=2.4, elev=elevation, azim=azimuth, device=self.device)
        cameras = FoVPerspectiveCameras(device=self.device, R=rotation, T=translation, fov=35.0)
        settings = RasterizationSettings(image_size=IMAGE_SIZE, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=False)
        fragments = MeshRasterizer(cameras=cameras, raster_settings=settings)(meshes)
        packed = fragments.pix_to_face[..., 0]
        valid = packed >= 0
        local_face = packed.remainder(face_count).clamp_min(0)
        part = face_parts[local_face]
        depth = fragments.zbuf[..., 0]
        rgb = torch.full((*valid.shape, 3), 255.0, device=self.device)
        colors = torch.tensor((STATIC_RGB, MOBILE_RGB), dtype=torch.float32, device=self.device)
        for image_index in range(len(vertices_list)):
            mask = valid[image_index]
            if not bool(mask.any()):
                raise RuntimeError("counterfactual render produced an empty view")
            values = depth[image_index][mask]
            span = (values.max() - values.min()).clamp_min(1e-8)
            shade = 1.0 - 0.28 * (values - values.min()) / span
            rgb[image_index][mask] = colors[part[image_index][mask]] * shade[:, None]
        rgb = rgb.round().clamp(0, 255).to(torch.uint8).permute(0, 3, 1, 2)
        return rgb.reshape(len(angles), len(VIEWS), 3, IMAGE_SIZE, IMAGE_SIZE).cpu()

    @staticmethod
    def _software_view(static: TriangleMesh, mobile: TriangleMesh, center: Tensor, scale: float, azimuth: float, elevation: float) -> Tensor:
        vertices = torch.cat((static.vertices, mobile.vertices)).numpy().astype(np.float64)
        vertices = (vertices - center.numpy()) / scale
        faces = torch.cat((static.faces, mobile.faces + len(static.vertices))).numpy()
        parts = np.concatenate((np.zeros(len(static.faces), dtype=np.int64), np.ones(len(mobile.faces), dtype=np.int64)))
        azimuth, elevation = math.radians(azimuth), math.radians(elevation)
        eye = np.array((math.cos(elevation) * math.sin(azimuth), math.sin(elevation), math.cos(elevation) * math.cos(azimuth)))
        world_up = np.array((0.0, 1.0, 0.0))
        if abs(float(np.dot(eye, world_up))) > 0.99:
            world_up = np.array((0.0, 0.0, 1.0))
        right = np.cross(world_up, eye)
        right /= np.linalg.norm(right)
        up = np.cross(eye, right)
        up /= np.linalg.norm(up)
        projected = np.stack((vertices @ right, vertices @ up), axis=-1)
        projected = (projected / 0.72 + 0.5) * (IMAGE_SIZE - 1)
        depth_values = -(vertices @ eye)
        zbuffer = np.full((IMAGE_SIZE, IMAGE_SIZE), np.inf, dtype=np.float64)
        part_buffer = np.full((IMAGE_SIZE, IMAGE_SIZE), -1, dtype=np.int8)
        for face_index, face in enumerate(faces):
            points = projected[face]
            x0 = max(0, int(math.floor(points[:, 0].min())))
            x1 = min(IMAGE_SIZE - 1, int(math.ceil(points[:, 0].max())))
            y0 = max(0, int(math.floor(points[:, 1].min())))
            y1 = min(IMAGE_SIZE - 1, int(math.ceil(points[:, 1].max())))
            if x0 > x1 or y0 > y1:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            p0, p1, p2 = points
            denominator = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
            if abs(float(denominator)) < 1e-12:
                continue
            w0 = ((p1[1] - p2[1]) * (xs - p2[0]) + (p2[0] - p1[0]) * (ys - p2[1])) / denominator
            w1 = ((p2[1] - p0[1]) * (xs - p2[0]) + (p0[0] - p2[0]) * (ys - p2[1])) / denominator
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            z = w0 * depth_values[face[0]] + w1 * depth_values[face[1]] + w2 * depth_values[face[2]]
            current = zbuffer[y0:y1 + 1, x0:x1 + 1]
            update = inside & (z < current)
            current[update] = z[update]
            part_buffer[y0:y1 + 1, x0:x1 + 1][update] = parts[face_index]
        valid = part_buffer >= 0
        if not valid.any():
            raise RuntimeError("counterfactual render produced an empty view")
        rgb = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), BACKGROUND_RGB, dtype=np.float64)
        foreground_depth = zbuffer[valid]
        span = max(float(foreground_depth.max() - foreground_depth.min()), 1e-8)
        shade = 1.0 - 0.28 * (foreground_depth - foreground_depth.min()) / span
        colors = np.asarray((STATIC_RGB, MOBILE_RGB), dtype=np.float64)
        rgb[valid] = colors[part_buffer[valid]] * shade[:, None]
        return torch.from_numpy(np.rint(rgb).clip(0, 255).astype(np.uint8)).permute(2, 0, 1)

    def _render_software(self, asset: ArticulatedAsset, angles: Sequence[float]) -> Tensor:
        candidates = []
        for angle in angles:
            mobile = _rotate(asset.mobile_zero, asset.axis, asset.pivot, angle)
            candidates.append(torch.stack([
                self._software_view(asset.static, mobile, asset.frame_center, asset.frame_scale, azimuth, elevation)
                for azimuth, elevation in VIEWS
            ]))
        return torch.stack(candidates)

    def _render(self, asset: ArticulatedAsset, angles: Sequence[float]) -> Tensor:
        result = self._render_pytorch3d(asset, angles) if self.device.type == "cuda" else self._render_software(asset, angles)
        if result.dtype != torch.uint8 or result.shape != (len(angles), len(VIEWS), 3, IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError("renderer violated the uint8 candidate mini-batch contract")
        foreground = (result != 255).any(dim=2).flatten(2).any(-1)
        if not bool(foreground.all()):
            raise RuntimeError("counterfactual render produced an empty view")
        self._views_checked += int(foreground.numel())
        return result

    def audit_receipt(self) -> dict[str, Any]:
        """Return compact evidence for the fail-closed cache receipt."""

        return {
            "schema": "splart-c-clip-ld-render-audit/v1",
            "no_empty_views_checked": self._views_checked > 0,
            "repeat_bit_identical": bool(self._audited),
            "assets_repeat_audited": len(self._audited),
            "views_checked": self._views_checked,
            "render_config_sha256": self.render_config_sha256,
            "runtime": self.runtime,
        }

    def iter_candidate_batches(
        self, *, row: dict[str, Any], side: int, canonical_q: Sequence[float],
        views: Sequence[Sequence[int]], batch_size: int,
    ) -> Iterator[dict[str, Any]]:
        self._validate_row(row)
        if side not in (0, 1) or tuple(tuple(value) for value in views) != VIEWS:
            raise ValueError("renderer received a non-canonical side or view schedule")
        expected_q = self._canonical_q[f"side{side}"]
        if list(canonical_q) != expected_q:
            raise ValueError("renderer received target-adaptive or reordered candidate coordinates")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("renderer batch size must be positive")
        asset = self._load_asset(row)
        angles = self._angles(row, canonical_q, asset)
        audit_key = (row["object_id"], asset.bundle_sha256)
        if audit_key not in self._audited:
            first = self._render(asset, angles[:1])
            repeated = self._render(asset, angles[:1])
            if not torch.equal(first, repeated):
                raise RuntimeError("counterfactual renderer failed the repeat-bit-identical audit")
            self._audited.add(audit_key)
        for start in range(0, len(angles), batch_size):
            yield {"start": start, "images": self._render(asset, angles[start:start + batch_size])}


def create_renderer(asset_root: Path, domain: str, plan: dict[str, Any], *, device: str = "cpu") -> StreamingCounterfactualRenderer:
    """Factory consumed by :mod:`build_clip_limit_cache` after script hashing."""

    return StreamingCounterfactualRenderer(Path(asset_root), domain, plan, device=device)


__all__ = ["ArticulatedAsset", "StreamingCounterfactualRenderer", "TriangleMesh", "create_renderer"]
