"""Target-free triangle-surface contact measurements, not endpoint predictions.

The observed-pair union is a diagnostic allowance, NOT a physical validity
certificate. Sliding across tessellation can create new pairs without a new
constraint. Missing surface intersection does not rule out solid containment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import math
from pathlib import Path

import numpy as np
import torch

from splart.observed_surfaces import validate_surface


@dataclass(frozen=True)
class ContactConfig:
    version: str = "surface-contact/v1"
    observed_samples: int = 65
    initial_contact_capacity: int = 64
    maximum_contact_capacity: int = 65536
    allowance: str = "exact_triangle_pair_union_over_observed_interval"
    endpoint_selection_authorized: bool = False


CONFIG = ContactConfig()


class ContactEnumerationOverflow(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backend_receipt() -> dict:
    import fcl
    distribution = metadata.distribution("python-fcl")
    files = []
    for relative in sorted(distribution.files or (), key=str):
        path = Path(distribution.locate_file(relative))
        if path.is_file() and path.suffix not in (".pyc",):
            files.append({"relative_path": str(relative).replace("\\", "/"), "sha256": file_sha256(path)})
    if not files:
        raise RuntimeError("backend distribution has no auditable files")
    return {"backend": "python-fcl", "version": distribution.version,
            "binding_version": str(getattr(fcl, "__version__", distribution.version)),
            "files": files, "geometry": "original_triangle_BVH_no_hull_no_radius",
            "signed_solid_distance": False}


def rotation_transform(axis: np.ndarray, pivot: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    if angle == 0:
        return np.eye(3), np.zeros(3)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = cosine * np.eye(3) + (1 - cosine) * np.outer(axis, axis) + sine * cross
    return rotation, pivot - rotation @ pivot


def _mesh_arrays(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64))
    faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int32))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("invalid finite mesh vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("invalid triangle indices")
    return vertices, faces


class TrianglePairQuery:
    """Complete discrete-pose contact enumeration with explicit cap rejection."""
    def __init__(self, static_vertices, static_faces, mobile_vertices, mobile_faces,
                 *, maximum_contacts: int = CONFIG.maximum_contact_capacity):
        import fcl
        if maximum_contacts < 1:
            raise ValueError("maximum contact capacity must be positive")
        self.fcl = fcl
        self.maximum_contacts = maximum_contacts
        self.static_arrays = _mesh_arrays(static_vertices, static_faces)
        self.mobile_arrays = _mesh_arrays(mobile_vertices, mobile_faces)
        self.geometries = []
        for vertices, faces in (self.static_arrays, self.mobile_arrays):
            mesh = fcl.BVHModel()
            mesh.beginModel(len(vertices), len(faces))
            mesh.addSubModel(vertices, faces)
            mesh.endModel()
            self.geometries.append(mesh)
        self.static = fcl.CollisionObject(self.geometries[0])
        self.mobile = fcl.CollisionObject(self.geometries[1])

    def query(self, rotation=None, translation=None) -> dict:
        fcl = self.fcl
        rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
        translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("invalid rigid transform")
        self.mobile.setTransform(fcl.Transform(rotation, translation))
        capacity = min(CONFIG.initial_contact_capacity, self.maximum_contacts)
        attempts = 0
        while True:
            attempts += 1
            result = fcl.CollisionResult()
            count = int(fcl.collide(self.static, self.mobile,
                                  fcl.CollisionRequest(num_max_contacts=capacity, enable_contact=True), result))
            if count != len(result.contacts):
                raise RuntimeError("backend contact count disagrees with returned witnesses")
            if count < capacity:
                break
            if capacity >= self.maximum_contacts:
                raise ContactEnumerationOverflow(f"contact enumeration reached capacity {capacity}; completeness unproven")
            capacity = min(capacity * 2, self.maximum_contacts)
        collision = bool(result.is_collision)
        if collision != bool(result.contacts):
            raise RuntimeError("backend collision flag disagrees with witness list")
        pairs = set()
        for contact in result.contacts:
            if contact.o1 is self.geometries[0] and contact.o2 is self.geometries[1]:
                pair = (int(contact.b1), int(contact.b2))
            elif contact.o2 is self.geometries[0] and contact.o1 is self.geometries[1]:
                pair = (int(contact.b2), int(contact.b1))
            else:
                raise RuntimeError("backend witness geometry identity not bound")
            if not (0 <= pair[0] < len(self.static_arrays[1]) and 0 <= pair[1] < len(self.mobile_arrays[1])):
                raise RuntimeError("backend witness does not identify valid mesh triangles")
            pairs.add(pair)
        if collision:
            # Surface intersections imply zero unsigned surface clearance. FCL
            # penetration_depth is not a reliable signed solid-mesh distance.
            clearance, distance_raw = 0.0, None
        else:
            distance_result = fcl.DistanceResult()
            distance_raw = float(fcl.distance(self.static, self.mobile,
                                             fcl.DistanceRequest(enable_nearest_points=False, enable_signed_distance=False),
                                             distance_result))
            if not math.isfinite(distance_raw) or distance_raw < 0:
                raise RuntimeError("inconsistent or nonfinite unsigned mesh distance")
            clearance = distance_raw
        return {"surface_intersection": collision, "surface_clearance": clearance,
                "backend_unsigned_distance": distance_raw, "triangle_pairs": sorted(pairs),
                "contact_count": count, "pair_count": len(pairs), "enumeration_complete": True,
                "enumeration_capacity": capacity, "enumeration_attempts": attempts}


def observed_pair_reference(observed: list[dict]) -> dict:
    if not observed:
        raise ValueError("observed queries required")
    if any(not row["enumeration_complete"] for row in observed):
        raise ContactEnumerationOverflow("observed contacts incomplete")
    sets = [set(tuple(pair) for pair in row["triangle_pairs"]) for row in observed]
    return {"seen_pairs": set.union(*sets), "persistent_pairs": set.intersection(*sets)}


def annotate_pairs(row: dict, reference: dict) -> dict:
    pairs = set(tuple(pair) for pair in row["triangle_pairs"])
    seen, persistent = reference["seen_pairs"], reference["persistent_pairs"]
    novel = pairs - seen
    seen_static = {p[0] for p in seen}
    seen_mobile = {p[1] for p in seen}
    recombined = {p for p in novel if p[0] in seen_static and p[1] in seen_mobile}
    return dict(row, observed_seen_pair_count=len(pairs & seen),
                persistent_observed_pair_count=len(pairs & persistent), novel_pair_count=len(novel),
                novel_pairs=sorted(novel), observed_region_recombination_count=len(recombined),
                boundary_confirmed=False)


def _pack(rows: list[dict]) -> dict:
    return {
        "surface_intersection": torch.tensor([r["surface_intersection"] for r in rows], dtype=torch.bool),
        "surface_clearance": torch.tensor([r["surface_clearance"] for r in rows], dtype=torch.float64),
        "backend_unsigned_distance": [r["backend_unsigned_distance"] for r in rows],
        **{key: torch.tensor([r[key] for r in rows], dtype=torch.int64) for key in
           ("contact_count", "pair_count", "enumeration_capacity", "enumeration_attempts", "observed_seen_pair_count",
            "persistent_observed_pair_count", "novel_pair_count", "observed_region_recombination_count")},
        "triangle_pairs": [torch.tensor(r["triangle_pairs"], dtype=torch.int64).reshape(-1, 2) for r in rows],
        "novel_pairs": [torch.tensor(r["novel_pairs"], dtype=torch.int64).reshape(-1, 2) for r in rows],
        "enumeration_complete": True,
    }


def measure_surface(value: dict) -> dict:
    """Measure fixed 65 observed and 2x129 outward poses without choosing limits."""
    validate_surface(value)
    mobile = value["mobile_vertices"].numpy()
    static, sf, mf = (value[k].numpy() for k in ("static_vertices", "static_faces", "mobile_faces"))
    axis, pivot = value["axis"].numpy(), value["pivot"].numpy()
    increment = float(value["angular_increment"])
    queries = [TrianglePairQuery(static, sf, vertices, mf) for vertices in mobile]
    # Half-interval local anchoring avoids accumulated inverse replay error.
    # At the exactly shared midpoint choose the same observed anchor under swap;
    # hashing only observed vertices does not convey an absolute/rest pose.
    midpoint_anchor = min(range(2), key=lambda side: hashlib.sha256(mobile[side].tobytes()).digest())
    observed = []
    observed_grid = torch.linspace(0, 1, CONFIG.observed_samples, dtype=torch.float64)
    for q in observed_grid.tolist():
        side = 0 if q < 0.5 else (1 if q > 0.5 else midpoint_anchor)
        rotation, translation = rotation_transform(axis, pivot, increment * (q - side))
        observed.append(queries[side].query(rotation, translation))
    reference = observed_pair_reference(observed)
    observed = [annotate_pairs(row, reference) for row in observed]
    sides = []
    for side in range(2):
        rows = []
        for distance in value["coordinates"][side].tolist():
            rotation, translation = rotation_transform(axis, pivot, increment * (-distance if side == 0 else distance))
            rows.append(annotate_pairs(queries[side].query(rotation, translation), reference))
        sides.append(_pack(rows))
    return {"schema": "surface-contact-measurement/v1", "object_id": value["object_id"], "split": value["split"],
            "coordinates": value["coordinates"].clone(), "observed_coordinates": observed_grid,
            "observed": _pack(observed), "sides": sides,
            "observed_seen_pairs": torch.tensor(sorted(reference["seen_pairs"]), dtype=torch.int64).reshape(-1, 2),
            "persistent_all_observed_pairs": torch.tensor(sorted(reference["persistent_pairs"]), dtype=torch.int64).reshape(-1, 2),
            "observed_intersection_count": sum(row["surface_intersection"] for row in observed),
            "observed_anchor_intersection": [observed[0]["surface_intersection"], observed[-1]["surface_intersection"]],
            "observed_overlapping_or_touching": any(row["surface_intersection"] for row in observed),
            "physical_validity": "unresolved_visual_surface_intersections_and_no_solid_containment_test",
            "novelty_caution": "new_triangle_pairs_can_be_tessellation_or_sliding_not_new_constraints",
            "boundary_confirmed": False, "endpoint_selection_authorized": False, "config": asdict(CONFIG)}


def assert_exact_swap(original: dict, swapped: dict) -> None:
    def compare(a, b):
        if isinstance(a, torch.Tensor):
            if not torch.equal(a, b):
                raise AssertionError("surface-contact exact swap mismatch")
        elif isinstance(a, dict):
            if set(a) != set(b):
                raise AssertionError("surface-contact swap key mismatch")
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, list):
            if len(a) != len(b):
                raise AssertionError("surface-contact swap list mismatch")
            for x, y in zip(a, b):
                compare(x, y)
        elif a != b:
            raise AssertionError("surface-contact exact swap scalar mismatch")
    compare(original["sides"], list(reversed(swapped["sides"])))
    for key, rows in original["observed"].items():
        other = swapped["observed"][key]
        if isinstance(rows, torch.Tensor):
            compare(rows, other.flip(0))
        elif isinstance(rows, list):
            compare(rows, list(reversed(other)))
        else:
            compare(rows, other)
    compare(original["observed_seen_pairs"], swapped["observed_seen_pairs"])
    compare(original["persistent_all_observed_pairs"], swapped["persistent_all_observed_pairs"])
