#!/usr/bin/env python3
"""CPU-only serialization preflight for the node2.2 profile contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from splart.gauge_energy_profile import GaugeEquivariantProfileHead, PROFILE_CHANNELS, swap_profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    features = torch.linspace(0, 1, 2 * 3 * 33 * len(PROFILE_CHANNELS)).reshape(1, 2, 3, 33, -1)
    lower = -torch.linspace(0.02, 2.0, 33).repeat(1, 3, 1)
    upper = 1.0 + torch.linspace(0.02, 2.0, 33).repeat(1, 3, 1)
    scalars = torch.stack((lower, upper), 1)
    head = GaugeEquivariantProfileHead()
    distance = head(features, scalars)
    sf, ss = swap_profiles(features, scalars)
    swapped = head(sf, ss)
    payload = {
        "schema": "gauge-energy-profile-preflight/v1",
        "shape": list(features.shape),
        "channels": list(PROFILE_CHANNELS),
        "nonnegative": bool((distance >= 0).all()),
        "swap_bit_exact": bool(torch.equal(swapped, distance.flip(1))),
        "contains_labels": False,
        "contains_geometry": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
