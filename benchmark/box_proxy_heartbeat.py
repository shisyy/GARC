"""Zero-argument, read-only heartbeat for the frozen Box proxy build."""

from __future__ import annotations

import json
import os
from pathlib import Path


PID = 144958
SCENE_ROOT = Path("/home/yptang/arbor-runs/splart-endpoint-extrapolation/dev3-proxy-v1/100247-Box")
LOG = Path("/home/yptang/arbor-logs/splart-endpoint-proxy-box-v1.log")


def main() -> None:
    try:
        os.kill(PID, 0)
        running = True
    except ProcessLookupError:
        running = False
    complete = SCENE_ROOT / "COMPLETE.json"
    counts = {
        modality: len(list((SCENE_ROOT / modality).rglob("*.png")))
        for modality in ("color", "depth", "part-seg")
    }
    tail = LOG.read_text(errors="replace").splitlines()[-8:] if LOG.is_file() else []
    print(
        json.dumps(
            {
                "pid": PID,
                "running": running,
                "complete": complete.is_file(),
                "png_counts": counts,
                "log_tail": tail,
            }
        )
    )


if __name__ == "__main__":
    main()
