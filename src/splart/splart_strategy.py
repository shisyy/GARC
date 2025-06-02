from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from gsplat.strategy import DefaultStrategy


@dataclass
class SplartStrategy(DefaultStrategy):
    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info_lst: List[Dict[str, Any]],
        packed: bool = False,
    ):
        for info in info_lst:
            for key in ("width", "height", "n_cameras", "radii", "gaussian_ids", self.key_for_gradient):
                assert key in info, f"{key} is required but missing."

            # normalize grads to [-1, 1] screen space
            if self.absgrad:
                grads = info[self.key_for_gradient].absgrad.clone()
            else:
                grads = info[self.key_for_gradient].grad.clone()
            grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
            grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

            # initialize state on the first run
            n_gaussian = len(list(params.values())[0])

            if state["grad2d"] is None:
                state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
            if state["count"] is None:
                state["count"] = torch.zeros(n_gaussian, device=grads.device)
            if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
                assert "radii" in info, "radii is required but missing."
                state["radii"] = torch.zeros(n_gaussian, device=grads.device)

            # update the running state
            if packed:
                # grads is [nnz, 2]
                gs_ids = info["use_ids"][info["gaussian_ids"]]  # [nnz]
                radii = info["radii"].max(dim=-1).values  # [nnz]
            else:
                # grads is [C, N, 2]
                sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
                gs_ids = info["use_ids"][torch.where(sel)[1]]  # [nnz]
                grads = grads[sel]  # [nnz, 2]
                radii = info["radii"][sel].max(dim=-1).values  # [nnz]

            state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
            if self.refine_scale2d_stop_iter > 0:
                # Should be ideally using scatter max
                state["radii"][gs_ids] = torch.maximum(
                    state["radii"][gs_ids],
                    # normalize radii to [0, 1] screen space
                    radii / float(max(info["width"], info["height"])),
                )
