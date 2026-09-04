import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "control" / "endpoint_middle_cap.py"
SPEC = importlib.util.spec_from_file_location("endpoint_middle_cap", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAP)


def _launch() -> tuple[list[str], dict[str, str]]:
    public_scene = "/home/yptang/public-middle/dev3/100247-Box"
    argv = [
        "splart",
        "--output-dir",
        str(CAP.RUN_ROOT / "model_ckpts"),
        "--experiment-name",
        "100247-Box/baseline",
        "--vis",
        "tensorboard",
        "--data",
        public_scene,
        "--max-num-iterations",
        "25000",
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": "2",
        "PYTORCH_CUDA_ALLOC_CONF": CAP.ALLOC_CONF,
        "SPLART_RUN_ROOT": str(CAP.RUN_ROOT),
        "SPLART_SOURCE_DIR": "/home/yptang/.arbor-worktrees/splart_endpoint_middle_baseline_deadbeef",
        "SPLART_SOURCE_COMMIT": "a" * 40,
        "SPLART_PUBLIC_SCENE_DIR": public_scene,
    }
    return argv, env


def test_cap_validates_fixed_fresh_public_launch() -> None:
    argv, env = _launch()
    assert CAP.validate_launch(argv, env)["source_commit"] == "a" * 40


@pytest.mark.parametrize("flag", ["--load-dir", "--resume", "--seed", "--view"])
def test_cap_rejects_selection_flags(flag: str) -> None:
    argv, env = _launch()
    argv.extend([flag, "forbidden"])
    with pytest.raises(RuntimeError, match="selection"):
        CAP.validate_launch(argv, env)


def test_cap_rejects_evaluator_or_physics_paths() -> None:
    argv, env = _launch()
    env["SEALED_EVALUATOR_PATH"] = "/home/yptang/arbor-reviews/secret.json"
    with pytest.raises(ValueError, match="evaluator-only"):
        CAP.validate_launch(argv, env)
    argv, env = _launch()
    argv.extend(["--pipeline.model.endpoint-physics", "true"])
    with pytest.raises(RuntimeError, match="physics"):
        CAP.validate_launch(argv, env)


@pytest.mark.parametrize("flag", ["--data", "--output-dir", "--experiment-name"])
def test_cap_rejects_duplicate_critical_flags(flag: str) -> None:
    argv, env = _launch()
    argv.extend([flag, "override"])
    with pytest.raises(RuntimeError, match="exactly once"):
        CAP.validate_launch(argv, env)


@pytest.mark.parametrize("flag", ["--load-dir", "--resume", "--seed", "--view"])
def test_cap_rejects_forbidden_assignment_form(flag: str) -> None:
    argv, env = _launch()
    argv.append(f"{flag}=forbidden")
    with pytest.raises(RuntimeError, match="selection"):
        CAP.validate_launch(argv, env)


def test_cap_rejects_experiment_traversal() -> None:
    argv, env = _launch()
    argv[argv.index("--experiment-name") + 1] = "../escape"
    with pytest.raises(RuntimeError, match="experiment"):
        CAP.validate_launch(argv, env)
