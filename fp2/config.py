"""Command-line arguments, stage settings and run directories shared by run_fp2.py and run_fp2_fou.py."""
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
# Training settings that define the model of a run; evaluation reads them from the run's args.json.
MODEL_ARGS = ("stage", "backbone", "emb_dim", "d_model", "n_heads", "depth", "reg_mode", "lora_rank", "lora_alpha",
              "horizon", "sentence_model_name", "clip_model_name")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("run")
    g.add_argument("--mode", default="train", choices=["train", "eval"])
    g.add_argument("--name", default="fp2",
                   help="Run name prefix: a stage writes to <name>_s<seed> plus a stage suffix (_mid, _ft, ...).")
    g.add_argument("--load_model", default=None,
                   help="Checkpoint <run>/<tag> to start from (training) or to evaluate, e.g. fp2_s0_mid/_best.")
    g.add_argument("--overwrite", action="store_true", help="Replace an existing run directory.")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--lr", type=float, default=None, help="Default: stage setting (with LoRA, --lora_lr is used).")
    g.add_argument("--n_gradient_steps", type=int, default=None, help="Default: stage setting.")
    g.add_argument("--patience", type=int, default=None,
                   help="Stop after this many evaluations without HV improvement (0: never). Default: stage setting.")
    g.add_argument("--eval_interval", type=int, default=None,
                   help="Default: stage setting. The last step is always evaluated.")
    g.add_argument("--save_interval", type=int, default=100_000, help="Steps between periodic checkpoints.")
    g.add_argument("--log_interval", type=int, default=1000)
    g.add_argument("--num_workers", type=int, default=8, help="Data-loading workers (per task in FP2-Fou).")

    g = parser.add_argument_group("model")
    g.add_argument("--backbone", default="rf", choices=["rf", "ddim"], help="Rectified flow, or DDIM (ablation).")
    g.add_argument("--emb_dim", type=int, default=128)
    g.add_argument("--d_model", type=int, default=384)
    g.add_argument("--n_heads", type=int, default=6)
    g.add_argument("--depth", type=int, default=4)
    g.add_argument("--contrastive_weight", type=float, default=0.01, help="Triplet weight (0: w/o Triplet).")
    g.add_argument("--contrastive_loss_type", default="triplet", choices=["triplet", "infonce"])
    g.add_argument("--reg_mode", default="l2", choices=["l2", "lora", "none"],
                   help="Post-training: l2 penalty towards FP2-Mid, LoRA adapters, or plain fine-tuning.")
    g.add_argument("--lora_rank", type=int, default=None, help="Default: 64 (2 objectives) or 8 (3 objectives, FP2-Fou).")
    g.add_argument("--lora_alpha", type=float, default=1.0)
    g.add_argument("--lora_lr", type=float, default=1e-4)
    g.add_argument("--sentence_model_name", default="all-MiniLM-L6-v2")
    g.add_argument("--clip_model_name", default="ViT-B/32")

    g = parser.add_argument_group("evaluation")
    g.add_argument("--eval_prefs", default="auto", choices=["auto", "grid", "front"],
                   help="grid: uniform preferences; front: preferences of the dataset's Pareto front; "
                        "auto: front in post-training, grid otherwise.")
    g.add_argument("--num_prefs", type=int, default=None,
                   help="Size of the preference grid. Default: 101 (100 for MO-Hopper-v3 in run_fp2.py).")
    g.add_argument("--max_prefs", type=int, default=500,
                   help="Larger Pareto fronts are evaluated on this many evenly spaced preferences (--eval_prefs front).")
    g.add_argument("--eval_name", default="test", help="Output name in --mode eval (files eval_<name>.*).")
    g.add_argument("--w_cfg", type=float, default=1.0, help="CFG weight of the numerical preference.")
    g.add_argument("--w_cg", type=float, default=0.01, help="CG weight of the text (0: w/o Text).")
    g.add_argument("--val_scale", type=float, default=1.2, help="Return condition at test time: flip(w) * val_scale.")
    g.add_argument("--temperature", type=float, default=0.5, help="Std of the initial noise.")
    g.add_argument("--sample_steps", type=int, default=10)
    g.add_argument("--n_elites", type=int, default=None,
                   help="Plans sampled per step; the one with the highest text log-likelihood is executed. "
                        "Default: 10 (4 for MO-Hopper-v3 in run_fp2.py).")
    g.add_argument("--num_envs", type=int, default=20, help="Parallel environments.")
    g.add_argument("--num_episodes", type=int, default=1, help="Episodes per preference.")
    return parser


def apply_stage(args, stages: Dict[int, dict]):
    """Run name, starting checkpoint and optimization settings of ``args.stage``; explicit arguments win."""
    cfg = stages[args.stage]
    name = f"{args.name}_s{args.seed}"
    args.run = name + cfg["suffix"]
    if args.load_model is None and cfg["load_from"]:
        args.load_model = cfg["load_from"].format(name=name)
    args.post_training = cfg["post_training"]
    for arg in ("lr", "n_gradient_steps", "patience", "eval_interval"):
        if getattr(args, arg) is None:
            setattr(args, arg, cfg[arg])
    print(f"=== Stage {args.stage}: run={args.run} load_model={args.load_model} lr={args.lr} "
          f"steps={args.n_gradient_steps} patience={args.patience} eval_interval={args.eval_interval} ===")


def load_training_args(args, run_dir: Path):
    """In eval mode, take the model settings (MODEL_ARGS) of the evaluated run from its args.json."""
    path = run_dir / "args.json"
    if path.exists():
        saved = json.loads(path.read_text())
        for key in MODEL_ARGS:
            if key in saved and saved[key] is not None and not (key == "stage" and args.stage is not None):
                setattr(args, key, saved[key])
    args.post_training = args.stage == 3


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def split_load_model(load_model: str) -> Tuple[str, str]:
    """'fp2_s0_mid/_best' -> ('fp2_s0_mid', '_best')."""
    run, _, tag = load_model.rpartition("/")
    if not run:
        raise ValueError(f"--load_model must be <run>/<tag>, got '{load_model}'")
    return run, tag


def check_run_dir(path: Path, args):
    """Fail early: training refuses to reuse a run unless --overwrite; evaluation needs the checkpoint."""
    if args.mode == "train" and path.exists() and not args.overwrite:
        raise FileExistsError(f"{path} exists (use --overwrite to replace it).")
    if args.mode == "eval":
        tag = split_load_model(args.load_model)[1]
        if not (path / f"diffusion_ckpt{tag}.pt").exists():
            raise FileNotFoundError(f"No checkpoint {path / f'diffusion_ckpt{tag}.pt'}")


def create_run_dir(path: Path, args) -> Path:
    if args.mode == "train" and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / ("args.json" if args.mode == "train" else f"args_{args.eval_name}.json"), "w") as f:
        json.dump(vars(args), f, indent=4, default=str)
    return path
