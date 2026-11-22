"""FP2 on one D4MORL task (2 objectives, or 3 for MO-Hopper-v3): three-stage training and evaluation.

    D=MO-Hopper-v2_50000_expert_uniform
    python scripts/run_fp2.py --dataset $D --split shattered --stage 1    # FP2-Pre  -> run fp2_s0
    python scripts/run_fp2.py --dataset $D --split shattered --stage 2    # FP2-Mid  -> run fp2_s0_mid (from fp2_s0/_best)
    python scripts/run_fp2.py --dataset $D --split shattered --stage 3    # FP2-Post -> run fp2_s0_ft  (from fp2_s0_mid/_best)
    python scripts/run_fp2.py --dataset $D --split shattered --mode eval --load_model fp2_s0_mid/_best

Runs are written to results/fp2/<split>/<dataset>/<run>/.
"""
import argparse

import torch
from torch.utils.data import DataLoader

from fp2.agent import build_agent, load_checkpoint, prepare_post_training
from fp2.config import (RESULTS_DIR, add_common_args, apply_stage, check_run_dir, create_run_dir, load_training_args,
                        set_seed, split_load_model)
from fp2.data.dataset import build_dataset, return_scale
from fp2.data.text_encoder import PromptEncoder
from fp2.evaluate import evaluate_checkpoint, make_evaluator
from fp2.train import forever, train

STAGES = {
    1: {"suffix": "", "load_from": None, "post_training": False, "n_gradient_steps": 400_000,
        "eval_interval": 5_000},
    2: {"suffix": "_mid", "load_from": "{name}/_best", "post_training": False, "n_gradient_steps": 400_000,
        "eval_interval": 5_000},
    3: {"suffix": "_ft", "load_from": "{name}_mid/_best", "post_training": True, "n_gradient_steps": 1_000,
        "eval_interval": 1_000},
}
# Settings that differ between the 2-objective tasks and MO-Hopper-v3 (3 objectives).
OBJECTIVES = {
    2: {"lr": (2e-4, 2e-5, 2e-5), "patience": (0, 10, 0), "grad_clip_norm": 1.0, "eps": 1e-3,
        "cond_hidden_dims": [256, 128], "n_elites": 10, "num_prefs": 101, "lora_rank": 64},
    3: {"lr": (2e-4, 2e-4, 2e-4), "patience": (0, 0, 0), "grad_clip_norm": None, "eps": 3e-3,
        "cond_hidden_dims": [128], "n_elites": 4, "num_prefs": 100, "lora_rank": 8},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="e.g. MO-Hopper-v2_50000_expert_uniform")
    parser.add_argument("--split", required=True, choices=["full", "shattered", "narrow"],
                        help="Held-out preferences: none / three interior regions / both extremes.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None,
                        help="1: FP2-Pre, 2: FP2-Mid, 3: FP2-Post. Required for training; in eval mode it is read "
                             "from the run (Stage-3 runs are evaluated on the held-out front).")
    parser.add_argument("--batch_size", type=int, default=64)
    add_common_args(parser)
    args = parser.parse_args()

    args.env_name = args.dataset.split("_")[0]
    args.n_obj = 3 if args.env_name == "MO-Hopper-v3" else 2
    settings = OBJECTIVES[args.n_obj]
    for key in ("n_elites", "num_prefs", "lora_rank"):
        if getattr(args, key) is None:
            setattr(args, key, settings[key])
    args.horizon = 16 if args.env_name.startswith(("MO-Hopper", "MO-Walker2d")) else 4
    if args.mode == "train":
        if args.stage is None:
            parser.error("--stage is required for training")
        stages = {s: dict(cfg, lr=settings["lr"][s - 1], patience=settings["patience"][s - 1])
                  for s, cfg in STAGES.items()}
        apply_stage(args, stages)
    elif args.load_model is None:
        parser.error("--load_model is required for evaluation")
    return args


def main():
    args = parse_args()
    settings = OBJECTIVES[args.n_obj]
    root = RESULTS_DIR / "fp2" / args.split / args.dataset
    load_run, load_tag = split_load_model(args.load_model) if args.load_model else (None, None)
    run_dir = root / (args.run if args.mode == "train" else load_run)
    if args.mode == "train" and load_run == args.run:
        raise ValueError("--load_model must point to another run.")
    check_run_dir(run_dir, args)
    if args.mode == "eval":
        load_training_args(args, run_dir)
    if args.split == "full" and args.post_training:
        raise ValueError("--split full has no held-out preferences, so there is no post-training (Stage 3).")
    set_seed(args.seed)

    stage = args.stage if args.mode == "train" else (3 if args.post_training else 2)
    dataset = build_dataset(args.dataset, stage, args.split, horizon=args.horizon, eps=settings["eps"],
                            sentence_model_name=args.sentence_model_name, clip_model_name=args.clip_model_name,
                            device=args.device)
    prompt_encoder = PromptEncoder(args.sentence_model_name, args.clip_model_name, args.device)
    in_dim = dataset.o_dim + dataset.a_dim
    fix_mask = torch.zeros((args.horizon, in_dim), device=args.device)
    fix_mask[0, :dataset.o_dim] = 1.     # the current state is given
    loss_weight = torch.ones((args.horizon, in_dim), device=args.device)
    loss_weight[1, :dataset.o_dim] = 10.  # emphasize the first predicted state
    agent = build_agent(args, in_dim, dataset.pref_dim, dataset.text_dim, settings["cond_hidden_dims"], fix_mask,
                        loss_weight, settings["grad_clip_norm"])
    # Post-training is evaluated on (at most 100 preferences of) the held-out front.
    evaluate = make_evaluator(agent, dataset, prompt_encoder, args, front_cap=(100, 100))
    if args.mode == "eval":
        evaluate_checkpoint(agent, evaluate, args, create_run_dir(run_dir, args))
        return

    if args.load_model:
        load_checkpoint(agent, root / load_run, load_tag)
    if args.post_training:
        prepare_post_training(agent, args)

    def batches():
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)
        scale = return_scale(args.horizon)
        for b in forever(loader):
            pref = b["pref"].to(args.device)
            val = b["val"].to(args.device) / scale
            yield {"x": torch.cat([b["obs"].to(args.device), b["act"].to(args.device)], -1),
                   "condition": torch.cat([pref, val], -1), "pref": pref,
                   "text": b["text_embedding"].to(args.device)}

    train(agent, batches(), evaluate, args, create_run_dir(run_dir, args))


if __name__ == "__main__":
    main()
