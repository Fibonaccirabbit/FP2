"""FP2-Fou: multi-task FP2 on the MO-MuJoCo tasks with leave-one-task-out transfer.

    python run_fp2_fou.py --held_out_env MO-Swimmer-v2 --stage 1   # pre-train on the other five tasks
    python run_fp2_fou.py --held_out_env MO-Swimmer-v2 --stage 2   # mid-train on their top-two fronts
    python run_fp2_fou.py --held_out_env MO-Swimmer-v2 --stage 3   # post-train on the held-out task (1K steps)
    python run_fp2_fou.py --held_out_env MO-Swimmer-v2 --stage 4   # from-scratch baseline on the held-out task
    python run_fp2_fou.py --held_out_env MO-Swimmer-v2 --mode eval --load_model fp2_s0_ft/_best

All tasks are zero-padded to the largest state / action / objective dimensions (27 / 8 / 3). During Stages 1-2
the held-out task is evaluated, which also selects the checkpoint kept for the next stage.
Runs are written to results/fp2_fou/<split>/<held-out dataset>/<run>/ with <run> = <name>_s<seed>[_mid|_ft|_scratch].
"""
import argparse

import torch
from torch.utils.data import DataLoader

from fp2.agent import build_agent, load_checkpoint, prepare_post_training
from fp2.config import (RESULTS_DIR, add_common_args, apply_stage, check_run_dir, create_run_dir, load_training_args,
                        set_seed, split_load_model)
from fp2.data.dataset import build_dataset, return_scale
from fp2.data.text_encoder import PromptEncoder
from fp2.envs import TASKS
from fp2.evaluate import evaluate_checkpoint, make_evaluator
from fp2.train import forever, train

MULTI_TASK_DIMS = (27, 8, 3)  # largest state / action / objective dimensions of TASKS
STAGES = {
    1: {"suffix": "", "load_from": None, "post_training": False, "lr": 2e-4, "n_gradient_steps": 400_000,
        "patience": 0, "eval_interval": 10_000},
    2: {"suffix": "_mid", "load_from": "{name}/_best", "post_training": False, "lr": 2e-5,
        "n_gradient_steps": 400_000, "patience": 10, "eval_interval": 5_000},
    3: {"suffix": "_ft", "load_from": "{name}_mid/_best", "post_training": True, "lr": 2e-5,
        "n_gradient_steps": 1_000, "patience": 0, "eval_interval": 1_000},
    4: {"suffix": "_scratch", "load_from": None, "post_training": False, "lr": 2e-4, "n_gradient_steps": 1_000,
        "patience": 0, "eval_interval": 1_000},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--held_out_env", required=True, choices=TASKS)
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2, 3, 4],
                        help="1: pre-train, 2: mid-train, 3: post-train, 4: scratch. Required for training; in eval "
                             "mode it is read from the run.")
    parser.add_argument("--quality", default="expert", choices=["expert", "amateur"])
    parser.add_argument("--split", default="full", choices=["full", "shattered", "narrow"])
    parser.add_argument("--per_task_batch_size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=32)
    add_common_args(parser)
    parser.set_defaults(lora_rank=8, num_prefs=101, n_elites=10, num_workers=4)
    args = parser.parse_args()
    args.dataset = f"{args.held_out_env}_50000_{args.quality}_uniform"
    if args.mode == "train":
        if args.stage is None:
            parser.error("--stage is required for training")
        apply_stage(args, STAGES)
    elif args.load_model is None:
        parser.error("--load_model is required for evaluation")
    return args


def mix_batches(batches, obs_dim, act_dim, pref_dim, augment):
    """Zero-pad per-task batches to the model dimensions and concatenate them.

    ``fix_mask`` marks the entries that are given rather than generated (the current state and the padded
    dimensions); ``loss_weight`` up-weights the first predicted state. With ``augment`` (Stages 1-2):
      * feature masking (p = 0.4 per sample): random state / action dimensions are zeroed and fixed;
      * dimension permutation (p = 0.1, tasks with fewer state dimensions than the model): the state (and action)
        dimensions are moved to random, sorted positions of the padded vector.
    """
    total = sum(b["obs"].shape[0] for b in batches)
    horizon, dim = batches[0]["obs"].shape[1], obs_dim + act_dim
    out = {"obs": torch.zeros(total, horizon, obs_dim), "act": torch.zeros(total, horizon, act_dim),
           "val": torch.zeros(total, pref_dim), "pref": torch.zeros(total, pref_dim),
           "text": torch.zeros(total, batches[0]["text_embedding"].shape[-1]),
           "fix_mask": torch.zeros(total, horizon, dim), "loss_weight": torch.ones(total, horizon, dim),
           "task_id": torch.zeros(total, dtype=torch.long)}
    obs_out, act_out, fix_mask = out["obs"], out["act"], out["fix_mask"]

    start = 0
    for task, b in enumerate(batches):
        n, _, o = b["obs"].shape
        a = b["act"].shape[2]
        rows = slice(start, start + n)
        obs_out[rows, :, :o], act_out[rows, :, :a] = b["obs"], b["act"]
        out["val"][rows, :b["val"].shape[-1]] = b["val"]
        out["pref"][rows, :b["pref"].shape[-1]] = b["pref"]
        out["text"][rows] = b["text_embedding"]
        out["task_id"][rows] = task
        fix_mask[rows, :, o:obs_dim] = 1.0
        fix_mask[rows, :, obs_dim + a:] = 1.0
        fix_mask[rows, 0, :o] = 1.0

        if augment:
            masked = torch.rand(n) < 0.4
            if masked.any():
                n_obs = torch.randint(1, max(2, o // 2), (1,)).item()
                n_act = torch.randint(1, max(2, a // 3), (1,)).item()
                for j in (start + masked.nonzero(as_tuple=True)[0]).tolist():
                    obs_idx, act_idx = torch.randperm(o)[:n_obs], torch.randperm(a)[:n_act]
                    obs_out[j, :, obs_idx], fix_mask[j, :, obs_idx] = 0.0, 1.0
                    act_out[j, :, act_idx], fix_mask[j, :, obs_dim + act_idx] = 0.0, 1.0
        if augment and o < obs_dim:
            permuted = torch.rand(n) < 0.1
            for j in (start + permuted.nonzero(as_tuple=True)[0]).tolist():
                slots = torch.randperm(obs_dim)[:o].sort()[0]
                obs_out[j] = _scatter(obs_out[j, :, :o], slots, obs_dim)
                fix_mask[j, :, :obs_dim] = 1.0
                fix_mask[j, 1:, slots] = 0.0
                if a < act_dim:
                    slots = torch.randperm(act_dim)[:a].sort()[0]
                    act_out[j] = _scatter(act_out[j, :, :a], slots, act_dim)
                    fix_mask[j, :, obs_dim:] = 1.0
                    fix_mask[j, :, obs_dim + slots] = 0.0
        out["loss_weight"][rows, 1, :o] = 10.0
        start += n
    return out


def _scatter(x, slots, dim):
    """(horizon, n) -> (horizon, dim) with the n features at ``slots`` and zeros elsewhere."""
    out = x.new_zeros(x.shape[0], dim)
    out[:, slots] = x
    return out


def main():
    args = parse_args()
    task_dir = RESULTS_DIR / "fp2_fou" / args.split / args.dataset
    run_dir = task_dir / (args.run if args.mode == "train" else split_load_model(args.load_model)[0])
    check_run_dir(run_dir, args)
    if args.mode == "eval":
        load_training_args(args, run_dir)
    set_seed(args.seed)

    multi_task = args.mode == "train" and args.stage in (1, 2)
    ds_kwargs = dict(horizon=args.horizon, sentence_model_name=args.sentence_model_name,
                     clip_model_name=args.clip_model_name, device=args.device)
    if multi_task:
        names = [f"{t}_50000_{args.quality}_uniform" for t in TASKS if t != args.held_out_env]
        datasets = [build_dataset(d, args.stage, args.split, **ds_kwargs) for d in names]
        target = build_dataset(args.dataset, 1, args.split, **ds_kwargs)  # evaluated on its whole split
    else:
        target = build_dataset(args.dataset, 2, args.split, **ds_kwargs)  # its top-two fronts
        datasets = [target]
    dims = (target.o_dim, target.a_dim, target.pref_dim) if args.stage == 4 else MULTI_TASK_DIMS

    prompt_encoder = PromptEncoder(args.sentence_model_name, args.clip_model_name, args.device)
    in_dim = dims[0] + dims[1]
    agent = build_agent(args, in_dim, dims[2], target.text_dim, [256, 128],
                        torch.zeros((args.horizon, in_dim), device=args.device),  # set per batch / per rollout
                        torch.ones((args.horizon, in_dim), device=args.device), grad_clip_norm=None)
    # Post-training is evaluated on the held-out task's front, subsampled to 100 preferences beyond 500.
    evaluate = make_evaluator(agent, target, prompt_encoder, args, front_cap=(100, 500), model_dims=dims)
    if args.mode == "eval":
        evaluate_checkpoint(agent, evaluate, args, create_run_dir(run_dir, args))
        return

    if args.load_model:
        load_run, load_tag = split_load_model(args.load_model)
        load_checkpoint(agent, task_dir / load_run, load_tag)
    if args.post_training:
        prepare_post_training(agent, args)

    def batches():
        loaders = [forever(DataLoader(ds, batch_size=args.per_task_batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True, drop_last=True))
                   for ds in datasets]
        scale = return_scale(args.horizon)
        while True:
            b = mix_batches([next(it) for it in loaders], *dims, augment=multi_task)
            b = {k: v.to(args.device) for k, v in b.items()}
            yield {"x": torch.cat([b["obs"], b["act"]], -1), "condition": torch.cat([b["pref"], b["val"] / scale], -1),
                   "pref": b["pref"], "text": b["text"], "fix_mask": b["fix_mask"], "loss_weight": b["loss_weight"],
                   "task_id": b["task_id"],
                   # every segment spans the full horizon; the all-True mask keeps the attention kernel unchanged
                   "attention_mask": torch.ones(b["obs"].shape[:2], dtype=torch.bool, device=args.device)}

    train(agent, batches(), evaluate, args, create_run_dir(run_dir, args))


if __name__ == "__main__":
    main()
