"""Training loop with periodic evaluation and best-HV checkpointing."""
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from fp2.agent import save_checkpoint
from fp2.evaluate import format_metrics, save_eval


def forever(loader) -> Iterator:
    """Cycle through a DataLoader indefinitely."""
    if len(loader) == 0:
        raise ValueError(f"The dataset has {len(loader.dataset)} segments, fewer than one batch.")
    while True:
        yield from loader


def train(agent, batches: Iterator[dict], evaluate: Callable, args, run_dir: Path):
    """Training loop with periodic evaluation (and at the last step); keeps the checkpoint with the best HV
    (_best) and the final one (_last).

    ``batches`` yields dicts with x, condition, pref, text and optional fix_mask / loss_weight / update kwargs;
    ``evaluate()`` returns (dataset, eval_rewards, eval_prefs, metrics).
    """
    sums, best_hv, bad_evals = np.zeros(3), -np.inf, 0
    for step in range(args.n_gradient_steps):
        batch = next(batches)
        if "fix_mask" in batch:
            agent.fix_mask, agent.loss_weight = batch.pop("fix_mask"), batch.pop("loss_weight")
        x, text = batch.pop("x"), batch.pop("text")
        out = agent.update(x, **batch)
        if not np.isfinite(out["loss"]):
            raise FloatingPointError(f"Non-finite loss {out['loss']} at step {step + 1}")
        sums += (out["loss"], agent.update_classifier(x, text)["loss"], out["grad_norm"])
        if (step + 1) % args.log_interval == 0:
            loss, classifier_loss, grad_norm = sums / args.log_interval
            print({"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "gradient_steps": step + 1,
                   "avg_loss_diffusion": loss, "avg_loss_classifier": classifier_loss, "avg_grad_norm": grad_norm},
                  flush=True)
            sums[:] = 0

        stop = False
        if (step + 1) % args.eval_interval == 0 or step + 1 == args.n_gradient_steps:
            agent.eval()
            dataset, eval_rewards, eval_prefs, metrics = evaluate()
            agent.train()
            print(f"[eval step {step + 1} | {dataset.env_name}] {format_metrics(metrics)}", flush=True)
            if metrics["hv"] > best_hv:
                best_hv, bad_evals = metrics["hv"], 0
                save_eval(run_dir, "best", dataset, eval_rewards, eval_prefs, dict(metrics, step=step + 1))
                save_checkpoint(agent, run_dir, "_best")
            else:
                bad_evals += 1
                stop = 0 < args.patience <= bad_evals
            if (step + 1) % args.save_interval == 0:
                save_eval(run_dir, f"{step + 1}", dataset, eval_rewards, eval_prefs, dict(metrics, step=step + 1))
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(agent, run_dir, f"{step + 1}")
        if stop:
            print(f"Early stopping: no HV improvement in {bad_evals} evaluations.")
            break
    save_checkpoint(agent, run_dir, "_last")
