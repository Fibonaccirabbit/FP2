"""Policy evaluation: preference sets, rollouts in the MO-MuJoCo environments, metrics and their outputs."""
import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fp2.envs  # noqa: E402,F401  (registers the MO-MuJoCo environments)
from fp2.agent import load_checkpoint  # noqa: E402
from fp2.config import set_seed, split_load_model  # noqa: E402
from fp2.data.prompts import preference_to_text  # noqa: E402
from fp2.metrics import heldout_mask, hypervolume, preference_alignment_score, sparsity  # noqa: E402


def make_evaluator(agent, dataset, prompt_encoder, args, front_cap: Tuple[int, int],
                   model_dims: Optional[Tuple[int, int, int]] = None) -> Callable:
    """evaluate() for train(): rolls out the policy for the --eval_prefs preferences of ``dataset``.

    Front preferences are subsampled evenly: in post-training per front_cap = (max_prefs, threshold) (fronts with
    more than ``threshold`` preferences are reduced to ``max_prefs``), otherwise to at most --max_prefs.
    """
    def evaluate():
        if args.eval_prefs == "front" or (args.eval_prefs == "auto" and args.post_training):
            cap = front_cap if args.post_training else (args.max_prefs, args.max_prefs)
            prefs = front_preferences(dataset, *cap)
        else:
            prefs = preference_grid(dataset.pref_dim, args.num_prefs)
        text = encode_prompts(prompt_encoder, prefs, dataset.env_name, args.device)
        eval_rewards = rollout(agent, dataset, prefs, text, args, model_dims)
        return dataset, eval_rewards, prefs, compute_metrics(eval_rewards, prefs, dataset)
    return evaluate


def evaluate_checkpoint(agent, evaluate: Callable, args, run_dir: Path):
    """--mode eval: load <run>/<tag> (adding LoRA adapters first if the run used them), evaluate and save."""
    if args.post_training and args.reg_mode == "lora":
        agent.apply_lora(args.lora_rank, args.lora_alpha, args.lora_lr)
    load_checkpoint(agent, run_dir, split_load_model(args.load_model)[1])
    agent.eval()
    dataset, eval_rewards, prefs, metrics = evaluate()
    print(f"[{dataset.dataset_name} | {dataset.split} | {args.load_model}] {format_metrics(metrics)}")
    save_eval(run_dir, args.eval_name, dataset, eval_rewards, prefs, metrics)


def preference_grid(n_obj: int, num_prefs: int) -> np.ndarray:
    """2 objectives: w1 = linspace(0, 1, num_prefs). 3 objectives: num_prefs points picked evenly from a
    25-level simplex grid (325 points)."""
    if n_obj == 2:
        w = torch.linspace(0, 1, num_prefs).reshape(-1, 1)
        return torch.cat([w, 1 - w], dim=-1).numpy()
    grid = np.asarray([[i, j, 1 - i - j] for k, i in enumerate(np.linspace(0, 1, 25))
                       for j in np.linspace(0, 1 - i, 25 - k)], dtype=np.float32)
    if num_prefs > len(grid):
        raise ValueError(f"--num_prefs must be at most {len(grid)} for 3 objectives")
    return grid[np.linspace(0, len(grid) - 1, num_prefs, dtype=int)]


def front_preferences(dataset, max_prefs: int, threshold: int) -> np.ndarray:
    """Preferences of the dataset's Pareto front; max_prefs of them, evenly spaced, if there are more than
    ``threshold``."""
    prefs = np.asarray(dataset.front_pref, dtype=np.float32)
    if len(prefs) > threshold:
        prefs = prefs[np.linspace(0, len(prefs) - 1, max_prefs, dtype=int)]
    return prefs


def pad(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.cat([x, x.new_zeros(*x.shape[:-1], dim - x.shape[-1])], dim=-1)


def rollout(agent, dataset, prefs: np.ndarray, text: torch.Tensor, args,
            model_dims: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """Return the vector returns [num_episodes, n_prefs, n_obj] of the planner in ``dataset.env_name``.

    At every step n_elites plans are sampled for the current state (CFG on [w, flip(w) * val_scale], CG on the
    text embedding) and the first action of the plan with the highest text log-likelihood is executed. With
    ``model_dims`` (multi-task model), states, actions and preferences are zero-padded to these dimensions.
    Each episode re-seeds the global random generators (as in the original experiments), so training resumes
    from the same random state after every evaluation.
    """
    obs_dim, act_dim, pref_dim = dataset.o_dim, dataset.a_dim, dataset.pref_dim
    m_obs, m_act, m_pref = model_dims or (obs_dim, act_dim, pref_dim)
    device, horizon, k = args.device, args.horizon, args.n_elites
    fix_mask = torch.zeros((horizon, m_obs + m_act), device=device)
    fix_mask[:, obs_dim:m_obs] = 1.        # padded state dimensions
    fix_mask[:, m_obs + act_dim:] = 1.     # padded action dimensions
    fix_mask[0, :obs_dim] = 1.             # the current state
    agent.fix_mask = fix_mask[None]

    prefs_t = torch.tensor(prefs, dtype=torch.float32, device=device)
    act_normalizer = dataset.normalizers.get("action")
    eval_rewards = np.zeros((args.num_episodes, len(prefs), pref_dim), dtype=np.float32)
    for ep in range(args.num_episodes):
        seed = args.seed + ep
        set_seed(seed)
        for start in range(0, len(prefs), args.num_envs):
            n = min(args.num_envs, len(prefs) - start)
            env = gym.vector.make(dataset.env_name, n)
            obs, done = env.reset(seed=seed), np.zeros(n, dtype=bool)
            w = prefs_t[start:start + n]
            condition = torch.cat([pad(w, m_pref), pad(torch.flip(w, dims=[-1]) * args.val_scale, m_pref)], -1)
            condition, text_k = condition.repeat(k, 1), text[start:start + n].repeat(k, 1)
            for t in range(501):
                prior = torch.zeros((n, horizon, m_obs + m_act), device=device)
                prior[:, 0, :obs_dim] = torch.tensor(dataset.normalizers["state"].normalize(obs), device=device,
                                                     dtype=torch.float32)
                plans, log_p = agent.sample(prior.repeat(k, 1, 1), condition, text_k, args.w_cfg, args.w_cg,
                                            args.sample_steps, args.temperature)
                best = log_p.view(k, n, -1).sum(-1).argmax(0)
                act = plans.view(k, n, horizon, -1)[best, torch.arange(n), 0, m_obs:m_obs + act_dim].cpu().numpy()
                if act_normalizer is not None:
                    act = act_normalizer.unnormalize(act)
                obs, _, step_done, info = env.step(act)
                if t == 0:  # as in PEDA/MODULI: an episode that ends at the first step gets no reward
                    done = step_done
                for i in range(n):
                    eval_rewards[ep, start + i] += info[i]["obj"] * (1 - done[i])
                done = np.logical_or(done, step_done)
                if done.all():
                    break
            env.close()
    return eval_rewards


def encode_prompts(prompt_encoder, prefs: np.ndarray, env_name: str, device: str) -> torch.Tensor:
    """Text embeddings (n, text_dim) of the prompts of the preferences ``prefs``."""
    return torch.from_numpy(prompt_encoder.encode([preference_to_text(p, env_name) for p in prefs])).to(device)


def compute_metrics(eval_rewards: np.ndarray, eval_prefs: np.ndarray, dataset) -> dict:
    """HV, SP and PAS; for held-out splits also the HV on held-out and on seen preferences."""
    metrics = {"hv": hypervolume(eval_rewards), "sp": sparsity(eval_rewards),
               "pas": preference_alignment_score(eval_rewards, eval_prefs, dataset), "n_prefs": len(eval_prefs)}
    if dataset.split != "full":
        mask = heldout_mask(eval_prefs, dataset.dropout_prefs)
        for name, m in (("heldout", mask), ("seen", ~mask)):
            if m.any():
                metrics[f"hv_{name}"] = hypervolume(eval_rewards[:, m])
    return metrics


def format_metrics(metrics: dict) -> str:
    return ", ".join(f"{k.upper()}: {metrics[k]:.4f}" for k in ("hv", "sp", "pas", "hv_heldout", "hv_seen")
                     if k in metrics)


def save_eval(run_dir: Path, name: str, dataset, eval_rewards: np.ndarray, eval_prefs: np.ndarray, metrics: dict):
    """eval_<name>.npz (returns, preferences), eval_<name>.json (metrics) and eval_<name>.pdf (plot)."""
    np.savez(run_dir / f"eval_{name}.npz", rewards=eval_rewards, prefs=eval_prefs)
    with open(run_dir / f"eval_{name}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    returns, front = eval_rewards.mean(0), np.asarray(dataset.front_rew)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d" if returns.shape[1] == 3 else None)
    if len(front):
        ax.scatter(*front.T, c="lightgrey", s=6, label="dataset front")
    points = ax.scatter(*returns.T, c=eval_prefs[:, 0], cmap="viridis", s=12, label="FP2")
    fig.colorbar(points, ax=ax, label="$w_1$")
    ax.set_title(dataset.env_name)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / f"eval_{name}.pdf")
    plt.close(fig)
