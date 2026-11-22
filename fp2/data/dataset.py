"""D4MORL trajectory segments with preference, return and text conditions.

One dataset class serves every preference split and training stage:

    split    full       all trajectories (no held-out preference region)
             shattered  30% of the trajectories held out at three interior preference regions
             narrow     30% of the trajectories held out at the two preference extremes

    stage 1  pre-train   all retained trajectories
          2  mid-train   top-two non-dominated fronts of the retained trajectories
          3  post-train  non-dominated trajectories inside the held-out regions

Trajectories are sorted by the first preference weight and the held-out regions are defined on that order
(as in PEDA / MODULI). Pareto fronts are computed on the trajectories that remain after the split.
"""
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fp2.data.prompts import preference_to_text
from fp2.data.text_encoder import PromptEncoder

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SPLITS = ("full", "shattered", "narrow")
STAGE_SUBSETS = {1: "all", 2: "fronts", 3: "heldout_fronts"}
MAX_PATH_LENGTH = 500
DISCOUNT = 0.997  # discount of the return-to-go condition within a segment


class GaussianNormalizer:

    def __init__(self, x):
        self.means, self.stds = x.mean(0), x.std(0)
        self.stds[self.stds == 0] = 1.

    def normalize(self, x):
        return (x - self.means[None, ]) / self.stds[None, ]


class MinMaxNormalizer:
    """Maps every feature to [-1, 1]."""

    def __init__(self, x):
        self.min, self.max = np.min(x, axis=0), np.max(x, axis=0)
        self.range = self.max - self.min
        self.range[self.range == 0] = 1

    def normalize(self, x):
        return (x.astype(np.float32) - self.min) / self.range * 2 - 1

    def unnormalize(self, x):
        return (x.astype(np.float32) + 1) / 2 * self.range + self.min


def undominated_indices(returns) -> np.ndarray:
    """Indices of the non-dominated rows with non-negative returns, in increasing order of the first objective."""
    returns = np.asarray(returns)
    candidates = np.flatnonzero((returns >= 0).all(axis=1))
    r = returns[candidates]
    # In descending lexicographic order every dominator of a point comes before it, and is itself dominated only by
    # earlier points, so each point needs to be compared with the non-dominated points found so far.
    sweep = np.lexsort(tuple(-r[:, k] for k in reversed(range(r.shape[1]))))
    front = np.empty_like(r)
    n_front = 0
    keep = np.zeros(len(returns), dtype=bool)
    for j in sweep:
        f = front[:n_front]
        if not ((f >= r[j]).all(1) & (f > r[j]).any(1)).any():
            front[n_front] = r[j]
            n_front += 1
            keep[candidates[j]] = True
    order = np.argsort(returns[:, 0])
    return order[keep[order]]


def heldout_indices(n, split, percent=0.3, points=3):
    """Indices (in preference order) of the held-out trajectories of a split (``shattered`` for ``full``)."""
    if split == "narrow":
        radius = int(n * percent / 2)
        return list(range(0, radius + 1)) + [n - 1 - i for i in range(0, radius + 1)]
    radius = int(n * percent / points / 2)
    centres = np.linspace(0, n - 1, points + 2, dtype=np.int32)[1:-1]
    return [c + j for c in centres for j in range(-radius, radius + 1)]


class D4MORLDataset(Dataset):
    """Segments of ``horizon`` steps. Items: normalized states and actions, the discounted normalized return of the
    segment (``val``), the trajectory preference and the text embedding of its prompt.

    Attributes used for evaluation: ``front_rew`` / ``front_pref`` (Pareto front of the retained trajectories) and
    ``dropout_prefs`` (preferences of the held-out trajectories).
    """

    def __init__(self, dataset_name: str, split: str = "full", subset: str = "all", horizon: int = 32,
                 eps: float = 1e-3, discount: float = 0.997, sentence_model_name: str = "all-MiniLM-L6-v2",
                 clip_model_name: str = "ViT-B/32", device: str = None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
        if subset == "heldout_fronts" and split == "full":
            raise ValueError("The full split has no held-out region; use 'shattered' or 'narrow'.")
        self.dataset_name, self.split, self.subset = dataset_name, split, subset
        self.env_name = dataset_name.split("_")[0]

        with open(DATA_DIR / self.env_name / f"{dataset_name}.pkl", "rb") as f:
            trajectories = pickle.load(f)
        self.o_dim = trajectories[0]["observations"].shape[-1]
        self.a_dim = trajectories[0]["actions"].shape[-1]
        self.pref_dim = trajectories[0]["preference"].shape[-1]
        n, T = len(trajectories), MAX_PATH_LENGTH

        obs = np.zeros((n, T, self.o_dim), dtype=np.float32)
        act = np.zeros((n, T, self.a_dim), dtype=np.float32)
        rew = np.zeros((n, T, self.pref_dim), dtype=np.float32)
        score = np.zeros((n, T, self.pref_dim), dtype=np.float32)
        pref = np.zeros((n, self.pref_dim), dtype=np.float32)
        seq_len = np.zeros((n,), dtype=np.int32)
        remaining_steps = np.arange(T, 0, -1, dtype=np.float32)[:, None].repeat(self.pref_dim, -1)
        for i, traj in enumerate(trajectories):
            length = traj["observations"].shape[0]
            obs[i, :length], act[i, :length], rew[i, :length] = traj["observations"], traj["actions"], traj["raw_rewards"]
            pref[i], seq_len[i] = traj["preference"][0], length
            # return-to-go, divided by the number of steps left until the 500-step limit
            score[i, :length] = np.cumsum(traj["raw_rewards"][::-1], axis=0)[::-1]
            score[i] /= remaining_steps
        del trajectories

        # Normalizers are fitted on all trajectories, so that every split and stage shares the same scaling.
        flat_obs, flat_act = obs.reshape(-1, self.o_dim), act.reshape(-1, self.a_dim)
        self.normalizers = {"state": GaussianNormalizer(flat_obs)}
        flat_obs[:] = self.normalizers["state"].normalize(flat_obs)
        if self.pref_dim == 2:  # actions of the 2-objective tasks are normalized (and un-normalized at rollout)
            self.normalizers["action"] = MinMaxNormalizer(flat_act)
            flat_act[:] = self.normalizers["action"].normalize(flat_act)

        # Held-out regions are defined on the trajectories sorted by the first preference weight.
        order = np.argsort(pref[:, 0])
        dropout = heldout_indices(n, "narrow" if split == "narrow" else "shattered")
        self.dropout_prefs = pref[order[dropout]]
        if split == "full":
            keep = np.arange(n)
        elif subset == "heldout_fronts":
            keep = np.sort(dropout)
        else:
            keep = np.setdiff1d(np.arange(n), dropout)
        sel = order[keep]  # retained trajectories, still in preference order

        self.seq_obs, self.seq_act, self.seq_pref = obs[sel], act[sel], pref[sel]
        del obs, act
        self.seq_score = self._normalize_scores(score[sel], eps)
        returns = rew[sel].sum(1)
        front = undominated_indices(returns)
        rest = np.setdiff1d(np.arange(len(sel)), front)
        second_front = rest[undominated_indices(returns[rest])]
        self.front_rew, self.front_pref = returns[front], self.seq_pref[front]

        # Segment index: (trajectory, start step) of every segment of the trajectories used by this stage.
        used = {"all": np.arange(len(sel)), "fronts": np.union1d(front, second_front),
                "heldout_fronts": np.sort(front)}[subset]
        n_segments = np.maximum(seq_len[sel][used] - horizon + 1, 0)
        self.segment_traj = np.repeat(used, n_segments).astype(np.int32)
        self.segment_start = (np.arange(n_segments.sum()) - np.repeat(np.cumsum(n_segments) - n_segments,
                                                                      n_segments)).astype(np.int32)
        self.horizon = horizon
        self.discount = (discount ** np.arange(horizon, dtype=np.float32))[:, None]

        # Text prompts: few distinct ones, embedded once each.
        texts = [preference_to_text(p, self.env_name) for p in self.seq_pref]
        self.prompts = sorted(set(texts))
        prompt_id = {p: k for k, p in enumerate(self.prompts)}
        self.prompt_ids = np.array([prompt_id[t] for t in texts], dtype=np.int32)
        self.prompt_embeddings = PromptEncoder(sentence_model_name, clip_model_name, device).encode(self.prompts)
        self.text_dim = self.prompt_embeddings.shape[1]
        print(f"[D4MORL] {dataset_name} | split={split} subset={subset} | {len(sel)} trajectories, "
              f"{len(used)} used, {len(self.segment_traj)} segments, {len(self.prompts)} prompts")

    def _normalize_scores(self, score, eps):
        """Min-max normalize the return-to-go among trajectories with neighbouring preferences:
        |w1 - w1'| <= eps for 2 objectives, ||w - w'|| < eps * sqrt(2) for 3."""
        normalized = np.zeros_like(score)
        if self.pref_dim == 2:
            p = self.seq_pref[:, 0]
            left, right = np.searchsorted(p, p - eps, side="left"), np.searchsorted(p, p + eps, side="right")
            neighbours = (slice(lo, hi) for lo, hi in zip(left, right))
        else:
            radius = eps * np.sqrt(2)
            p = self.seq_pref[:, 0]  # sorted, and |w1 - w1'| <= ||w - w'||: only a window can contain neighbours
            left = np.searchsorted(p, p - 1.01 * radius - 1e-6, side="left")
            right = np.searchsorted(p, p + 1.01 * radius + 1e-6, side="right")
            neighbours = (lo + np.flatnonzero(np.linalg.norm(self.seq_pref[lo:hi] - w, 2, axis=1) < radius)
                          for w, lo, hi in zip(self.seq_pref, left, right))
        for i, nb in enumerate(neighbours):
            lo, hi = score[nb].min(0), score[nb].max(0)
            normalized[i] = (score[i] - lo) / (hi - lo + 1e-6)
        return normalized

    def __len__(self):
        return len(self.segment_traj)

    def __getitem__(self, idx):
        i, start = self.segment_traj[idx], self.segment_start[idx]
        end = start + self.horizon
        return {
            "obs": torch.tensor(self.seq_obs[i, start:end]),
            "act": torch.tensor(self.seq_act[i, start:end]),
            "val": torch.tensor((self.seq_score[i, start:end] * self.discount).sum(0)),
            "pref": torch.tensor(self.seq_pref[i]),
            "text_embedding": torch.tensor(self.prompt_embeddings[self.prompt_ids[i]]),
        }


def build_dataset(dataset_name: str, stage: int, split: str, **kwargs) -> D4MORLDataset:
    return D4MORLDataset(dataset_name, split=split, subset=STAGE_SUBSETS[stage], **kwargs)


def return_scale(horizon: int) -> float:
    """Normalizer of the segment return-to-go condition."""
    return (DISCOUNT ** np.arange(horizon, dtype=np.float32)).sum()
