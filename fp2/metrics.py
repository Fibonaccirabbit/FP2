"""Evaluation metrics: hypervolume (HV), sparsity (SP) and the preference alignment score (PAS)."""
import numpy as np


def _dominated_volume(points: np.ndarray) -> float:
    """Volume of the union of the boxes [0, p] over the rows p of ``points`` (all coordinates >= 0).

    The last objective is swept from the top: between consecutive distinct values z_k > z_{k+1} the cross-section
    is the volume dominated, in the remaining objectives, by the points with z >= z_k.
    """
    if points.shape[1] == 1:
        return float(points.max())
    levels = np.unique(points[:, -1])[::-1]
    volume = 0.0
    for k, z in enumerate(levels):
        lower = levels[k + 1] if k + 1 < len(levels) else 0.0
        volume += (z - lower) * _dominated_volume(points[points[:, -1] >= z, :-1])
    return volume


def hypervolume(eval_rewards: np.ndarray) -> float:
    """Mean HV over evaluation episodes; eval_rewards: [n_episodes, n_prefs, n_obj], reference point 0
    (only returns with all objectives >= 0 contribute). Rounded to 4 decimals as in PEDA."""
    volumes = []
    for rewards in np.asarray(eval_rewards, dtype=np.float64):
        points = rewards[(rewards >= 0).all(axis=1)]
        volumes.append(round(_dominated_volume(points), 4) if len(points) else 0.0)
    return float(np.mean(volumes))


def sparsity(eval_rewards: np.ndarray) -> float:
    """SP of the non-dominated episode-averaged returns: for every objective the solutions are sorted and the
    squared gaps between neighbours are summed; the total is divided by N - 1 (lower is better)."""
    returns = eval_rewards.mean(0)
    front = np.array([r for i, r in enumerate(returns) if not any(
        np.all(o >= r) and np.any(o > r) for j, o in enumerate(returns) if j != i)])
    if len(front) <= 1:
        return 0.0
    return float(sum(np.sum(np.diff(np.sort(front[:, d])) ** 2) for d in range(front.shape[1])) / (len(front) - 1))


def preference_alignment_score(eval_rewards: np.ndarray, eval_prefs: np.ndarray, dataset) -> float:
    """Mean PAS over the evaluation preferences w.

    Returns are min-max normalized with the dataset's Pareto front, and y_gt is the front point whose
    preference is closest to w.
      2 objectives: 0.5 * (1 + cos(y_gt - w, y_pred - w)).
      3 objectives: 0.5 * cos01(y_gt, y_pred) + 0.25 * (cos01(y_gt, w) + cos01(y_pred, w)), cos01 = (1 + cos) / 2.
    """
    front_rew, front_pref = np.asarray(dataset.front_rew), np.asarray(dataset.front_pref)
    if len(front_rew) == 0:
        return float("nan")
    preds = eval_rewards.mean(axis=0)
    if eval_prefs.shape[-1] == 2:
        front_rew, front_pref, preds = front_rew.astype(np.float32), front_pref.astype(np.float32), preds.astype(np.float32)
        nadir, span = front_rew.min(axis=0), front_rew.max(axis=0) - front_rew.min(axis=0) + 1e-8
    else:
        nadir, span = front_rew.min(axis=0), np.clip(front_rew.max(axis=0) - front_rew.min(axis=0), 1e-8, None)

    def unit(v):
        n = np.linalg.norm(v)
        return v / (n + 1e-8) if n > 1e-8 else np.zeros_like(v)

    def cos01(a, b):
        return 0.5 * (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8) + 1.0)

    scores = []
    for pred, w in zip(preds, eval_prefs):
        if len(w) == 2:
            w = w.astype(np.float32) / (np.sum(w) + 1e-8)
            y_gt = (front_rew[np.argmin(np.linalg.norm(front_pref - w[None], axis=1))] - nadir) / span
            cos = float(np.clip(np.dot(unit(y_gt - w), unit((pred - nadir) / span - w)), -1.0, 1.0))
            scores.append(0.5 * (1.0 + cos))
        else:
            y_gt = (front_rew[np.argmin(np.linalg.norm(front_pref - w, axis=1))] - nadir) / span
            gt_dir, pred_dir = y_gt / (np.linalg.norm(y_gt) + 1e-8), (pred - nadir) / span
            pred_dir = pred_dir / (np.linalg.norm(pred_dir) + 1e-8)
            pref_dir = w / (np.linalg.norm(w) + 1e-8)
            scores.append(0.5 * cos01(gt_dir, pred_dir) + 0.25 * (cos01(gt_dir, pref_dir) + cos01(pred_dir, pref_dir)))
    return float(np.mean(np.array(scores, dtype=np.float32 if eval_prefs.shape[-1] == 2 else np.float64)))


def heldout_mask(eval_prefs: np.ndarray, heldout_prefs: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """True for evaluation preferences within ``tol`` (L2) of a held-out training preference."""
    d = np.linalg.norm(np.asarray(eval_prefs)[:, None, :] - np.asarray(heldout_prefs)[None, :, :], axis=-1)
    return (d < tol).any(axis=1)
