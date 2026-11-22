"""Check the D4MORL datasets and precompute the text embeddings of their preference prompts (training data and
evaluation grids).

    python prepare_data.py                       # every dataset found under data/
    python prepare_data.py --datasets MO-Hopper-v2_50000_expert_uniform MO-Swimmer-v2_50000_expert_uniform

Optional: the embeddings are otherwise computed on first use. Afterwards, training and evaluation need neither the
text models nor network access.
"""
import argparse
import pickle
import re

import numpy as np

from fp2.data.dataset import DATA_DIR
from fp2.data.prompts import preference_to_text
from fp2.data.text_encoder import PromptEncoder
from fp2.envs import TASKS
from fp2.evaluate import preference_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", help="Dataset names; default: all files under data/<env>/.")
    parser.add_argument("--sentence_model_name", default="all-MiniLM-L6-v2")
    parser.add_argument("--clip_model_name", default="ViT-B/32")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    pattern = re.compile(r"MO-[A-Za-z0-9]+-v\d_\d+_(expert|amateur)_uniform")  # D4MORL file names
    names = args.datasets or sorted(p.stem for env in TASKS for p in (DATA_DIR / env).glob("*.pkl")
                                    if pattern.fullmatch(p.stem))
    if not names:
        raise SystemExit(f"No datasets found. Download D4MORL (https://github.com/baitingzbt/PEDA) and place the "
                         f"files as {DATA_DIR}/<env>/<env>_50000_<expert|amateur>_uniform.pkl")
    encoder = PromptEncoder(args.sentence_model_name, args.clip_model_name, args.device)
    for name in names:
        path = DATA_DIR / name.split("_")[0] / f"{name}.pkl"
        if not path.exists():
            print(f"[missing] {path}")
            continue
        with open(path, "rb") as f:
            trajectories = pickle.load(f)
        if not (isinstance(trajectories, list) and trajectories and "preference" in trajectories[0]):
            print(f"[skipped] {path}: not a D4MORL trajectory list")
            continue
        prefs = np.asarray([traj["preference"][0] for traj in trajectories], dtype=np.float32)
        # prompts of the training data and of the evaluation preference grids
        env = name.split("_")[0]
        grids = [preference_grid(prefs.shape[1], n) for n in (100, 101)]
        prompts = sorted({preference_to_text(p, env) for p in np.concatenate([prefs, *grids])})
        encoder.encode(prompts)
        print(f"[ok] {name}: {len(prefs)} trajectories, {len(prompts)} distinct prompts")
    print(f"Prompt embeddings: {encoder.path} ({len(encoder.table)} prompts)")


if __name__ == "__main__":
    main()
