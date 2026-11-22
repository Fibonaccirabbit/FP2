"""Text embeddings of the preference prompts: L2-normalized all-MiniLM-L6-v2 (384-d) and CLIP ViT-B/32 (512-d)
embeddings, concatenated (896-d).

The prompt templates produce only a few hundred distinct prompts, so embeddings are stored per prompt in one cache
file shared by all datasets, training and evaluation. The text models are loaded only when a prompt is not cached.
"""
import fcntl
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


class PromptEncoder:

    def __init__(self, sentence_model_name: str = "all-MiniLM-L6-v2", clip_model_name: str = "ViT-B/32",
                 device: str = None):
        self.sentence_model_name, self.clip_model_name = sentence_model_name, clip_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tag = f"{sentence_model_name}_{clip_model_name}".replace("/", "-")
        self.path = CACHE_DIR / f"prompt_embeddings_{tag}.npz"
        self.table: Dict[str, np.ndarray] = self._load()
        self._models = None

    def encode(self, prompts: List[str]) -> np.ndarray:
        """(len(prompts), dim) float32 embeddings."""
        missing = sorted(set(prompts) - self.table.keys())
        if missing:
            embeddings = self._embed(missing)
            with self._lock():
                self.table = {**self._load(), **self.table, **dict(zip(missing, embeddings))}
                self._save()
        if not prompts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.table[p] for p in prompts]).astype(np.float32)

    @property
    def dim(self) -> int:
        return len(next(iter(self.table.values()))) if self.table else 0

    @contextmanager
    def _lock(self):
        """Exclusive lock, so that runs updating the cache at the same time do not drop each other's prompts."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _load(self) -> Dict[str, np.ndarray]:
        if not self.path.exists():
            return {}
        try:
            with np.load(self.path) as data:
                return dict(zip(data["prompts"].tolist(), data["embeddings"]))
        except (zipfile.BadZipFile, KeyError, ValueError) as e:
            raise RuntimeError(f"Corrupt prompt-embedding cache {self.path}; delete it to rebuild it.") from e

    def _save(self):
        prompts = sorted(self.table)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp.npz")
        try:
            np.savez(tmp, prompts=np.array(prompts), embeddings=np.stack([self.table[p] for p in prompts]))
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    @torch.no_grad()
    def _embed(self, prompts: List[str]) -> np.ndarray:
        import clip
        from sentence_transformers import SentenceTransformer
        # Loading the models must not change the training randomness, whether or not the prompts were cached.
        with torch.random.fork_rng(devices=[]):
            if self._models is None:
                # CLIP runs in fp16 on GPUs by default; fp32 keeps the embeddings independent of the device.
                self._models = (SentenceTransformer(self.sentence_model_name, device=self.device),
                                clip.load(self.clip_model_name, device=self.device)[0].float())
            sentence_model, clip_model = self._models
            s = sentence_model.encode(prompts, convert_to_numpy=True, show_progress_bar=False)
            c = clip_model.encode_text(clip.tokenize(prompts, truncate=True).to(self.device)).float().cpu().numpy()
        return np.concatenate([s / np.linalg.norm(s, axis=1, keepdims=True),
                               c / np.linalg.norm(c, axis=1, keepdims=True)], axis=1)
