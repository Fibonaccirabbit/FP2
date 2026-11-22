"""The FP2 agent: flow policy with condition MLP and text classifier, its checkpoints and post-training setup."""
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from fp2.models.classifier import TextClassifier
from fp2.models.flow import DDIM, RectifiedFlow
from fp2.models.networks import ClassifierDiT, ConditionMLP, DiT1d


def build_agent(args, in_dim: int, pref_dim: int, text_dim: int, cond_hidden_dims: List[int],
                fix_mask: torch.Tensor, loss_weight: torch.Tensor, grad_clip_norm: Optional[float]):
    """Velocity DiT, condition MLP, text classifier and the flow model."""
    nn_diffusion = DiT1d(in_dim, args.emb_dim, args.d_model, args.n_heads, args.depth).to(args.device)
    nn_condition = ConditionMLP(2 * pref_dim, args.emb_dim, cond_hidden_dims, nn.SiLU()).to(args.device)
    nn_classifier = ClassifierDiT(in_dim, text_dim, args.emb_dim, args.d_model // 2, args.n_heads,
                                  max(args.depth // 2, 1)).to(args.device)
    print(f"Parameters: flow {sum(p.numel() for p in nn_diffusion.parameters()):,}, "
          f"classifier {sum(p.numel() for p in nn_classifier.parameters()):,}")
    classifier = TextClassifier(nn_classifier, device=args.device)
    flow_class = DDIM if args.backbone == "ddim" else RectifiedFlow
    return flow_class(nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm=grad_clip_norm,
                      lr=args.lr if args.lr is not None else 2e-4, contrastive_weight=args.contrastive_weight,
                      contrastive_loss_type=args.contrastive_loss_type, device=args.device)


def save_checkpoint(agent, run_dir: Path, tag: str):
    agent.save(run_dir / f"diffusion_ckpt{tag}.pt")
    agent.classifier.save(run_dir / f"classifier_ckpt{tag}.pt")


def load_checkpoint(agent, run_dir: Path, tag: str):
    path = run_dir / f"diffusion_ckpt{tag}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint {path}")
    print(f"Loading {path}")
    agent.load(path)
    agent.classifier.load(run_dir / f"classifier_ckpt{tag}.pt")


def prepare_post_training(agent, args):
    """Stage 3: l2 penalty towards the loaded weights, or LoRA adapters (--reg_mode)."""
    if args.reg_mode == "l2":
        agent.anchor_weights()
    elif args.reg_mode == "lora":
        agent.apply_lora(args.lora_rank, args.lora_alpha, args.lora_lr)
