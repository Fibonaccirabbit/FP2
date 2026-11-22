"""FP2 policy: a rectified flow over trajectory segments, plus the DDIM backbone of the ablation.

The flow learns a velocity v(x_t, t | c) with x_t = (1 - t) x_0 + t * eps (x_0: data, eps ~ N(0, I)) and target
v* = eps - x_0. At test time it integrates from noise to data with
  * classifier-free guidance (CFG) on the numerical condition c = [preference, return-to-go], and
  * classifier guidance (CG) from the text classifier, with weight w_cg * t.
Training adds a triplet regularizer on the velocity field and, for post-training, either an l2 penalty towards
the starting (FP2-Mid) weights or LoRA adapters.

Adapted from CleanDiffuser (https://github.com/CleanDiffuserTeam/CleanDiffuser).
"""
import math
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def expand_to(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """(b,) -> (b, 1, ..., 1) with ``ndim`` dimensions."""
    return x.reshape(x.shape + (1,) * (ndim - x.dim()))


# ----------------------------------------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """y = W x + (alpha / rank) B A x with W frozen; B starts at zero, so wrapping does not change the output."""

    def __init__(self, original: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.original = original.requires_grad_(False)
        w = original.weight
        self.lora_A = nn.Parameter(torch.empty(rank, original.in_features, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank, device=w.device, dtype=w.dtype))
        self.scaling = alpha / rank
        self.enabled = True

    # nn.MultiheadAttention reads the weights of its out_proj directly.
    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def forward(self, x):
        out = self.original(x)
        if self.enabled:
            out = out + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return out


def inject_lora(module: nn.Module, rank: int, alpha: float):
    """Wrap the Linear layers of a DiT (MLPs, output head) with LoRA; adaLN modulations are skipped.

    The attention output projections are wrapped as well, but nn.MultiheadAttention reads their weights directly,
    so these adapters stay at zero. They are kept so that LoRA runs reproduce the original experiments.
    """
    for name, child in module.named_children():
        if name == "adaLN_modulation":
            continue
        if isinstance(child, nn.MultiheadAttention):
            child.out_proj = LoRALinear(child.out_proj, rank, alpha)
        elif isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
        else:
            inject_lora(child, rank, alpha)


def set_lora_enabled(module: nn.Module, enabled: bool):
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.enabled = enabled


# ------------------------------------------------------------------------------------------------- rectified flow
class RectifiedFlow:
    # The batch-level triplet term is used as a hinge, max(0, .), in the rectified flow (not in the DDIM ablation).
    hinge_contrastive = True

    def __init__(self, nn_diffusion: nn.Module, nn_condition: nn.Module, fix_mask: torch.Tensor,
                 loss_weight: torch.Tensor, classifier, grad_clip_norm: Optional[float] = None,
                 lr: float = 2e-4, weight_decay: float = 1e-5, ema_rate: float = 0.995, epsilon: float = 1e-3,
                 cfg_dropout: float = 0.1, contrastive_weight: float = 0.01, contrastive_temperature: float = 1.0,
                 contrastive_loss_type: str = "triplet", weight_reg_lambda: float = 0.1, device: str = "cpu"):
        """
        Args:
            fix_mask: (horizon, dim), 1 for entries that are given (the current state) rather than generated.
            loss_weight: (horizon, dim) weight of every entry in the flow-matching loss.
            classifier: text classifier used for CG and for ranking the sampled plans.
            cfg_dropout: probability of dropping the condition of a whole batch (CFG).
            contrastive_weight: weight of the triplet / InfoNCE regularizer (0 disables it).
            contrastive_temperature: lambda of the triplet loss, or the InfoNCE temperature.
            weight_reg_lambda: scale of the l2 post-training penalty (see ``anchor_weights``).
        """
        self.device = device
        self.model = nn.ModuleDict({"diffusion": nn_diffusion.to(device), "condition": nn_condition.to(device)})
        self.model_ema = deepcopy(self.model).requires_grad_(False).eval()
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema_rate = ema_rate
        self.grad_clip_norm = grad_clip_norm
        self.classifier = classifier
        self.fix_mask = fix_mask[None]
        self.loss_weight = loss_weight[None]

        self.t_range = (epsilon, 1.0)
        self.cfg_dropout = cfg_dropout
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_loss_type = contrastive_loss_type
        self.weight_reg_lambda = weight_reg_lambda
        self._anchor = {}
        self._lora_enabled = False

    # --------------------------------------------------------------------------------------- state
    def train(self):
        self.model.train()
        self.classifier.model.train()

    def eval(self):
        self.model.eval()
        self.classifier.model.eval()

    def save(self, path):
        torch.save({"model": self.model.state_dict(), "model_ema": self.model_ema.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.mul_(self.ema_rate).add_(p, alpha=1. - self.ema_rate)

    # ------------------------------------------------------------------------------- post-training
    def anchor_weights(self):
        """l2 post-training: penalize weight_reg_lambda * sum_tensors mean((theta - theta_anchor)^2) from now on."""
        self._anchor = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def apply_lora(self, rank: int, alpha: float, lr: float):
        """LoRA post-training: freeze all weights and optimize only adapters on the DiT."""
        for m in (self.model, self.model_ema):
            m.requires_grad_(False)
            inject_lora(m["diffusion"], rank, alpha)
        self.model_ema.requires_grad_(False)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
        self._lora_enabled = True
        print(f"[LoRA] rank={rank}: {sum(p.numel() for p in params):,} trainable parameters")

    # ------------------------------------------------------------------------------------ training
    def add_noise(self, x0):
        t = torch.rand((x0.shape[0],), device=self.device) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        eps = torch.randn_like(x0)
        xt = (1.0 - expand_to(t, x0.dim())) * x0 + expand_to(t, x0.dim()) * eps
        return (1. - self.fix_mask) * xt + self.fix_mask * x0, t, eps

    @staticmethod
    def regression_target(x0, eps):
        return eps - x0

    @staticmethod
    def hard_negatives(pref: torch.Tensor, task_id: Optional[torch.Tensor] = None, eps_same: float = 1e-6):
        """Index of the nearest-preference other sample in the batch (identical preferences excluded).

        With ``task_id`` (multi-task training) negatives come from the same task, plus a random 10% of the
        cross-task pairs.
        """
        b = pref.shape[0]
        dist = torch.norm(pref.unsqueeze(1) - pref.unsqueeze(0), p=2, dim=-1)
        invalid = torch.eye(b, device=pref.device, dtype=torch.bool) | (dist < eps_same)
        if task_id is not None:
            cross_task = task_id.unsqueeze(1) != task_id.unsqueeze(0)
            invalid |= cross_task & ~(torch.rand_like(cross_task.float()) < 0.1)
        sim = 1.0 - dist / (dist.max() + 1e-8)
        sim[invalid] = -float("inf")
        neg = torch.argmax(sim, dim=1)
        for i in torch.where(torch.isinf(sim).all(dim=1))[0]:  # no valid candidate: any other sample
            others = torch.arange(b, device=pref.device)
            others = others[others != i]
            neg[i] = others[torch.randint(0, others.numel(), (1,), device=pref.device)]
        return neg

    def contrastive_loss(self, pred, target, neg):
        """triplet: mean_b[d(pred_b, target_b) - lambda * d(pred_b, target_neg(b))] with d the mean squared error;
        infonce: cross-entropy over the cosine similarities of predictions and targets. Fixed entries are ignored."""
        b = pred.shape[0]
        pred = (pred * (1 - self.fix_mask)).view(b, -1)
        target = (target * (1 - self.fix_mask)).view(b, -1)
        if self.contrastive_loss_type == "triplet":
            pos = torch.mean((pred - target) ** 2, dim=-1)
            negative = torch.mean((pred - target[neg]) ** 2, dim=-1)
            return (pos - self.contrastive_temperature * negative).mean()
        sim = F.normalize(pred, dim=-1) @ F.normalize(target, dim=-1).t() / self.contrastive_temperature
        return F.cross_entropy(sim, torch.arange(b, device=pred.device))

    def loss(self, x0, condition, pref, attention_mask=None, task_id=None):
        """
        Args:
            x0: (b, horizon, dim) trajectory segments; condition: (b, cond_dim); pref: (b, n_obj).
            attention_mask, task_id: multi-task training (padded time steps, task-restricted negatives).
        """
        b = x0.shape[0]
        xt, t, eps = self.add_noise(x0)
        if torch.rand(1).item() < self.cfg_dropout:
            condition = None
        cond = self.model["condition"](condition) if condition is not None else None
        pred = self.model["diffusion"](xt, t, cond, attention_mask=attention_mask)
        target = self.regression_target(x0, eps)

        loss = (((pred - target) ** 2) * self.loss_weight * (1 - self.fix_mask)).view(b, -1).mean(dim=1)
        if cond is not None and self.contrastive_weight > 0 and b > 1:
            contrastive = self.contrastive_loss(pred, target, self.hard_negatives(pref, task_id))
            if self.hinge_contrastive:
                contrastive = contrastive.clamp(min=0.0)
            loss = loss + self.contrastive_weight * contrastive
        loss = loss.mean()
        if self._anchor:
            loss = loss + self.weight_reg_lambda * sum(
                F.mse_loss(p, self._anchor[n]) for n, p in self.model.named_parameters() if n in self._anchor)
        return loss

    def update(self, x0, condition, pref, **kwargs):
        """One optimizer step; returns the loss and the gradient norm (before clipping)."""
        loss = self.loss(x0, condition, pref, **kwargs)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm or float("inf"))
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.ema_update()
        return {"loss": loss.item(), "grad_norm": grad_norm.item()}

    def update_classifier(self, x0, text):
        xt, t, _ = self.add_noise(x0)
        return self.classifier.update(xt, t, text)

    # ------------------------------------------------------------------------------------ guidance
    @torch.no_grad()
    def cfg_prediction(self, xt, t, condition, w):
        """w * f(x_t, t | c) + (1 - w) * f(x_t, t | null)."""
        model = self.model_ema
        if w == 0.0:
            return model["diffusion"](xt, t, None)
        cond = model["condition"](condition)
        if w == 1.0:
            return model["diffusion"](xt, t, cond)
        if self._lora_enabled:  # the unconditional branch uses the model without adapters
            pred = model["diffusion"](xt, t, cond)
            set_lora_enabled(model["diffusion"], False)
            pred_uncond = model["diffusion"](xt, t, torch.zeros_like(cond))
            set_lora_enabled(model["diffusion"], True)
        else:
            both = model["diffusion"](torch.cat([xt, xt]), torch.cat([t, t]), torch.cat([cond, torch.zeros_like(cond)]))
            pred, pred_uncond = both[:xt.shape[0]], both[xt.shape[0]:]
        return w * pred + (1 - w) * pred_uncond

    def classifier_gradient(self, xt, t, text, pred):
        """grad_x log p(text | x_t, t), rescaled so that its norm never exceeds the norm of ``pred``."""
        grad = self.classifier.grad_logp(xt, t, text)
        p_norm = pred.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-6)
        g_norm = grad.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-6)
        return grad * expand_to((p_norm / g_norm).clamp(max=1.0).squeeze(1), grad.dim())

    def guided_prediction(self, xt, t, condition, w_cfg, text, w_cg):
        pred = self.cfg_prediction(xt, t, condition, w_cfg)
        w = w_cg * t[0].item()
        return pred + w * self.classifier_gradient(xt, t, text, pred) if w != 0.0 else pred

    def sample(self, prior, condition, text, w_cfg=1.0, w_cg=0.0, sample_steps=10, temperature=1.0):
        """Integrate the guided ODE from t = 1 to t = epsilon (Euler) with the EMA model.

        Args:
            prior: (n, horizon, dim); the entries selected by fix_mask (the current state) are kept.
            condition: (n, cond_dim) numerical condition; text: (n, text_dim) text embedding.
        Returns:
            plans (n, horizon, dim) and their classifier log-likelihood (n, 1).
        """
        xt = torch.randn_like(prior) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        ts = torch.linspace(self.t_range[1], self.t_range[0], sample_steps + 1).to(self.device)
        for i in range(sample_steps):
            t = torch.full((prior.shape[0],), ts[i], dtype=torch.float32, device=self.device)
            xt = xt + (ts[i + 1] - ts[i]) * self.guided_prediction(xt, t, condition, w_cfg, text, w_cg)
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        return xt, self.final_log_p(xt, text, self.t_range[0])

    @torch.no_grad()
    def final_log_p(self, x, text, t):
        return self.classifier.logp(x, torch.full((x.shape[0],), t, dtype=torch.float32, device=self.device), text)


# ----------------------------------------------------------------------------------------------- DDIM ablation
def cosine_noise_schedule(t: torch.Tensor, s: float = 0.008):
    """alpha(t), sigma(t) of the cosine schedule. As in CleanDiffuser, the last entry of ``t`` is set to 0.9946
    in place."""
    t[-1] = 0.9946
    alpha = (np.pi / 2. * (t + s) / (1 + s)).cos() / np.cos(np.pi / 2. * s / (1 + s))
    return alpha, (1. - alpha ** 2).sqrt()


class DDIM(RectifiedFlow):
    """x_t = alpha(t) x_0 + sigma(t) eps with noise prediction, CG as eps <- eps - w sigma(t) grad, and the
    deterministic DDIM sampler; everything else is shared with the rectified flow."""
    hinge_contrastive = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_range = (self.t_range[0], 0.9946)

    def add_noise(self, x0):
        t = torch.rand((x0.shape[0],), device=self.device) * (self.t_range[1] - self.t_range[0]) + self.t_range[0]
        eps = torch.randn_like(x0)
        alpha, sigma = cosine_noise_schedule(t)
        xt = expand_to(alpha, x0.dim()) * x0 + expand_to(sigma, x0.dim()) * eps
        return (1. - self.fix_mask) * xt + self.fix_mask * x0, t, eps

    @staticmethod
    def regression_target(x0, eps):
        return eps

    def guided_prediction(self, xt, t, condition, w_cfg, text, w_cg):
        pred = self.cfg_prediction(xt, t, condition, w_cfg)
        w = w_cg * t[0].item()
        if w == 0.0:
            return pred
        grad = self.classifier_gradient(xt, t, text, pred)
        sigma = cosine_noise_schedule(t)[1]  # after the gradient: the schedule modifies t in place
        return pred - w * expand_to(sigma, pred.dim()) * grad

    def sample(self, prior, condition, text, w_cfg=1.0, w_cg=0.0, sample_steps=10, temperature=1.0):
        xt = torch.randn_like(prior) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        ts = torch.linspace(self.t_range[0], self.t_range[1], sample_steps + 1, dtype=torch.float32)
        alphas, sigmas = (v.to(self.device) for v in cosine_noise_schedule(ts))
        for i in reversed(range(1, sample_steps + 1)):
            t = torch.full((prior.shape[0],), ts[i], dtype=torch.float32, device=self.device)
            eps_pred = self.guided_prediction(xt, t, condition, w_cfg, text, w_cg)
            x0_pred = (xt - sigmas[i] * eps_pred) / alphas[i].clamp(min=1e-8)
            xt = alphas[i - 1] * x0_pred + sigmas[i - 1] * eps_pred
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        return xt, self.final_log_p(xt, text, 0.0)
