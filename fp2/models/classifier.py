"""Text classifier for classifier guidance.

The network sees a noisy segment x_t and predicts the 896-d text embedding of the segment's preference,
trained with the mean squared error; log p(text | x_t, t) = -mean((f(x_t, t) - e_txt)^2).
"""
from copy import deepcopy

import torch
import torch.nn as nn


class TextClassifier:

    def __init__(self, network: nn.Module, lr: float = 2e-4, weight_decay: float = 1e-4, ema_rate: float = 0.995,
                 device: str = "cpu"):
        self.device = device
        self.model = network
        self.model_ema = deepcopy(network).requires_grad_(False).eval()
        self.optim = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema_rate = ema_rate

    def update(self, xt, t, text):
        loss = ((self.model(xt, t) - text) ** 2).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.mul_(self.ema_rate).add_(p, alpha=1. - self.ema_rate)
        return {"loss": loss.item()}

    def logp(self, xt, t, text):
        """(b, 1) log-likelihood of the text embeddings (b, 896) under the EMA network."""
        return -((self.model_ema(xt, t) - text) ** 2).mean(dim=-1, keepdim=True)

    def grad_logp(self, xt, t, text):
        with torch.enable_grad():
            xt = xt.detach().requires_grad_()
            return torch.autograd.grad(self.logp(xt, t, text).sum(), xt)[0]

    def save(self, path):
        torch.save({"model": self.model.state_dict(), "model_ema": self.model_ema.state_dict()}, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model_ema.load_state_dict(state["model_ema"])
