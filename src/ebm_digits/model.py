"""Joint-class EBM: MLP or CNN energy on MNIST (e.g. 8x8 or 28x28)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class EnergyMLP(nn.Module):
    """MLP mapping flattened images (e.g. 8x8 -> 64) to 10 class energies."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.LeakyReLU(0.2, inplace=True)])
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        if x.shape[1] != self.in_dim:
            raise ValueError(f"Expected flattened dim {self.in_dim}, got {x.shape[1]}")
        return self.net(x)


class EnergyCNN(nn.Module):
    """CNN mapping [B,1,H,W] to 10 class energies; works for small H,W (8+) via adaptive pool."""

    def __init__(self, num_classes: int = 10, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected NCHW images, got shape {tuple(x.shape)}")
        return self.head(self.features(x))


def build_energy_net(
    energy_type: str,
    img_size: int,
    *,
    hidden_dims: tuple[int, ...] = (256, 256),
    num_classes: int = 10,
) -> nn.Module:
    """Create EnergyMLP or EnergyCNN for a given image side length."""
    if energy_type == "mlp":
        in_dim = img_size * img_size
        return EnergyMLP(in_dim=in_dim, hidden_dims=hidden_dims, num_classes=num_classes)
    if energy_type == "cnn":
        return EnergyCNN(num_classes=num_classes)
    raise ValueError(f"energy_type must be 'mlp' or 'cnn', got {energy_type!r}")


def energy_cfg_from_args(
    energy_type: str,
    img_size: int,
    hidden_dims: tuple[int, ...] = (256, 256),
) -> dict[str, Any]:
    cfg: dict[str, Any] = {"type": energy_type, "img_size": img_size}
    if energy_type == "mlp":
        cfg["in_dim"] = img_size * img_size
        cfg["hidden_dims"] = list(hidden_dims)
    return cfg


def load_energy_net(energy_cfg: dict[str, Any]) -> nn.Module:
    """Rebuild energy net from checkpoint metadata."""
    energy_type = energy_cfg.get("type", "mlp")
    if energy_type == "mlp":
        return EnergyMLP(
            in_dim=int(energy_cfg["in_dim"]),
            hidden_dims=tuple(int(h) for h in energy_cfg["hidden_dims"]),
        )
    if energy_type == "cnn":
        return EnergyCNN()
    raise ValueError(f"Unknown energy type {energy_type!r} in checkpoint")


class EBM(nn.Module):
    """Joint energies f(x, ·); Langevin in pixel space; gen loss as in Grau et al.-style JEM."""

    def __init__(
        self,
        energy_net: nn.Module,
        alpha: float,
        sigma: float,
        ld_steps: int,
        image_shape: tuple[int, int, int] = (1, 8, 8),
        gen_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.energy_net = energy_net
        self.nll = nn.NLLLoss(reduction="none")
        self.image_shape = image_shape
        c, h, w = image_shape
        self.D = int(c * h * w)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.ld_steps = int(ld_steps)
        self.gen_weight = float(gen_weight)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        f_xy = self.energy_net(x)
        return torch.argmax(f_xy, dim=1)

    def class_loss(self, f_xy: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(f_xy, dim=1)
        return self.nll(log_probs, y)

    def gen_loss(self, x: torch.Tensor, f_xy: torch.Tensor) -> torch.Tensor:
        """L_gen = -(logsumexp(f(x)) - logsumexp(f(x^-))) per sample (textbook JEM form)."""
        with torch.no_grad():
            x_sampled = self.sample(batch_size=x.shape[0])
        f_x_sample_y = self.energy_net(x_sampled)
        return -(torch.logsumexp(f_xy, dim=1) - torch.logsumexp(f_x_sample_y, dim=1))

    def forward(self, x: torch.Tensor, y: torch.Tensor, reduction: str = "avg") -> torch.Tensor:
        f_xy = self.energy_net(x)
        l_clf = self.class_loss(f_xy, y)
        l_gen = self.gen_loss(x, f_xy)
        combined = l_clf + self.gen_weight * l_gen
        if reduction == "sum":
            return combined.sum()
        return combined.mean()

    def energy_gradient(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.energy_net.training
        self.energy_net.eval()
        x_i = x.detach().float().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.energy_net(x_i)
            u = torch.logsumexp(logits, dim=1).sum()
            (grad,) = torch.autograd.grad(u, x_i, create_graph=False)
        self.energy_net.train(was_training)
        return grad

    def energy_gradient_conditional(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        was_training = self.energy_net.training
        self.energy_net.eval()
        x_i = x.detach().float().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.energy_net(x_i)
            idx = torch.arange(x_i.size(0), device=x_i.device, dtype=torch.long)
            u = logits[idx, y.long()].sum()
            (grad,) = torch.autograd.grad(u, x_i, create_graph=False)
        self.energy_net.train(was_training)
        return grad

    def langevin_dynamics_step(
        self,
        x_old: torch.Tensor,
        alpha: float,
        sigma: float,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if labels is None:
            grad_energy = self.energy_gradient(x_old)
        else:
            grad_energy = self.energy_gradient_conditional(x_old, labels)
        noise = torch.randn_like(grad_energy) * sigma
        x_new = x_old + alpha * grad_energy + noise
        return x_new.clamp(-1.0, 1.0)

    def sample(
        self,
        batch_size: int = 64,
        x: torch.Tensor | None = None,
        *,
        labels: torch.Tensor | None = None,
        init: str = "uniform",
        init_std: float = 0.35,
        anneal_sigma: bool = False,
        num_snapshots: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Langevin chain; default uniform init in [-1,1] (book-style).

        If num_snapshots > 0, also return intermediate frames [S, B, C, H, W]
        at evenly spaced steps (including initial noise and final image).
        """
        del x
        if labels is not None and labels.shape[0] != batch_size:
            raise ValueError("labels length must match batch_size")
        c, h, w = self.image_shape
        device = next(self.energy_net.parameters()).device
        if labels is not None:
            labels = labels.to(device=device, dtype=torch.long)

        if init == "uniform":
            x_sampled = 2.0 * torch.rand(batch_size, c, h, w, device=device, dtype=torch.float32) - 1.0
        elif init == "gaussian":
            x_sampled = torch.clamp(
                torch.randn(batch_size, c, h, w, device=device, dtype=torch.float32) * init_std,
                -1.0,
                1.0,
            )
        else:
            raise ValueError("init must be 'uniform' or 'gaussian'")

        capture = num_snapshots is not None and num_snapshots > 0
        snapshots: list[torch.Tensor] = []
        if capture:
            if num_snapshots == 1:
                mark_after_steps = {self.ld_steps}
            else:
                mark_after_steps = {
                    int(round(i * self.ld_steps / (num_snapshots - 1)))
                    for i in range(num_snapshots)
                }
            if 0 in mark_after_steps:
                snapshots.append(x_sampled.detach().clone())

        for t in range(self.ld_steps):
            sigma_t = self.sigma
            if anneal_sigma and self.ld_steps > 1:
                w_lin = 1.0 - t / (self.ld_steps - 1)
                sigma_t = self.sigma * (0.05 + 0.95 * w_lin)
            x_sampled = self.langevin_dynamics_step(
                x_sampled, self.alpha, float(sigma_t), labels=labels
            )
            if capture and (t + 1) in mark_after_steps:
                snapshots.append(x_sampled.detach().clone())

        if capture:
            return x_sampled, torch.stack(snapshots, dim=0)
        return x_sampled
