"""Generate samples; defaults match textbook-style 8x8 + ~20 Langevin steps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from ebm_digits.model import EBM, load_energy_net


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from trained EBM")
    p.add_argument("--checkpoint", type=Path, default=Path("runs/ebm_checkpoint.pt"))
    p.add_argument("--output", type=Path, default=Path("runs/samples.png"))
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument(
        "--label",
        type=int,
        default=None,
        choices=list(range(10)),
        help="Generate one digit of this class (0-9); sets n-samples=1 and conditional mode",
    )
    p.add_argument(
        "--sample-steps",
        type=int,
        default=20,
        help="Langevin steps at inference (book figure often uses ~20; increase if needed)",
    )
    p.add_argument(
        "--num-snapshots",
        type=int,
        default=0,
        help="If > 0, save this many evenly spaced frames during Langevin (noise -> final)",
    )
    p.add_argument(
        "--snapshots-output",
        type=Path,
        default=None,
        help="PNG for intermediate chain (default: <output_stem>_chain.png)",
    )
    p.add_argument(
        "--langevin-alpha",
        type=float,
        default=None,
        help="Step size (default: value stored in checkpoint)",
    )
    p.add_argument(
        "--langevin-sigma",
        type=float,
        default=None,
        help="Noise scale (default: value stored in checkpoint)",
    )
    p.add_argument(
        "--viz-scale",
        type=int,
        default=0,
        help="If > 0 and larger than model resolution, upsample each patch for viewing (0 = native pixels)",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--conditional",
        action="store_true",
        help="Condition on digit labels (cycles 0..9 across the batch)",
    )
    return p.parse_args()


def _to_display(x: torch.Tensor, viz_scale: int) -> torch.Tensor:
    imgs = ((x.cpu() + 1.0) * 0.5).clamp(0, 1)
    if viz_scale > 0 and imgs.shape[-1] < viz_scale:
        imgs = F.interpolate(imgs, size=(viz_scale, viz_scale), mode="nearest")
    return imgs


def _snapshot_step_indices(num_snapshots: int, ld_steps: int) -> list[int]:
    """Ordered Langevin step index per frame (matches capture order in EBM.sample)."""
    if num_snapshots <= 0:
        return []
    if num_snapshots == 1:
        return [ld_steps]
    target_steps = {
        int(round(i * ld_steps / (num_snapshots - 1))) for i in range(num_snapshots)
    }
    ordered: list[int] = []
    if 0 in target_steps:
        ordered.append(0)
    for t in range(ld_steps):
        if (t + 1) in target_steps:
            ordered.append(t + 1)
    return ordered


def _save_snapshot_grid(
    snapshots: torch.Tensor,
    path: Path,
    n_samples: int,
    viz_scale: int,
    ld_steps: int,
) -> None:
    """snapshots: [S, B, C, H, W] -> PNG with Langevin step labels on a time axis."""
    s_count = snapshots.shape[0]
    steps = _snapshot_step_indices(s_count, ld_steps)
    if len(steps) != s_count:
        steps = list(range(s_count))

    if n_samples == 1:
        nrows, ncols = 1, s_count
    else:
        nrows, ncols = s_count, n_samples

    cell = max(1.0, viz_scale / 28.0) if viz_scale > 0 else 0.45
    fig_w = max(4.0, ncols * cell * 1.35)
    fig_h = max(2.0, nrows * cell * 1.55)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for s in range(s_count):
        imgs = _to_display(snapshots[s], viz_scale)
        for b in range(n_samples):
            ax = axes[s, b] if n_samples > 1 else axes[0, s]
            ax.imshow(imgs[b, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    if n_samples == 1:
        axes[0, 0].figure.subplots_adjust(bottom=0.22)
        for s, step in enumerate(steps):
            axes[0, s].set_xlabel(str(step), fontsize=9)
        fig.text(
            0.5,
            0.04,
            "Langevin step",
            ha="center",
            va="center",
            fontsize=10,
        )
        if ncols > 1:
            fig.add_artist(
                plt.Line2D(
                    [0.08, 0.92],
                    [0.12, 0.12],
                    transform=fig.transFigure,
                    color="0.35",
                    linewidth=1.0,
                    clip_on=False,
                )
            )
            fig.text(0.06, 0.115, "0", ha="right", va="center", fontsize=8, color="0.35")
            fig.text(0.94, 0.115, str(ld_steps), ha="left", va="center", fontsize=8, color="0.35")
    else:
        for s, step in enumerate(steps):
            axes[s, 0].set_ylabel(f"step {step}", fontsize=8, rotation=0, labelpad=28, va="center")
        fig.text(0.02, 0.5, "Langevin step", ha="center", va="center", rotation=90, fontsize=10)

    fig.suptitle(f"Langevin sampling chain (step 0 → {ld_steps})", fontsize=11, y=0.98)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    n_samples = 1 if args.label is not None else args.n_samples
    use_conditional = args.conditional or args.label is not None

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["ebm_config"]
    energy_cfg = ckpt.get("energy_cfg")
    if energy_cfg is None:
        raise RuntimeError("Checkpoint missing energy_cfg. Train again with the current ebm-train.")

    energy_net = load_energy_net(energy_cfg).to(device)

    energy_net.load_state_dict(ckpt["energy_net"])

    alpha = float(cfg["alpha"]) if args.langevin_alpha is None else args.langevin_alpha
    sigma = float(cfg["sigma"]) if args.langevin_sigma is None else args.langevin_sigma

    ebm = EBM(
        energy_net,
        alpha=alpha,
        sigma=sigma,
        ld_steps=args.sample_steps,
        image_shape=tuple(cfg["image_shape"]),
        gen_weight=float(cfg.get("gen_weight", 1.0)),
    ).to(device)
    ebm.eval()

    labels: torch.Tensor | None = None
    if use_conditional:
        if args.label is not None:
            labels = torch.full((n_samples,), args.label, device=device, dtype=torch.long)
        else:
            labels = torch.arange(n_samples, device=device) % 10

    num_snapshots = args.num_snapshots if args.num_snapshots > 0 else None
    if num_snapshots is not None:
        out = ebm.sample(
            batch_size=n_samples,
            labels=labels,
            num_snapshots=num_snapshots,
        )
        assert isinstance(out, tuple)
        samples, snapshots = out
    else:
        samples = ebm.sample(batch_size=n_samples, labels=labels)
        snapshots = None

    imgs = _to_display(samples, args.viz_scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid_nrow = 1 if n_samples == 1 else 8
    save_image(imgs, str(args.output), nrow=grid_nrow, padding=0, pad_value=0)
    h, w = int(samples.shape[-2]), int(samples.shape[-1])
    print(f"wrote {args.output} ({n_samples} sample(s), each {h}×{w} px)")

    if snapshots is not None:
        chain_path = args.snapshots_output
        if chain_path is None:
            chain_path = args.output.with_name(f"{args.output.stem}_chain{args.output.suffix}")
        _save_snapshot_grid(
            snapshots, chain_path, n_samples, args.viz_scale, args.sample_steps
        )
        print(
            f"wrote {chain_path} ({snapshots.shape[0]} frames, "
            f"steps 0..{args.sample_steps} during Langevin)"
        )


if __name__ == "__main__":
    main()
