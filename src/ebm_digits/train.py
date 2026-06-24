"""Train the joint EBM on MNIST (default: 8x8, MLP, book-style loss, 70 epochs)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from ebm_digits.data import get_mnist_dataloaders
from ebm_digits.model import EBM, build_energy_net, energy_cfg_from_args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EBM on MNIST")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("runs"))
    p.add_argument("--epochs", type=int, default=10, help="Textbook-style long run (e.g. Fig. 7.2)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument(
        "--gen-weight",
        type=float,
        default=1.0,
        help="Scale for L_gen (1.0 = L_clf + L_gen as in many textbook JEM plots)",
    )
    p.add_argument(
        "--energy-net",
        type=str,
        default="mlp",
        choices=("mlp", "cnn"),
        help="Energy network: mlp (flat pixels) or cnn (spatial; better at 28x28)",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--img-size",
        type=int,
        default=8,
        help="MNIST resized to img_size x img_size (8 matches many small demos)",
    )
    p.add_argument(
        "--ld-steps",
        type=int,
        default=20,
        help="Langevin steps when drawing negatives inside gen_loss (book uses ~20 at sampling)",
    )
    p.add_argument("--langevin-alpha", type=float, default=1e-2, help="Langevin step size")
    p.add_argument("--langevin-sigma", type=float, default=1e-2, help="Langevin noise scale")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def train() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_mnist_dataloaders(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    hidden_dims = (256, 256)
    energy_net = build_energy_net(
        args.energy_net,
        args.img_size,
        hidden_dims=hidden_dims,
    ).to(device)
    image_shape = (1, args.img_size, args.img_size)
    ebm = EBM(
        energy_net,
        alpha=args.langevin_alpha,
        sigma=args.langevin_sigma,
        ld_steps=args.ld_steps,
        image_shape=image_shape,
        gen_weight=args.gen_weight,
    ).to(device)

    optimizer = torch.optim.Adam(ebm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        ebm.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = ebm(images, labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ebm.parameters(), args.grad_clip)
            optimizer.step()

            running += loss.item() * images.size(0)
            n += images.size(0)
            pbar.set_postfix(batch_loss=f"{loss.item():.4f}")

        train_loss = running / max(n, 1)

        ebm.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                pred = ebm.classify(images)
                correct += (pred == labels).sum().item()
                total += labels.numel()

        acc = correct / max(total, 1)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} test_acc={acc:.4f}")

    ckpt_path = args.output_dir / "ebm_checkpoint.pt"
    torch.save(
        {
            "energy_net": energy_net.state_dict(),
            "energy_cfg": energy_cfg_from_args(
                args.energy_net, args.img_size, hidden_dims=hidden_dims
            ),
            "ebm_config": {
                "alpha": ebm.alpha,
                "sigma": ebm.sigma,
                "ld_steps": ebm.ld_steps,
                "image_shape": ebm.image_shape,
                "gen_weight": ebm.gen_weight,
                "img_size": args.img_size,
                "energy_net": args.energy_net,
            },
        },
        ckpt_path,
    )
    print(f"wrote {ckpt_path} (energy-net={args.energy_net}, img-size={args.img_size})")


if __name__ == "__main__":
    train()
