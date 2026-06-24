"""CLI wrapper: `uv run python scripts/train.py` (same as `ebm-train`)."""

from ebm_digits.train import train

if __name__ == "__main__":
    train()
