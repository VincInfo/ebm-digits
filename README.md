# Run
uv run ebm-sample --checkpoint "./runs/ebm_checkpoint.pt" --label 3 --sample-steps 150 --num-snapshots 8

# Train
uv run ebm-train --img-size 28 --epochs 70 --ld-steps 20 --batch-size 128
If training is unstable:
uv run ebm-train --img-size 28 --epochs 70 --gen-weight 0.5 --ld-steps 10

uv run ebm-train --energy-net cnn --img-size 28 --epochs 70
