#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <ckpt_path> <out_dir> [gpu_id]" >&2
  exit 1
fi

CKPT=$1
OUTDIR=$2
GPU=${3:-0}

python tools/compute_imp_score.py --ckpt "$CKPT" --out "$CKPT/imp_score.npz" --subsample 2 --gpu "$GPU"

python tools/prune_and_save.py --ckpt "$CKPT" --imp "$CKPT/imp_score.npz" \
       --prune_ratio 0.4 --out_dir "$OUTDIR"

python train.py --model_path "$OUTDIR" --resume "$OUTDIR" --finetune \
       --finetune_iters 5000 --freeze_iters 2000 \
       --mlp_color_lr 1e-4 --mlp_opacity_lr 1e-4 \
       --pos_lr_after 1e-4 --offset_lr_after 5e-5 --gpu "$GPU"

python tools/vq_anchor_feat.py --ckpt "$OUTDIR" --imp "$CKPT/imp_score.npz" \
       --quant_ratio 0.6 --codebook_size 8192 --ema_decay 0.9 --commit 1.0 \
       --phase1_steps 2000 --phase2_steps 2000 --out "$OUTDIR/vq_anchor_feat.npz"
