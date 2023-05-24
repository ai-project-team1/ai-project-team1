#!/usr/bin/env bashs

for alpha in 0.0 2.0 3.0 4.0 5.0; do
  for seed in 42; do
    python classifier.py \
      --model_checkpoint_path checkpoints_cl_1 \
      --tweets_per_row 5 \
      --batch_size 16 \
      --epochs 5 \
      --seed $seed \
      --alpha $alpha
  done
done