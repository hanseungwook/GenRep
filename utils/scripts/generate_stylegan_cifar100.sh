#!/bin/bash

python3 generate_dataset_stylegan.py \
    --out_dir /disk_d/han/data/cifar100_generated/ \
    --checkpoint_path /disk_d/han/data/stylegan_checkpoints/220000.pt \
    --partition train \
    --truncation 2.0 \
    --batch_size 32 \
    --num_imgs 500 \
    --dataset_type cifar100 \
    --num_neighbors 1 \
    --std 0.2
