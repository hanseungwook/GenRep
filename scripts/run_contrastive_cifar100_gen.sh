#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python3 main_unified.py --method SimCLR --cosine \
	--dataset /disk_d/han/data/cifar100_generated/stylegan_cifar100_tr2.0_gauss1_std0.2_NS500_NN1 --walk_method my_gauss \
    --batch_size 256 --learning_rate 0.03 --temp 0.5 --num_workers 64 \
	--cache_folder /disk_d/han/GenRep/results/gan_cifar100/ >> log_train_simclr_cifar100.txt &