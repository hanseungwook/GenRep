#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python main_linear.py --learning_rate 0.3 \
	--ckpt /disk_d/han/GenRep/results/gan_cifar100/SimCLR/gan_models/SimCLR_gan_my_gauss_resnet50_ncontrast.1_ratiodata.1.0_lr_0.03_decay_0.0001_bsz_256_temp_0.5_trial_0_cosine_stylegan_cifar100_tr2.0_gauss1_std0.2_NS500_NN1/last.pth \
	--num_workers 16 --dataset cifar100 --data_folder /disk_d/han/data/ >> log_test_simclr_cifar100.txt &