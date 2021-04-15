#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/a_p/ --model-path /result/rot_sup/resnet50/p_a/stage2/best_model.ckpt --trg-domain Product



CUDA_VISIBLE_DEVICES=1 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/a_p/ --src-domain Art --trg-domain Product --stage 2

