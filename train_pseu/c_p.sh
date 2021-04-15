#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/c_p/ --model-path /result/rot_sup/resnet50/p_c/stage2/best_model.ckpt --trg-domain Product



CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/c_p/ --src-domain Clipart --trg-domain Product --stage 2

