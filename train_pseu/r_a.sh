#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/r_a/ --model-path /result/rot_sup/resnet50/a_r/stage2/best_model.ckpt --trg-domain Art

CUDA_VISIBLE_DEVICES=4 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/r_a/ --src-domain RealWorld --trg-domain Art --stage 2 
