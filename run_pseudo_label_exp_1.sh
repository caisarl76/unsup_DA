#!/bin/bash

# CUDA_VISIBLE_DEVICES=4 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/r_a/ --model-path /result/rot_sup/resnet50/a_r/stage2/best_model.ckpt --trg-domain Art

# CUDA_VISIBLE_DEVICES=4 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/r_a/ --src-domain RealWorld --trg-domain Art --stage 2  



CUDA_VISIBLE_DEVICES=0 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/c_r/stage1 --model-path /result/rot_sup/resnet50/r_c/stage2/best_model.ckpt --trg-domain Clipart 

CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/r_c/ --src-domain RealWorld --trg-domain Clipart --stage 2 


CUDA_VISIBLE_DEVICES=0 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/p_r/stage1 --model-path /result/rot_sup/resnet50/r_p/stage2/best_model.ckpt --trg-domain Product

CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/r_p/ --src-domain RealWorld --trg-domain Product --stage 2




