#!/bin/bash

# CUDA_VISIBLE_DEVICES=5 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/a_r/ --model-path /result/rot_sup/resnet50/r_a/stage2/best_model.ckpt --trg-domain RealWorld

# CUDA_VISIBLE_DEVICES=5 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/a_r/ --src-domain Art --trg-domain RealWorld --stage 2  



# CUDA_VISIBLE_DEVICES=5 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/c_r/ --model-path /result/rot_sup/resnet50/r_c/stage2/best_model.ckpt --trg-domain RealWorld

# CUDA_VISIBLE_DEVICES=5 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/c_r/ --src-domain Clipart --trg-domain RealWorld --stage 2


CUDA_VISIBLE_DEVICES=5 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/p_r/ --model-path /result/rot_sup/resnet50/r_p/stage2/best_model.ckpt --trg-domain Product

CUDA_VISIBLE_DEVICES=5 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/p_r/ --src-domain Product --trg-domain RealWorld --stage 2

