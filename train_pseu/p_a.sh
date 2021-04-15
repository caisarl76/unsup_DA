#!/bin/bash





CUDA_VISIBLE_DEVICES=3 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/p_a/ --model-path /result/rot_sup/resnet50/a_p/stage2/best_model.ckpt --trg-domain Art

CUDA_VISIBLE_DEVICES=3 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/p_a/ --src-domain Product --trg-domain Art --stage 2

