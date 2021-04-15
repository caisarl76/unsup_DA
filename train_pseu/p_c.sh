#!/bin/bash





CUDA_VISIBLE_DEVICES=3 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/p_c/ --model-path /result/rot_sup/resnet50/c_p/stage2/best_model.ckpt --trg-domain Clipart


CUDA_VISIBLE_DEVICES=3 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/p_c/ --src-domain Product --trg-domain Clipart --stage 2

