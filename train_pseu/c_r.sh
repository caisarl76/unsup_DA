#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/c_r/ \
	 --src-domain Clipart  --trg-domain RealWorld #--train-teacher

CUDA_VISIBLE_DEVICES=1 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result \
	        --save-dir pseudo/c_r/student --src-domain Clipart --trg-domain RealWorld --stage 2

