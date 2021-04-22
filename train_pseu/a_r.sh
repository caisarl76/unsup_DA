#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/a_r/ \
	--src-domain Art  --trg-domain RealWorld #--train-teacher

CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result \
	--save-dir pseudo/a_r/student --src-domain Art --trg-domain RealWorld --stage 2
