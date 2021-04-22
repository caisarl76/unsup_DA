#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/r_a/ \
	--train-teacher  --src-domain RealWorld --trg-domain Art

CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result \
	--save-dir pseudo/r_a/student --stage 2 --src-domain RealWorld --trg-domain Art
