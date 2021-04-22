#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/c_a/ \
	--train-teacher  --src-domain Clipart  --trg-domain Art





CUDA_VISIBLE_DEVICES=1 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result \
	        --save-dir pseudo/c_a/student --src-domain Clipart --trg-domain Art --stage 2

