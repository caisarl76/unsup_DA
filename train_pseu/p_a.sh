#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/p_a/ \
	--src-domain Product  --trg-domain Art

CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result \
	        --save-dir pseudo/p_a/student --src-domain Product --trg-domain Art --stage 2

