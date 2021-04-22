#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/p_c/ \
	--src-domain Product  --trg-domain Clipart

CUDA_VISIBLE_DEVICES=4 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result \
	        --save-dir pseudo/p_c/student --src-domain Product --trg-domain Clipart --stage 2

