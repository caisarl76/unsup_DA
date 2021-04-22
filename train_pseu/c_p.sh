#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/c_p/ \
	--src-domain Clipart  --trg-domain Product

CUDA_VISIBLE_DEVICES=5  python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result \
	        --save-dir pseudo/c_p/student --src-domain Clipart --trg-domain Product --stage 2

