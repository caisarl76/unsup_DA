#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result --save-dir pseudo/r_p \
	--train-teacher --src-domain RealWorld --trg-domain Product

CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ --save-root /result \
	--save-dir pseudo/r_p/student --stage 2 --src-domain RealWorld --trg-domain Product
