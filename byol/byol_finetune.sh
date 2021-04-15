#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python byol_finetune.py --data-root /data/OfficeHomeDataset_10072016 --save-root /result --domain RealWorld --save-dir byol_finetune/freeze_r --freeze

CUDA_VISIBLE_DEVICES=1 python byol_finetune.py --data-root /data/OfficeHomeDataset_10072016 --save-root /result --domain Art --save-dir byol_finetune/freeze_a --freeze

CUDA_VISIBLE_DEVICES=1 python byol_finetune.py --data-root /data/OfficeHomeDataset_10072016 --save-root /result --domain Clipart --save-dir byol_finetune/freeze_c --freeze

CUDA_VISIBLE_DEVICES=1 python byol_finetune.py --data-root /data/OfficeHomeDataset_10072016 --save-root /result --domain Product --save-dir byol_finetune/freeze_p --freeze


CUDA_VISIBLE_DEVICES=1 python byol_finetune.py --data-root /data/OfficeHomeDataset_10072016 --save-root /result --domain RealWorld --save-dir byol_finetune/r

CUDA_VISIBLE_DEVICES=1 python byol_finetune.py --data-root /data/OfficeHomeDataset_10072016 --save-root /result --domain Art --save-dir byol_finetune/a

CUDA_VISIBLE_DEVICES=1 python byol_finetune.py --data-root /data/OfficeHomeDataset_10072016 --save-root /result --domain Clipart --save-dir byol_finetune/c

CUDA_VISIBLE_DEVICES=1 python byol_finetune.py --data-root /data/OfficeHomeDataset_10072016 --save-root /result --domain Product --save-dir byol_finetune/p

