#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python four_domain_train_on_rot.py --src-domain Clipart --trg-domain Art  --save-dir rot_ssl/byol/c_a --ssl --byol --data-root /data/OfficeHomeDataset_10072016 --save-root /result


CUDA_VISIBLE_DEVICES=3 python four_domain_train_on_rot.py --src-domain Clipart --trg-domain Product  --save-dir rot_ssl/byol/c_p --ssl --byol --data-root /data/OfficeHomeDataset_10072016 --save-root /result


CUDA_VISIBLE_DEVICES=3 python four_domain_train_on_rot.py --src-domain Clipart --trg-domain RealWorld  --save-dir rot_ssl/byol/c_r --ssl --byol --data-root /data/OfficeHomeDataset_10072016 --save-root /result


CUDA_VISIBLE_DEVICES=3 python four_domain_train_on_rot.py --src-domain Product  --trg-domain Art --save-dir rot_ssl/byol/p_a --ssl --byol --data-root /data/OfficeHomeDataset_10072016 --save-root /result

CUDA_VISIBLE_DEVICES=3 python four_domain_train_on_rot.py --src-domain Product  --trg-domain Clipart --save-dir rot_ssl/byol/p_c --ssl --byol --data-root /data/OfficeHomeDataset_10072016 --save-root /result

CUDA_VISIBLE_DEVICES=3 python four_domain_train_on_rot.py --src-domain Product  --trg-domain RealWorld  --save-dir rot_ssl/byol/p_r --ssl --byol --data-root /data/OfficeHomeDataset_10072016 --save-root /result



