#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --trg-domain Clipart --src-domain RealWorld --save-dir rot_ssl/byol/r_c --ssl --byol --data-root /data/OfficeHomeDataset_10072016 --save-root /result


CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain RealWorld  --trg-domain Art  --save-dir rot_ssl/byol/r_a --ssl --byol --data-root /data/OfficeHomeDataset_10072016 --save-root /result


CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain RealWorld  --trg-domain Product  --save-dir rot_ssl/byol/r_p --ssl --byol --data-root /data/OfficeHomeDataset_10072016 --save-root /result


CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Art  --trg-domain Clipart  --save-dir rot_ssl/byol/a_c --ssl --byol --data-root /data/OfficeHomeDataset_10072016 --save-root /result


CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Art  --trg-domain Product  --save-dir rot_ssl/byol/a_p --ssl --byol --data-root /data/OfficeHomeDataset_10072016 --save-root /result


CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Art  --trg-domain RealWorld  --save-dir rot_ssl/byol/a_r --ssl --byol --data-root /data/OfficeHomeDataset_10072016 --save-root /result


