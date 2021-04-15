#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Art --trg-domain Clipart --save-dir rot_ssl/resnet50/a_c --ssl --data-root /data/OfficeHomeDataset_10072016 --save-root /result


CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Art --trg-domain RealWorld --save-dir rot_ssl/resnet50/a_r --ssl --data-root /data/OfficeHomeDataset_10072016 --save-root /result 


CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Art --trg-domain Product --save-dir rot_ssl/resnet50/a_p --ssl  --data-root /data/OfficeHomeDataset_10072016 --save-root /result



CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Product --trg-domain Art --save-dir rot_ssl/resnet50/p_a --ssl --data-root /data/OfficeHomeDataset_10072016 --save-root /result



CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Product --trg-domain Clipart --save-dir rot_ssl/resnet50/p_c --ssl --data-root /data/OfficeHomeDataset_10072016 --save-root /result



CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Product --trg-domain RealWorld --save-dir rot_ssl/resnet50/p_r --ssl --data-root /data/OfficeHomeDataset_10072016 --save-root /result
