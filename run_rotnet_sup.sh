#!/bin/bash


CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --trg-domain Clipart --src-domain RealWorld --save-dir rot_sup/resnet50/r_c --data-root
/data/OfficeHomeDataset_10072016 --save-root /result
CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --trg-domain Art --src-domain RealWorld --save-dir rot_sup/resnet50/r_a --data-root
/data/OfficeHomeDataset_10072016 --save-root /result
CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --trg-domain Product --src-domain RealWorld --save-dir rot_sup/resnet50/r_p --data-root
/data/OfficeHomeDataset_10072016 --save-root /result
CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Clipart --trg-domain RealWorld --save-dir rot_sup/resnet50/c_r --data-root
/data/OfficeHomeDataset_10072016 --save-root /result
CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Clipart --trg-domain Art --save-dir rot_sup/resnet50/c_a --data-root
/data/OfficeHomeDataset_10072016 --save-root /result
CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Clipart --trg-domain Product --save-dir rot_sup/resnet50/c_p --data-root
/data/OfficeHomeDataset_10072016 --save-root /result
CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Art --trg-domain Clipart --save-dir rot_sup/resnet50/a_c --data-root
/data/OfficeHomeDataset_10072016 --save-root /result
CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Art --trg-domain RealWorld --save-dir rot_sup/resnet50/a_r --data-root
/data/OfficeHomeDataset_10072016 --save-root /result
CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Art --trg-domain Product --save-dir rot_sup/resnet50/a_p --data-root
/data/OfficeHomeDataset_10072016 --save-root /result
CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Product --trg-domain Art --save-dir rot_sup/resnet50/p_a --data-root
/data/OfficeHomeDataset_10072016 --save-root /result
CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Product --trg-domain Clipart --save-dir rot_sup/resnet50/p_c --data-root
/data/OfficeHomeDataset_10072016 --save-root /result
CUDA_VISIBLE_DEVICES=2 python four_domain_train_on_rot.py --src-domain Product --trg-domain RealWorld --save-dir rot_sup/resnet50/p_r --data-root
/data/OfficeHomeDataset_10072016 --save-root /result

