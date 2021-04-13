#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --trg-domain Clipart --src-domain RealWorld --save-dir rot_ssl/resnet50/r_c --ssl
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --trg-domain Art --src-domain RealWorld --save-dir rot_ssl/resnet50/r_a --ssl
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --trg-domain Product --src-domain RealWorld --save-dir rot_ssl/resnet50/r_p --ssl
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Clipart --trg-domain RealWorld --save-dir rot_ssl/resnet50/c_r --ssl
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Clipart --trg-domain Art --save-dir rot_ssl/resnet50/c_a --ssl
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Clipart --trg-domain Product --save-dir rot_ssl/resnet50/c_p --ssl
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Art --trg-domain Clipart --save-dir rot_ssl/resnet50/a_c --ssl
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Art --trg-domain RealWorld --save-dir rot_ssl/resnet50/a_r --ssl
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Art --trg-domain Product --save-dir rot_ssl/resnet50/a_p --ssl
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Product --trg-domain Art --save-dir rot_ssl/resnet50/p_a --ssl
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Product --trg-domain Clipart --save-dir rot_ssl/resnet50/p_c --ssl
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Product --trg-domain RealWorld --save-dir rot_ssl/resnet50/p_r --ssl
