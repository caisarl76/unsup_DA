#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --trg-domain Clipart --src-domain RealWorld --save-dir rot_sup/resnet50/r_c
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --trg-domain Art --src-domain RealWorld --save-dir rot_sup/resnet50/r_a
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --trg-domain Product --src-domain RealWorld --save-dir rot_sup/resnet50/r_p
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Clipart --trg-domain RealWorld --save-dir rot_sup/resnet50/c_r
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Clipart --trg-domain Art --save-dir rot_sup/resnet50/c_a
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Clipart --trg-domain Product --save-dir rot_sup/resnet50/c_p
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Art --trg-domain Clipart --save-dir rot_sup/resnet50/a_c
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Art --trg-domain RealWorld --save-dir rot_sup/resnet50/a_r
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Art --trg-domain Product --save-dir rot_sup/resnet50/a_p
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Product --trg-domain Art --save-dir rot_sup/resnet50/p_a
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Product --trg-domain Clipart --save-dir rot_sup/resnet50/p_c
CUDA_VISIBLE_DEVICES=0 python four_domain_train_on_rot.py --src-domain Product --trg-domain RealWorld --save-dir rot_sup/resnet50/p_r

