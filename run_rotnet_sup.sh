#!/bin/bash


# CUDA_VISIBLE_DEIVCES=0 python new_train.py --save-dir rot_sup_resnet/r_a --src-domain RealWorld --trg-domain Art
 
# CUDA_VISIBLE_DEIVCES=0 python new_train.py --save-dir rot_sup_resnet/r_p --src-domain RealWorld --trg-domain Product

# CUDA_VISIBLE_DEIVCES=0 python new_train.py --save-dir rot_sup_resnet/a_r --src-domain Art --trg-domain RealWorld

# CUDA_VISIBLE_DEIVCES=0 python new_train.py --save-dir rot_sup_resnet/p_r --src-domain Product --trg-domain RealWorld


CUDA_VISIBLE_DEIVCES=0 python new_train.py --save-dir rot_sup_resnet/c_a --src-domain Clipart --trg-domain Art
CUDA_VISIBLE_DEIVCES=0 python new_train.py --save-dir rot_sup_resnet/c_p --src-domain Clipart --trg-domain Product

CUDA_VISIBLE_DEIVCES=0 python new_train.py --save-dir rot_sup_resnet/a_c --src-domain Art --trg-domain Clipart
CUDA_VISIBLE_DEIVCES=0 python new_train.py --save-dir rot_sup_resnet/a_p --src-domain Art --trg-domain Product

CUDA_VISIBLE_DEIVCES=0 python new_train.py --save-dir rot_sup_resnet/p_a --src-domain Product --trg-domain Art
CUDA_VISIBLE_DEIVCES=0 python new_train.py --save-dir rot_sup_resnet/p_c --src-domain Product --trg-domain Clipart
