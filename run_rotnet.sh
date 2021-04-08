#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python rot_train.py --trg-domain Clipart --src-domain Product --save-dir rot_ssl/p_c

CUDA_VISIBLE_DEVICES=0 python rot_train.py --trg-domain Art --src-domain Product --save-dir rot_ssl/p_a

CUDA_VISIBLE_DEVICES=0 python rot_train.py --trg-domain Clipart --src-domain Art --save-dir rot_ssl/a_c

CUDA_VISIBLE_DEVICES=0 python rot_train.py --trg-domain Product --src-domain Art --save-dir rot_ssl/a_p

CUDA_VISIBLE_DEVICES=0 python rot_train.py --trg-domain Product --src-domain Clipart --save-dir rot_ssl/c_p

CUDA_VISIBLE_DEVICES=0 python rot_train.py --trg-domain Art --src-domain Clipart --save-dir rot_ssl/c_a





# CUDA_VISIBLE_DEVICES=0 python rot_train.py --trg-domain Clipart --save-dir rot_ssl/r_c

# CUDA_VISIBLE_DEVICES=0 python rot_train.py --trg-domain Art --save-dir rot_ssl/r_a

# CUDA_VISIBLE_DEVICES=0 python rot_train.py --trg-domain Product --save-dir rot_ssl/r_p

# CUDA_VISIBLE_DEVICES=0 python rot_train.py --trg-domain RealWorld --src-domain Clipart --save-dir rot_ssl/c_r

# CUDA_VISIBLE_DEVICES=0 python rot_train.py --trg-domain RealWorld --src-domain Art --save-dir rot_ssl/a_r

# CUDA_VISIBLE_DEVICES=0 python rot_train.py --trg-domain RealWorld --src-domain Product --save-dir rot_ssl/p_r

