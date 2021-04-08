#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python cheat_train.py --trg_domain Clipart --src_domain RealWorld --save-dir cheat_jigsaw_r_c
# CUDA_VISIBLE_DEVICES=0 python cheat_train.py --trg_domain Art --src_domain RealWorld --save-dir cheat_jigsaw_r_a
# CUDA_VISIBLE_DEVICES=0 python cheat_train.py --trg_domain Product --src_domain RealWorld --save-dir cheat_jigsaw_r_p
# CUDA_VISIBLE_DEVICES=0 python cheat_train.py --trg_domain RealWorld --src_domain Clipart --save-dir cheat_jigsaw_c_r
# CUDA_VISIBLE_DEVICES=0 python cheat_train.py --trg_domain RealWorld --src_domain Product --save-dir cheat_jigsaw_p_r
# CUDA_VISIBLE_DEVICES=0 python cheat_train.py --trg_domain RealWorld --src_domain Art --save-dir cheat_jigsaw_c_a


CUDA_VISIBLE_DEVICES=0 python cheat_train.py --src_domain Product --trg_domain Clipart --save-dir cheat_jigsaw/p_c
CUDA_VISIBLE_DEVICES=0 python cheat_train.py --src_domain Product --trg_domain Art --save-dir cheat_jigsaw/p_a

CUDA_VISIBLE_DEVICES=0 python cheat_train.py --src_domain Art --trg_domain Clipart --save-dir cheat_jigsaw/a_c
CUDA_VISIBLE_DEVICES=0 python cheat_train.py --src_domain Art --trg_domain Product --save-dir cheat_jigsaw/a_p

CUDA_VISIBLE_DEVICES=0 python cheat_train.py --src_domain Clipart --trg_domain Art --save-dir cheat_jigsaw/c_a
CUDA_VISIBLE_DEVICES=0 python cheat_train.py --src_domain Clipart --trg_domain Product --save-dir cheat_jigsaw/c_p

