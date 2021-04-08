#!/bin/bash


# CUDA_VISIBLE_DEVICES=0 python dsbn_on_jigsaw.py --trg-domain Clipart --save-dir jigsaw_ssl_r_c
# CUDA_VISIBLE_DEVICES=0 python dsbn_on_jigsaw.py --trg-domain Product --save-dir jigsaw_ssl_r_p
# CUDA_VISIBLE_DEVICES=0 python dsbn_on_jigsaw.py --trg-domain Art --save-dir jigsaw_ssl_r_a

# CUDA_VISIBLE_DEVICES=0 python dsbn_on_jigsaw.py --src-domain Clipart --trg-domain RealWorld --save-dir jigsaw_ssl_c_r
# CUDA_VISIBLE_DEVICES=0 python dsbn_on_jigsaw.py --src-domain Product --trg-domain RealWorld --save-dir jigsaw_ssl_p_r
# CUDA_VISIBLE_DEVICES=0 python dsbn_on_jigsaw.py --src-domain Art --trg-domain RealWorld --save-dir jigsaw_ssl_a_r



CUDA_VISIBLE_DEVICES=0 python dsbn_on_jigsaw.py --src-domain Clipart --trg-domain Product --save-dir jigsaw_ssl/c_p
CUDA_VISIBLE_DEVICES=0 python dsbn_on_jigsaw.py --src-domain Clipart --trg-domain Art --save-dir jigsaw_ssl/c_a

CUDA_VISIBLE_DEVICES=0 python dsbn_on_jigsaw.py --src-domain Product --trg-domain Art --save-dir jigsaw_ssl/p_a
CUDA_VISIBLE_DEVICES=0 python dsbn_on_jigsaw.py --src-domain Product --trg-domain Clipart --save-dir jigsaw_ssl/p_c

CUDA_VISIBLE_DEVICES=0 python dsbn_on_jigsaw.py --src-domain Art --trg-domain Product --save-dir jigsaw_ssl/a_p
CUDA_VISIBLE_DEVICES=0 python dsbn_on_jigsaw.py --src-domain Art --trg-domain Clipart --save-dir jigsaw_ssl/a_c
