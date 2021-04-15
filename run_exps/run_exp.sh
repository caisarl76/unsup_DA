#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python new_train_multi.py --save-dir result/testing/c_a/ --trg-domain Clipart Product --iters 20000 550 --ssl 

CUDA_VISIBLE_DEVICES=0 python new_train_multi.py --save-dir result/testing/c --trg-domain Clipart --iters 10000 550 --ssl
CUDA_VISIBLE_DEVICES=0 python new_train_multi.py --save-dir result/testing/a --trg-domain Product --iters 10000 550 --ssl
