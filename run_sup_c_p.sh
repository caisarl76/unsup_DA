#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python new_train_multi.py --save-dir result/c_sup --trg-domain Clipart

CUDA_VISIBLE_DEVICES=0 python new_train_multi.py --save-dir result/p_sup --trg-domain Product
