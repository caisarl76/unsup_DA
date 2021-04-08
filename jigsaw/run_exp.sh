#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --batch 128 --cores 5 --checkpoint result/try3_RealWorld

CUDA_VISIBLE_DEVICES=0 python train.py --batch 128 --cores 5 --checkpoint result/try3_Clipart --domain Clipart

CUDA_VISIBLE_DEVICES=0 python train.py --batch 128 --cores 5 --checkpoint result/try3_Art --domain Art

CUDA_VISIBLE_DEVICES=0 python train.py --batch 128 --cores 5 --checkpoint result/try3_Product --domain Product
