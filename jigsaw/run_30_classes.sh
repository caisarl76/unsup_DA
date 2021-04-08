#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --batch 40 --cores 5 --save-dir result/class30_RealWorld --domain RealWorld

CUDA_VISIBLE_DEVICES=0 python train.py --batch 40 --cores 5 --save-dir result/class30_Clipart --domain Clipart

CUDA_VISIBLE_DEVICES=0 python train.py --batch 40 --cores 5 --save-dir result/class30_Art --domain Art

CUDA_VISIBLE_DEVICES=0 python train.py --batch 40 --cores 5 --save-dir result/class30_Product --domain Product
