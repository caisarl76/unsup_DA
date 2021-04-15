#!/bin/bash 
echo exp: /result/rot_sup/resnet50/p_a 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/p_a/stage2/best_model.ckpt --domain Product
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/p_a/stage2/best_model.ckpt --domain Art
echo exp: /result/rot_sup/resnet50/p_c 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/p_c/stage2/best_model.ckpt --domain Product
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/p_c/stage2/best_model.ckpt --domain Clipart
echo exp: /result/rot_sup/resnet50/a_r 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/a_r/stage2/best_model.ckpt --domain Art
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/a_r/stage2/best_model.ckpt --domain RealWorld
echo exp: /result/rot_sup/resnet50/c_p 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/c_p/stage2/best_model.ckpt --domain Clipart
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/c_p/stage2/best_model.ckpt --domain Product
echo exp: /result/rot_sup/resnet50/r_c 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/r_c/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/r_c/stage2/best_model.ckpt --domain Clipart
echo exp: /result/rot_sup/resnet50/p_r 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/p_r/stage2/best_model.ckpt --domain Product
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/p_r/stage2/best_model.ckpt --domain RealWorld
echo exp: /result/rot_sup/resnet50/a_c 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/a_c/stage2/best_model.ckpt --domain Art
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/a_c/stage2/best_model.ckpt --domain Clipart
echo exp: /result/rot_sup/resnet50/c_r 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/c_r/stage2/best_model.ckpt --domain Clipart
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/c_r/stage2/best_model.ckpt --domain RealWorld
echo exp: /result/rot_sup/resnet50/r_p 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/r_p/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/r_p/stage2/best_model.ckpt --domain Product
echo exp: /result/rot_sup/resnet50/a_p 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/a_p/stage2/best_model.ckpt --domain Art
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/a_p/stage2/best_model.ckpt --domain Product
echo exp: /result/rot_sup/resnet50/r_a 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/r_a/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/r_a/stage2/best_model.ckpt --domain Art
echo exp: /result/rot_sup/resnet50/c_a 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/c_a/stage2/best_model.ckpt --domain Clipart
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/c_a/stage2/best_model.ckpt --domain Art
echo exp: /result/rot_ssl/byol/p_a 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/p_a/stage2/best_model.ckpt --domain Product
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/p_a/stage2/best_model.ckpt --domain Art
echo exp: /result/rot_ssl/byol/c_p 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/c_p/stage2/best_model.ckpt --domain Clipart
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/c_p/stage2/best_model.ckpt --domain Product
echo exp: /result/rot_ssl/byol/r_c 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/r_c/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/r_c/stage2/best_model.ckpt --domain Clipart
echo exp: /result/rot_ssl/byol/a_c 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/a_c/stage2/best_model.ckpt --domain Art
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/a_c/stage2/best_model.ckpt --domain Clipart
echo exp: /result/rot_ssl/byol/c_r 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/c_r/stage2/best_model.ckpt --domain Clipart
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/c_r/stage2/best_model.ckpt --domain RealWorld
echo exp: /result/rot_ssl/byol/r_p 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/r_p/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/r_p/stage2/best_model.ckpt --domain Product
echo exp: /result/rot_ssl/byol/r_a 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/r_a/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/r_a/stage2/best_model.ckpt --domain Art
echo exp: /result/rot_ssl/byol/c_a 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/c_a/stage2/best_model.ckpt --domain Clipart
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/byol/c_a/stage2/best_model.ckpt --domain Art
echo exp: /result/rot_ssl/resnet50/p_a 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/p_a/stage2/best_model.ckpt --domain Product
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/p_a/stage2/best_model.ckpt --domain Art
echo exp: /result/rot_ssl/resnet50/p_c 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/p_c/stage2/best_model.ckpt --domain Product
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/p_c/stage2/best_model.ckpt --domain Clipart
echo exp: /result/rot_ssl/resnet50/a_r 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/a_r/stage2/best_model.ckpt --domain Art
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/a_r/stage2/best_model.ckpt --domain RealWorld
echo exp: /result/rot_ssl/resnet50/r_c 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/r_c/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/r_c/stage2/best_model.ckpt --domain Clipart
echo exp: /result/rot_ssl/resnet50/a_c 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/a_c/stage2/best_model.ckpt --domain Art
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/a_c/stage2/best_model.ckpt --domain Clipart
echo exp: /result/rot_ssl/resnet50/c_r 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/c_r/stage2/best_model.ckpt --domain Clipart
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/c_r/stage2/best_model.ckpt --domain RealWorld
echo exp: /result/rot_ssl/resnet50/r_p 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/r_p/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/r_p/stage2/best_model.ckpt --domain Product
echo exp: /result/rot_ssl/resnet50/a_p 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/a_p/stage2/best_model.ckpt --domain Art
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/a_p/stage2/best_model.ckpt --domain Product
echo exp: /result/rot_ssl/resnet50/r_a 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/r_a/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl/resnet50/r_a/stage2/best_model.ckpt --domain Art
echo exp: /result/rot_ssl_byol/resnet50/r_c 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl_byol/resnet50/r_c/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_ssl_byol/resnet50/r_c/stage2/best_model.ckpt --domain Clipart
echo exp: /result/rot_sup/resnet50/p_a 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/p_a/stage2/best_model.ckpt --domain Product
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/p_a/stage2/best_model.ckpt --domain Art
echo exp: /result/rot_sup/resnet50/p_c 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/p_c/stage2/best_model.ckpt --domain Product
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/p_c/stage2/best_model.ckpt --domain Clipart
echo exp: /result/rot_sup/resnet50/a_r 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/a_r/stage2/best_model.ckpt --domain Art
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/a_r/stage2/best_model.ckpt --domain RealWorld
echo exp: /result/rot_sup/resnet50/c_p 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/c_p/stage2/best_model.ckpt --domain Clipart
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/c_p/stage2/best_model.ckpt --domain Product
echo exp: /result/rot_sup/resnet50/r_c 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/r_c/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/r_c/stage2/best_model.ckpt --domain Clipart
echo exp: /result/rot_sup/resnet50/p_r 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/p_r/stage2/best_model.ckpt --domain Product
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/p_r/stage2/best_model.ckpt --domain RealWorld
echo exp: /result/rot_sup/resnet50/a_c 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/a_c/stage2/best_model.ckpt --domain Art
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/a_c/stage2/best_model.ckpt --domain Clipart
echo exp: /result/rot_sup/resnet50/c_r 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/c_r/stage2/best_model.ckpt --domain Clipart
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/c_r/stage2/best_model.ckpt --domain RealWorld
echo exp: /result/rot_sup/resnet50/r_p 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/r_p/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/r_p/stage2/best_model.ckpt --domain Product
echo exp: /result/rot_sup/resnet50/a_p 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/a_p/stage2/best_model.ckpt --domain Art
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/a_p/stage2/best_model.ckpt --domain Product
echo exp: /result/rot_sup/resnet50/r_a 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/r_a/stage2/best_model.ckpt --domain RealWorld
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/r_a/stage2/best_model.ckpt --domain Art
echo exp: /result/rot_sup/resnet50/c_a 
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/c_a/stage2/best_model.ckpt --domain Clipart
CUDA_VISIBLE_DEVICES=4 python test.py --data-root /data/OfficeHomeDataset_10072016 --model-path /result/rot_sup/resnet50/c_a/stage2/best_model.ckpt --domain Art
