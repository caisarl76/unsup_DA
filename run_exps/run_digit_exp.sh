#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python svhn_to_mnist.py --save-dir s_m_sup --src-domain svhn --trg-domain mnist
CUDA_VISIBLE_DEVICES=0 python svhn_to_mnist.py --save-dir m_s_sup --src-domain mnist --trg-domain svhn

CUDA_VISIBLE_DEVICES=0 python mnist_to_svhn_ssl.py --ssl --save-dir m_s_ssl --src-domain mnist --trg-domain svhn
