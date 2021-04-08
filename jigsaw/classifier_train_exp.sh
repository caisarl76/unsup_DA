#!/bin/bash

# domain_list=("RealWorld" "Clipart" "Product" "Art")
domain_list=("Product" "Art")
for domain in "${domain_list[@]}";
do
	echo "run exp $domain"
	CUDA_VISIBLE_DEVICES=0 python classifier_train.py --domain $domain --save-dir result/finetune_classifier/$domain
done
