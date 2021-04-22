#!/bin/bash


gpu=${1}
src=${2}
trg=${3}
save_dir=${4}
teacher_root=${5}

function stage1(){
	echo gpu ${1} 
	echo src ${2}  trg ${3}
	echo save_dir ${4}
	
	CUDA_VISIBLE_DEVICES=${1} python pseudo_train.py --data-root /data/OfficeHomeDataset_10072016/ \
		--save-root ${4} --save-dir ${2}_${3} \
		--teacher-root ${5} \
		--src-domain ${2} --trg-domain ${3}
}

function stage2(){
	echo gpu ${1} 
	echo src ${2}  trg ${3}
	echo save_dir ${4}
	CUDA_VISIBLE_DEVICES=${1} python four_domain_train_on_rot.py --data-root /data/OfficeHomeDataset_10072016/ \
		--save-root ${4} \
		--save-dir ${2}_${3}/student --stage 2 --src-domain ${2} --trg-domain ${3}
}

stage1 ${1} ${2} ${3} ${4} ${5}

# stage2 ${1} ${2} ${3} ${4} ${5}
