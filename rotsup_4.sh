#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python four_domain_train_on_rot.py --src-domain Art --trg-domain Clipart --save-dir rot_sup/resnet50/a_c --data-root /data/OfficeHomeDataset_10072016 --save-root /result


CUDA_VISIBLE_DEVICES=4 python four_domain_train_on_rot.py --src-domain Art --trg-domain RealWorld --save-dir rot_sup/resnet50/a_r --data-root /data/OfficeHomeDataset_10072016 --save-root /result 


CUDA_VISIBLE_DEVICES=4 python four_domain_train_on_rot.py --src-domain Art --trg-domain Product --save-dir rot_sup/resnet50/a_p   --data-root /data/OfficeHomeDataset_10072016 --save-root /result



CUDA_VISIBLE_DEVICES=4 python four_domain_train_on_rot.py --src-domain Product --trg-domain Art --save-dir rot_sup/resnet50/p_a  --data-root /data/OfficeHomeDataset_10072016 --save-root /result



CUDA_VISIBLE_DEVICES=4 python four_domain_train_on_rot.py --src-domain Product --trg-domain Clipart --save-dir rot_sup/resnet50/p_c  --data-root /data/OfficeHomeDataset_10072016 --save-root /result



CUDA_VISIBLE_DEVICES=4 python four_domain_train_on_rot.py --src-domain Product --trg-domain RealWorld --save-dir rot_sup/resnet50/p_r  --data-root /data/OfficeHomeDataset_10072016 --save-root /result
