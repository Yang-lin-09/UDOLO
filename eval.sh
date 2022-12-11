#!/bin/bash

python eval.py --dataset scannet --checkpoint_path log_scannet/104_checkpoint.tar --dump_dir eval_scannet --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --batch_size 1 --faster_eval --feedback --vis_disable