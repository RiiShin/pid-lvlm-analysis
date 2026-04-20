#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -g gw42
#PJM -j
#------- Program execution -------#


### You must check num_choices in the script before running it!!!


source /work/gw42/w42010/anaconda3/bin/activate l_embed_new


cd /work/gw42/w42010/msra/all_new_type/models_scripts/general_models/devis


CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u qwen25_72b_devi.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/mmbench/mmbench_en_ \
    --vector_save_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/wis_comp_test \
    --model_size 72 \
    --data_mode train \
    >/work/gw42/w42010/msra/all_new_type/results/devis_vector/wis_comp_test/logs/qwen25_72b.logs 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u qwen2_72b_devi.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/mmbench/mmbench_en_ \
    --vector_save_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/wis_comp_test \
    --model_size 72 \
    --data_mode train \
    >/work/gw42/w42010/msra/all_new_type/results/devis_vector/wis_comp_test/logs/qwen2_72b.logs 2>&1 &


wait
echo "All jobs finished."