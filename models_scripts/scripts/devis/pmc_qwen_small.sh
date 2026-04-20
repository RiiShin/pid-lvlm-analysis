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


CUDA_VISIBLE_DEVICES=4 nohup python -u qwen2_small_devi.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pmc/pmc_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pmc/pmc_images \
    --vector_save_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pmc \
    --model_size 2 \
    --data_mode train \
    >/work/gw42/w42010/msra/all_new_type/results/devis_vector/pmc/logs/qwen2_2b.logs 2>&1 &


CUDA_VISIBLE_DEVICES=5 nohup python -u qwen2_small_devi.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pmc/pmc_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pmc/pmc_images \
    --vector_save_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pmc \
    --model_size 7 \
    --data_mode train \
    >/work/gw42/w42010/msra/all_new_type/results/devis_vector/pmc/logs/qwen2_7b.logs 2>&1 &


CUDA_VISIBLE_DEVICES=6 nohup python -u qwen25_small_devi.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pmc/pmc_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pmc/pmc_images \
    --vector_save_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pmc \
    --model_size 3 \
    --data_mode train \
    >/work/gw42/w42010/msra/all_new_type/results/devis_vector/pmc/logs/qwen25_3b.logs 2>&1 &


CUDA_VISIBLE_DEVICES=7 nohup python -u qwen25_small_devi.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pmc/pmc_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pmc/pmc_images \
    --vector_save_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pmc \
    --model_size 7 \
    --data_mode train \
    >/work/gw42/w42010/msra/all_new_type/results/devis_vector/pmc/logs/qwen25_7b.logs 2>&1 &


wait
echo "All jobs finished."