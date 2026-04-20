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


cd /work/gw42/w42010/msra/all_new_type/models_scripts/general_models/embeds


CUDA_VISIBLE_DEVICES=0 nohup python -u qwen2_small_embed.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_images \
    --save_dir /work/gw42/w42010/msra/all_new_type/results/embeds/pope \
    --num_choices 2 \
    --model_qwen_size 2 \
    --data_mode train \
    --devi_vector_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pope \
    >/work/gw42/w42010/msra/all_new_type/results/embeds/pope/logs/qwen2_2b_train.logs 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python -u qwen2_small_embed.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_images \
    --save_dir /work/gw42/w42010/msra/all_new_type/results/embeds/pope \
    --num_choices 2 \
    --model_qwen_size 2 \
    --data_mode val \
    --devi_vector_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pope \
    >/work/gw42/w42010/msra/all_new_type/results/embeds/pope/logs/qwen2_2b_val.logs 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python -u qwen2_small_embed.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_images \
    --save_dir /work/gw42/w42010/msra/all_new_type/results/embeds/pope \
    --num_choices 2 \
    --model_qwen_size 7 \
    --data_mode train \
    --devi_vector_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pope \
    >/work/gw42/w42010/msra/all_new_type/results/embeds/pope/logs/qwen2_7b_train.logs 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python -u qwen2_small_embed.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_images \
    --save_dir /work/gw42/w42010/msra/all_new_type/results/embeds/pope \
    --num_choices 2 \
    --model_qwen_size 7 \
    --data_mode val \
    --devi_vector_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pope \
    >/work/gw42/w42010/msra/all_new_type/results/embeds/pope/logs/qwen2_7b_val.logs 2>&1 &


CUDA_VISIBLE_DEVICES=4 nohup python -u qwen25_small_embed.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_images \
    --save_dir /work/gw42/w42010/msra/all_new_type/results/embeds/pope \
    --num_choices 2 \
    --model_qwen_size 3 \
    --data_mode train \
    --devi_vector_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pope \
    >/work/gw42/w42010/msra/all_new_type/results/embeds/pope/logs/qwen25_3b_train.logs 2>&1 &


CUDA_VISIBLE_DEVICES=5 nohup python -u qwen25_small_embed.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_images \
    --save_dir /work/gw42/w42010/msra/all_new_type/results/embeds/pope \
    --num_choices 2 \
    --model_qwen_size 3 \
    --data_mode val \
    --devi_vector_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pope \
    >/work/gw42/w42010/msra/all_new_type/results/embeds/pope/logs/qwen25_3b_val.logs 2>&1 &


CUDA_VISIBLE_DEVICES=6 nohup python -u qwen25_small_embed.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_images \
    --save_dir /work/gw42/w42010/msra/all_new_type/results/embeds/pope \
    --num_choices 2 \
    --model_qwen_size 7 \
    --data_mode train \
    --devi_vector_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pope \
    >/work/gw42/w42010/msra/all_new_type/results/embeds/pope/logs/qwen25_7b_train.logs 2>&1 &


CUDA_VISIBLE_DEVICES=7 nohup python -u qwen25_small_embed.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_images \
    --save_dir /work/gw42/w42010/msra/all_new_type/results/embeds/pope \
    --num_choices 2 \
    --model_qwen_size 7 \
    --data_mode val \
    --devi_vector_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pope \
    >/work/gw42/w42010/msra/all_new_type/results/embeds/pope/logs/qwen25_7b_val.logs 2>&1 &


wait
echo "All jobs finished."