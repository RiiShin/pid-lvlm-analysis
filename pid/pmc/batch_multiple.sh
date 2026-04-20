#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=00:30:00
#PJM -g gw42
#PJM -j
#------- Program execution -------#


source /work/gw42/w42010/anaconda3/bin/activate llava_embed

CUDA_VISIBLE_DEVICES=0 nohup python -u batch_vlm_final_drop.py \
    --directory /work/gw42/w42010/msra/all_new_type/results/embeds/pmc/ \
    --file_name qwen2_2.json \
    >./records/qwen2_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u batch_vlm_final_drop.py \
    --directory /work/gw42/w42010/msra/all_new_type/results/embeds/pmc/ \
    --file_name qwen2_7.json \
    >./records/qwen2_7.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u batch_vlm_final_drop.py \
    --directory /work/gw42/w42010/msra/all_new_type/results/embeds/pmc/ \
    --file_name qwen25_3.json \
    >./records/qwen25_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u batch_vlm_final_drop.py \
    --directory /work/gw42/w42010/msra/all_new_type/results/embeds/pmc/ \
    --file_name qwen25_7.json \
    >./records/qwen25_7.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4 nohup python -u batch_vlm_final_drop.py \
#     --directory /work/gw42/w42010/msra/all_new_type/results/embeds/pmc/ \
#     --file_name llava_ov_7.json \
#     >./records/llava_ov_7.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 nohup python -u batch_vlm_final_drop.py \
#     --directory /work/gw42/w42010/msra/all_new_type/results/embeds/pmc/ \
#     --file_name llava_ov_0.5.json \
#     >./records/llava_ov_0.5.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 nohup python -u batch_vlm_final_drop.py \
#     --directory /work/gw42/w42010/msra/all_new_type/results/embeds/pmc/ \
#     --file_name llama32_90.json \
#     >./records/llama32_90.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python -u batch_vlm_final_drop.py \
#     --directory /work/gw42/w42010/msra/all_new_type/results/embeds/pmc/ \
#     --file_name llama32_11.json \
#     >./records/llama32_11.log 2>&1 &


wait
echo "All jobs finished."