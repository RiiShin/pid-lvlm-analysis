#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -g gw42
#PJM -j
#------- Program execution -------#


### You must check num_choices in the script before running it!!!
### mmbench num_choices is 4
### pope num_choices is 2
### pmc num_choices is 4

### num num num_choices!!!!!!!!!!!

source /work/gw42/w42010/anaconda3/bin/activate l_embed_new


cd /work/gw42/w42010/msra/all_new_type/models_scripts/general_models/embeds


CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u llama_11b_embed.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pmc/pmc_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pmc/pmc_images \
    --save_dir /work/gw42/w42010/msra/all_new_type/results/embeds/pmc \
    --num_choices 4 \
    --model_llama_size 11 \
    --data_mode train \
    --devi_vector_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pmc \
    >/work/gw42/w42010/msra/all_new_type/results/embeds/pmc/logs/llama_11b_train.logs 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u llama_11b_embed.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pmc/pmc_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pmc/pmc_images \
    --save_dir /work/gw42/w42010/msra/all_new_type/results/embeds/pmc \
    --num_choices 4 \
    --model_llama_size 11 \
    --data_mode val \
    --devi_vector_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pmc \
    >/work/gw42/w42010/msra/all_new_type/results/embeds/pmc/logs/llama_11b_val.logs 2>&1 &


wait
echo "All jobs finished."