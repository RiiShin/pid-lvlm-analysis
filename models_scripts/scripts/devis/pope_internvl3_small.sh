#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=regular-a
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


cd /work/gw42/w42010/msra/all_new_type/models_scripts/general_models/devis


CUDA_VISIBLE_DEVICES=0 nohup python -u internvl3_small_devi.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_images \
    --vector_save_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pope \
    --model_size 2 \
    --data_mode train \
    >/work/gw42/w42010/msra/all_new_type/results/devis_vector/pope/logs/internvl3_2b.logs 2>&1 &


CUDA_VISIBLE_DEVICES=4 nohup python -u internvl3_small_devi.py \
    --read_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_ \
    --image_dir /work/gw42/w42010/msra/all_new_type/data/pope/pope_images \
    --vector_save_dir /work/gw42/w42010/msra/all_new_type/results/devis_vector/pope \
    --model_size 8 \
    --data_mode train \
    >/work/gw42/w42010/msra/all_new_type/results/devis_vector/pope/logs/internvl3_8b.logs 2>&1 &


wait
echo "All jobs finished."