#!/bin/bash

# set timeout -1
# set password "7Sanctuaries!"
export tgt_host="yxu21@login.delta.ncsa.illinois.edu"
export tgt_dir="/scratch/bdhh/yxu21/pruning/LLM-Shearing/ckpts"
export src_dir="/scratch/yx3038/Research/pruning/LLM-Shearing/ckpts/llama2_7b_pruning_scaling_doremi_to2.7b_sl2048"

rsync -avz --progress ${src_dir} ${tgt_host}:${tgt_dir}

# expect {
#     "password:" {
#         send "$password\r"
#         exp_continue
#     }
#     "option (1-1):" {
#         send "1\r"
#     }
# }