export WANDB_ENTITY=hoochoo
learning_rate=5e-4
rank=8


#deepspeed --master_port 29500 --include=localhost:0 finetune.py \
#CUDA_VISIBLE_DEVICES=0 python finetune.py \
deepspeed --master_port 29500 --include=localhost:0 finetune.py \
    --custom_mode dora \
    --lr ${learning_rate} \
    --lora_r ${rank} \
    --train_bs 8 \
    --accumulation_steps 1 \
    --model meta-llama/Meta-Llama-3.1-8B\
    --seed 526 \
    --logging_steps 1 \
    --target_modules no_head \
    --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --run_project cal_fb_time \
    --run_name llama3_8b_lora\
    --run_id llama3_8b_lora\
    --job_id llama3_8b_lora\
    --save_dir paca_models/dummy\
    --quantize \
    --deepspeed ds_config_zero0_no_offload.json\
    --task instruct --dataset oasst1 --epochs 1 

deepspeed --master_port 29500 --include=localhost:0 finetune.py \
    --custom_mode dora \
    --lr ${learning_rate} \
    --lora_r ${rank} \
    --train_bs 7 \
    --accumulation_steps 1 \
    --model meta-llama/Meta-Llama-3.1-8B\
    --seed 526 \
    --logging_steps 1 \
    --target_modules no_head \
    --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --run_project cal_fb_time \
    --run_name llama3_8b_lora\
    --run_id llama3_8b_lora\
    --job_id llama3_8b_lora\
    --save_dir paca_models/dummy\
    --quantize \
    --deepspeed ds_config_zero0_no_offload.json\
    --task instruct --dataset oasst1 --epochs 1 


deepspeed --master_port 29500 --include=localhost:0 finetune.py \
    --custom_mode dora \
    --lr ${learning_rate} \
    --lora_r ${rank} \
    --train_bs 6 \
    --accumulation_steps 1 \
    --model meta-llama/Meta-Llama-3.1-8B\
    --seed 526 \
    --logging_steps 1 \
    --target_modules no_head \
    --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --run_project cal_fb_time \
    --run_name llama3_8b_lora\
    --run_id llama3_8b_lora\
    --job_id llama3_8b_lora\
    --save_dir paca_models/dummy\
    --quantize \
    --deepspeed ds_config_zero0_no_offload.json\
    --task instruct --dataset oasst1 --epochs 1 

deepspeed --master_port 29500 --include=localhost:0 finetune.py \
    --custom_mode dora \
    --lr ${learning_rate} \
    --lora_r ${rank} \
    --train_bs 5 \
    --accumulation_steps 1 \
    --model meta-llama/Meta-Llama-3.1-8B\
    --seed 526 \
    --logging_steps 1 \
    --target_modules no_head \
    --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --run_project cal_fb_time \
    --run_name llama3_8b_lora\
    --run_id llama3_8b_lora\
    --job_id llama3_8b_lora\
    --save_dir paca_models/dummy\
    --quantize \
    --deepspeed ds_config_zero0_no_offload.json\
    --task instruct --dataset oasst1 --epochs 1 

deepspeed --master_port 29500 --include=localhost:0 finetune.py \
    --custom_mode dora \
    --lr ${learning_rate} \
    --lora_r ${rank} \
    --train_bs 4 \
    --accumulation_steps 1 \
    --model meta-llama/Meta-Llama-3.1-8B\
    --seed 526 \
    --logging_steps 1 \
    --target_modules no_head \
    --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --run_project cal_fb_time \
    --run_name llama3_8b_lora\
    --run_id llama3_8b_lora\
    --job_id llama3_8b_lora\
    --save_dir paca_models/dummy\
    --quantize \
    --deepspeed ds_config_zero0_no_offload.json\
    --task instruct --dataset oasst1 --epochs 1 

deepspeed --master_port 29500 --include=localhost:0 finetune.py \
    --custom_mode dora \
    --lr ${learning_rate} \
    --lora_r ${rank} \
    --train_bs 3 \
    --accumulation_steps 1 \
    --model meta-llama/Meta-Llama-3.1-8B\
    --seed 526 \
    --logging_steps 1 \
    --target_modules no_head \
    --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --run_project cal_fb_time \
    --run_name llama3_8b_lora\
    --run_id llama3_8b_lora\
    --job_id llama3_8b_lora\
    --save_dir paca_models/dummy\
    --quantize \
    --deepspeed ds_config_zero0_no_offload.json\
    --task instruct --dataset oasst1 --epochs 1 

deepspeed --master_port 29500 --include=localhost:0 finetune.py \
    --custom_mode dora \
    --lr ${learning_rate} \
    --lora_r ${rank} \
    --train_bs 2 \
    --accumulation_steps 1 \
    --model meta-llama/Meta-Llama-3.1-8B\
    --seed 526 \
    --logging_steps 1 \
    --target_modules no_head \
    --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --run_project cal_fb_time \
    --run_name llama3_8b_lora\
    --run_id llama3_8b_lora\
    --job_id llama3_8b_lora\
    --save_dir paca_models/dummy\
    --quantize \
    --deepspeed ds_config_zero0_no_offload.json\
    --task instruct --dataset oasst1 --epochs 1 

deepspeed --master_port 29500 --include=localhost:0 finetune.py \
    --custom_mode dora \
    --lr ${learning_rate} \
    --lora_r ${rank} \
    --train_bs 1 \
    --accumulation_steps 1 \
    --model meta-llama/Meta-Llama-3.1-8B\
    --seed 526 \
    --logging_steps 1 \
    --target_modules no_head \
    --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --run_project cal_fb_time \
    --run_name llama3_8b_lora\
    --run_id llama3_8b_lora\
    --job_id llama3_8b_lora\
    --save_dir paca_models/dummy\
    --quantize \
    --deepspeed ds_config_zero0_no_offload.json\
    --task instruct --dataset oasst1 --epochs 1 








