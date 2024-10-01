learning_rate=5e-3

deepspeed --master_port 29501 --include=localhost:1 finetune.py \
    --custom_mode paca \
    --lr ${learning_rate} \
    --lora_r 64 \
    --train_bs 4 \
    --accumulation_steps 4 \
    --model meta-llama/Meta-Llama-3-8B \
    --seed 42 \
    --logging_steps 1 \
    --target_modules no_head \
    --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --deepspeed ds_config_zero0_no_offload.json\
    --task instruct --dataset oasst1 --epochs 1 
