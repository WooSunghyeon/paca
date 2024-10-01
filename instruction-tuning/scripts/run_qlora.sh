learning_rate=5e-4

deepspeed --master_port 29501 --include=localhost:0 finetune.py \
    --custom_mode lora \
    --lr ${learning_rate} \
    --lora_r 64 \
    --train_bs 8 \
    --accumulation_steps 2 \
    --model meta-llama/Meta-Llama-3.1-70B \
    --seed 526 \
    --logging_steps 1 \
    --target_modules no_head \
    --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --quantize \
    --deepspeed ds_config_zero0_no_offload.json\
    --task instruct --dataset oasst1 --epochs 1 





