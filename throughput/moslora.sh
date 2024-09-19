export WANDB_ENTITY=hoochoo
learning_rate=5e-4
rank=8

for train_bs in {19..1}
do
    deepspeed --master_port 29500 --include=localhost:0 finetune.py \
        --custom_mode moslora \
        --lr ${learning_rate} \
        --lora_r ${rank} \
        --train_bs ${train_bs} \
        --accumulation_steps 1 \
        --model meta-llama/Meta-Llama-3.1-8B \
        --seed 526 \
        --logging_steps 1 \
        --target_modules no_head \
        --metrics_enabled 0 \
        --lr_scheduler_type cosine \
        --run_project cal_fb_time \
        --run_name llama3_8b_lora_bs${train_bs} \
        --run_id llama3_8b_lora_bs${train_bs} \
        --job_id llama3_8b_lora_bs${train_bs} \
        --save_dir paca_models/dummy \
        --quantize \
        --deepspeed ds_config_zero0_no_offload.json \
        --task instruct --dataset oasst1 --epochs 1
done