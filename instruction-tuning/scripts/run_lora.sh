export WANDB_ENTITY=hoochoo
export WANDB_PROJECT=alpaca_10k

deepspeed --master_port 29501 finetune.py \
    --train_samples 10000 \
    --custom_mode lora \
    --lr 4e-4 \
    --lora_r 64 \
    --train_bs 4 \
    --accumulation_steps 4 \
    --model /home/wshey/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6 \
    --seed 42 \
    --logging_steps 1 \
    --target_modules no_head \
    --generate_samples \
    --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --run_project instruct-paca \
    --run_name lora\
    --run_id lora\
    --job_id lora\
    --save_dir output_models/llama3-8b/lora\
    --quantize \
    --default_model_eval \
    --deepspeed ds_config_zero0_no_offload.json\
    --task instruct --dataset alpaca-clean --epochs 1 
