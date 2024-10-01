data_dir=$1
seed=42
learning_rate=1e-4
rank=16
alpha=64

model_name_or_path=meta-llama/LLaMA-2-7b-hf
dataset_path="data/${data_dir}"
output_dir="output_models/mmlu/llama/finetuned_llama_paca_${data_dir}_seed_${seed}_${learning_rate}_rank_${rank}_alpha_${alpha}"

# Safety related arguments
trust_remote_code=0

# Finetune
exp_id=paca_llama2
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate ${learning_rate} \
    --block_size 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --grad_accum_steps 4 \
    --use_paca 1 \
    --lora_r ${rank} \
    --lora_alpha ${alpha} \
    --save_aggregated_lora 0\
    --bf16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 20 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 500000000 \
    --dataloader_num_workers 1 \
    --deepspeed configs/ds_config_zero0_no_offload.json\
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err


output_model_path=output_models/mmlu/llama/merge_paca_${data_dir}_seed_${seed}_${learning_rate}_rank_${rank}_alpha_${alpha}
device=cpu

if [ ${device} == "cpu" ]; then
    python examples/merge_lora.py \
        --model_name_or_path ${model_name_or_path} \
        --lora_model_path ${output_dir} \
        --output_model_path ${output_model_path} \
        --device ${device} \
        
elif [ ${device} == "gpu" ]; then
    echo "Error: Merging LoRA weights using gpu not supported yet. Please use cpu."
else
    echo "Error: Unknown device \"${device}\"" 1>&2
    exit 1
fi

lm_eval \
    --model hf \
    --model_args pretrained=${output_model_path}\
    --tasks mmlu \
    --output_path mmlu_results/llama/paca \
    --num_fewshot 5 \
    --batch_size 8 \
    --cache_requests true\