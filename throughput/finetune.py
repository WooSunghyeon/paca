import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
    BitsAndBytesConfig,
)
import argparse
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PacaConfig
from peft.tuners.lora import LoraLayer
import wandb
from utils import (
    load_data,
    format_dataset,
    DataCollatorForCausalLM,
    generate_alpaca,
    get_subset,
    get_parameters_count,
    generate_samples,
    preprocess_logits_for_metrics,
    compute_metrics,
    MetricEvalCallback,
    GradientLogCallback,
    find_all_linear_names,
)
import time

torch.backends.cuda.matmul.allow_tf32 = True


def run(args):
    job_id = args.job_id if hasattr(args, 'job_id') and args.job_id is not None else os.environ.get("SLURM_JOB_ID", "0")
    run_id = args.run_id if hasattr(args, 'run_id') and args.run_id is not None else wandb.util.generate_id()

    set_seed(args.seed)

    wandb.init(
        id=run_id,
        name=None if args.run_name is None else args.run_name,
        group=None if args.run_group is None else args.run_group,
        project=args.run_project,
        mode="offline" if args.offline else "online",
    )
    wandb.config.update({"seed_val": args.seed})
    wandb.config.update(dict(args._get_kwargs()))
    wandb.config.update({"job_id": job_id, "init_job_id": args.init_job_id})

    imdb = load_dataset("imdb")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="right", use_fast=False, legacy=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token

    # asserts for tokenizer
    assert (
        tokenizer("### Response:", add_special_tokens=False)["input_ids"]
        + tokenizer(
            f"{'' if isinstance(tokenizer, LlamaTokenizer) else ' '}Negative",
            add_special_tokens=False,
        )["input_ids"]
        == tokenizer("### Response: Negative", add_special_tokens=False)["input_ids"]
    )
    assert (
        tokenizer(f"{tokenizer.bos_token}test", add_special_tokens=False)["input_ids"]
        == [tokenizer.bos_token_id]
        + tokenizer("test", add_special_tokens=False)["input_ids"]
    )

    def _imdb_to_alpaca(examples, _instruction, answers, cut_off=1000):
        instruction = []
        input = []
        output = []
        for i in range(len(examples["text"])):
            instruction.append(_instruction)
            input.append(f'"{examples["text"][i][:cut_off]}"')
            output.append(answers[0] if examples["label"][i] == 0 else answers[1])
        return {"output": output, "instruction": instruction, "input": input}

    def imdb_to_alpaca_easy(examples):
        return _imdb_to_alpaca(
            examples,
            'Given the following review, classify its sentiment. Answer with the exact sentence - "Review is negative." or "Review is positive.", but without quotes.',
            ["Review is negative.", "Review is positive."],
        )

    def imdb_to_alpaca_quotes(examples):
        return _imdb_to_alpaca(
            examples,
            'Given the following review, classify its sentiment. Answer with the exact sentence - "Review is negative." or "Review is positive.", with quotes.',
            ['"Review is negative."', '"Review is positive."'],
        )

    def imdb_to_alpaca_brackets(examples):
        return _imdb_to_alpaca(
            examples,
            'Given the following review, classify its sentiment. Answer with the exact sentence - "Review is negative." or "Review is positive.", but without quotes and put your answer in square brackets.',
            ["[Review is negative.]", "[Review is positive.]"],
        )

    eval_ds = {}
    ds_names = ["easy", "quotes", "brackets"]
    for name in ds_names:
        eval_ds[name] = (
            imdb["test"]
            if (args.eval_samples is None or args.eval_samples == 0)
            else get_subset(imdb["test"], args.eval_samples)
        )
        eval_ds[name] = eval_ds[name].map(
            eval(f"imdb_to_alpaca_{name}"),
            batched=True,
            remove_columns=imdb["train"].column_names,
        )
        eval_ds[name] = format_dataset(eval_ds[name], "alpaca-clean")

    if args.task == "instruct":
        dataset = load_data(args.dataset)
        dataset = format_dataset(dataset, args.dataset)
        train_ds = (
            dataset["train"]
            if (args.train_samples is None or args.train_samples == 0)
            else dataset["train"].select(range(args.train_samples))
        )
    elif args.task == "imdb":
        name = args.dataset
        train_ds = (
            imdb["train"]
            if (args.train_samples is None or args.train_samples == 0)
            else get_subset(imdb["train"], args.train_samples)
        )
        train_ds = train_ds.map(
            eval(f"imdb_to_alpaca_{name}"),
            batched=True,
            remove_columns=imdb["train"].column_names,
        )
        train_ds = format_dataset(train_ds, "alpaca-clean")
    else:
        raise NotImplementedError

    if args.nf4:
        model = AutoModelForCausalLM.from_pretrained(
           args.model,
           torch_dtype=torch.bfloat16,
           device_map="auto",
           quantization_config=BitsAndBytesConfig(
               load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        model = prepare_model_for_kbit_training(model)
        model.config.torch_dtype = torch.bfloat16
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    print(model)

    if args.custom_mode != "full":
        if args.target_modules == "lm_head":
            target_modules = ["lm_head"]
        elif args.target_modules == "all":
            target_modules = find_all_linear_names(
                model, lm_head=True
            )
        else:
            target_modules = find_all_linear_names(model)
        print("target_modules:", target_modules)
        
        if args.custom_mode == "lora":
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.00,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif args.custom_mode == "dora":
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.00,
                use_dora=True,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif args.custom_mode == "moslora":
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.00,
                use_moslora=True,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
        elif args.custom_mode == "paca":
            config = PacaConfig(
                r=args.lora_r,
                paca_alpha=args.lora_alpha,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
        #config.custom = {
        #    "mode": args.custom_mode,
        #    "submode": None,
        #    "d_init": 1.0,
        #    "sqrt_a": 5.0,
        #    "identity": not args.custom_disable_identity,
        #    "init_type": args.init_type,
        #    "d_init_type": args.d_init_type,
        #    "custom_scaling": args.custom_scaling == 1,
        #    "shared_dim": {"A": args.shared_dim, "B": args.shared_dim}
        #    if args.shared_uv == 1
        #    else None,
        #    "dynamic_uv": args.dynamic_uv == 1,
        #    "shared_matrices": None,
        #    "shared_d": False,
        #    "shared_d_vector": None,
        #    "trainable_uv": False,
        #    "nonlin": 0,
        #    "use_float64": False,
        #    "norm_penalty": 0,
        #    "norm_alpha": 0.0,
        #}
        model = get_peft_model(model, config)
        print(model)
        #if args.quantize:
        #    for name, module in model.named_modules():
        #        if isinstance(module, LoraLayer):
        #            module = module.to(torch.bfloat16)
        #        if "norm" in name:
        #            module = module.to(torch.float32)
        #        if "lm_head" in name or "embed_tokens" in name:
        #            if hasattr(module, "weight"):
        #                if module.weight.dtype == torch.float32:
        #                    module = module.to(torch.bfloat16)

    # print(model)

    training_args = TrainingArguments(
        output_dir="training_output",
        optim="adamw_torch",
        remove_unused_columns=False,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        dataloader_num_workers=4,
        num_train_epochs=args.epochs,
        weight_decay=args.wd,
        save_strategy="no",
        logging_steps=args.logging_steps,
        report_to="wandb",
        gradient_accumulation_steps=args.accumulation_steps,
        bf16=args.quantize,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        deepspeed=args.deepspeed,
    )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=512,
        target_max_len=256,
        train_on_source=False,
        predict_with_generate=False,
    )

    callbacks = []

    if args.metrics_enabled:
        metric_ds = train_ds.select(range(args.metric_samples))
        metricEvalCallback = MetricEvalCallback(
            metric_ds, tokenizer, model, args.metric_bs
        )
        callbacks.append(metricEvalCallback)

    gradientLogCallback = GradientLogCallback(model=model)
    callbacks.append(gradientLogCallback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

    params_trainable = get_parameters_count(model, requires_grad=True)
    params_total = get_parameters_count(model, requires_grad=False)

    print(f"Trainable parameters: {params_trainable}")
    print(f"Total number of parameters: {params_total}")
    wandb.config.update(
        {"params_trainable": params_trainable, "params_total": params_total}
    )
    wandb.log({"params_trainable": params_trainable, "params_total": params_total})
    print(args.default_model_eval)
    if args.default_model_eval:
        with torch.autocast("cuda"):
            model.eval()
            with torch.no_grad():
                start = time.time()
                #for n in ds_names:
                #    res = trainer.evaluate(eval_ds[n], metric_key_prefix=f"eval_{n}")
                #    print(f"eval {n}:", res)
                print(f"eval took {time.time() - start} seconds")
                if args.generate_samples:
                    to_eval = [data_collator(eval_ds[n].select([0])) for n in ds_names]
                    to_eval += [data_collator(train_ds.select([0]))]
                    samples_before = generate_samples(model, tokenizer, to_eval)

    model.train()
    trainer.train()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    with torch.autocast("cuda"):
        model.eval()
        with torch.no_grad():
            start = time.time()
            #for n in ds_names:
            #    res = trainer.evaluate(eval_ds[n], metric_key_prefix=f"eval_{n}")
            #    print(f"final eval {n}:", res)
            print(f"final eval took {time.time() - start} seconds")
            if args.generate_samples:
                samples_after = generate_samples(model, tokenizer, [], model_id=job_id)
                # samples_after = generate_samples(model, tokenizer, to_eval)
                data = []
                for i in range(len(samples_after)):
                    data.append(["", samples_after[i]])
                wandb.log(
                    {"generations": wandb.Table(data=data, columns=["before", "after"])}
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom_mode",
        type=str,
        default="full",
        choices=["full", "lora", "dora", "moslora", "paca"],
        help="mode of finetuning",
    )
    parser.add_argument(
        "--custom_submode",
        type=str,
        default="none",
        choices=["none", "lora_svd", "lora_half", "lora_half_svd"],
        help="submode of finetuning",
    )
    parser.add_argument("--custom_scaling", type=int, default=0)
    parser.add_argument("--shared_dim", type=int, default=0)
    parser.add_argument("--shared_uv", type=int, default=0)
    parser.add_argument("--dynamic_uv", type=int, default=0)
    parser.add_argument("--custom_d_init", type=float, default=1.0)
    parser.add_argument("--custom_sqrt_a", type=float, default=5)
    parser.add_argument("--custom_disable_identity", action="store_true")
    parser.add_argument("--init_type", type=int, default=1)
    parser.add_argument("--d_init_type", type=int, default=0)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--target_modules", type=str, default="lm_head")

    parser.add_argument(
        "--task", type=str, default="instruct", choices=["instruct", "imdb"]
    )
    parser.add_argument("--dataset", type=str, default="alpaca-clean")
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--metric_samples", type=int, default=100)
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--metric_bs", type=int, default=10)
    parser.add_argument("--eval_bs", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--metrics_enabled", type=int, default=0, choices=[0, 1])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--generate_samples", action="store_true")
    parser.add_argument("--default_model_eval", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")

    parser.add_argument("--run_project", type=str, default="default")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--run_group", type=str, default="default")
    parser.add_argument("--init_job_id", type=str, default=0)
    parser.add_argument("--job_id", type=str, default=None, help="Custom job ID")
    parser.add_argument("--run_id", type=str, default=None, help="Custom run ID")
    parser.add_argument("--offline", action="store_true")\
        
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--local_rank", type=int)
    
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    print("======= args =======")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("====================")

    run(args)
