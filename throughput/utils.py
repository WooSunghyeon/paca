import transformers
from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import torch
import copy
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import numpy as np
import time
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import wandb
import evaluate
import bitsandbytes as bnb
import json
import os
import shortuuid


IGNORE_INDEX = -100


def find_all_linear_names(model, lm_head=False, quantize=False):
    cls = bnb.nn.Linear4bit if quantize else torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if not lm_head and "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    modules = list(lora_module_names)
    if lm_head:
        modules = (modules + ["lm_head"]) if "lm_head" not in modules else modules
    return modules


def get_subset(ds, size):
    # get subset with equal number of positive and negative examples
    counts = {"pos_left": size // 2, "neg_left": size - (size // 2)}

    def filter_fn(example):
        if counts["pos_left"] > 0 and example["label"] == 1:
            counts["pos_left"] -= 1
            return True
        if counts["neg_left"] > 0 and example["label"] == 0:
            counts["neg_left"] -= 1
            return True
        return False

    return ds.filter(filter_fn)


def get_parameters_count(model, requires_grad=False):
    total_params = 0
    unique_tensors = set()
    for name, module in model.named_modules():
        for attr_str in dir(module):
            if attr_str == "trainer":  # Skip the trainer attribute
                continue
            target_attr = getattr(module, attr_str)
            if type(target_attr) in (torch.Tensor, torch.nn.Parameter):
                if (
                    id(target_attr) not in unique_tensors
                ):  # Check if the tensor was already counted
                    if not requires_grad or target_attr.requires_grad:
                        # print(name, attr_str, target_attr.shape)
                        total_params += torch.numel(target_attr)
                    unique_tensors.add(
                        id(target_attr)
                    )  # Add the tensor id to the set of counted tensors
    return total_params


def find_end_index(tensor, sequence):
    sequence_length = len(sequence)

    last_match = -1

    for i in range(len(tensor) - sequence_length + 1):
        if torch.equal(tensor[i : i + sequence_length], sequence):
            last_match = i + sequence_length
        elif last_match > -1:
            return last_match

    return last_match


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    # target_sequence = torch.tensor([21017, 15286, 25]).to("cuda:0")
    target_sequence = torch.tensor([-100, -100]).to("cuda:0")
    pred_ids = torch.argmax(logits, dim=-1)
    pred_ids = pred_ids[:, :-2]
    labels = labels[:, 1:-1]
    equal = []
    for i in range(pred_ids.shape[0]):
        p = pred_ids[i]
        l = labels[i]
        idx = find_end_index(l, target_sequence)
        # if i == 0:
        #     print(p)
        #     print(l)
        # print(p[idx - 4 : idx + 4], l[idx - 4 : idx + 4])
        if idx == -1:
            raise ValueError("Sequence not found")
        m = l[idx:] == -100
        comp = p[idx:] == l[idx:]
        comp += m
        equal.append(comp.min())
    return equal


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.concatenate(predictions)
    return {"Accuracy": predictions.sum() / len(predictions)}


def generate_samples(model, tokenizer, to_eval, model_id="default"):
    print("Generating samples...")
    start = time.time()
    data = []

    answers_folder_path = os.path.join(os.getcwd(), "answers")
    os.makedirs(answers_folder_path, exist_ok=True)
    
    answer_file_path = os.path.join(answers_folder_path, f"{model_id}.jsonl")
    
    # Creating the answer file initially
    with open(answer_file_path, "w") as ans_file:
        pass

    middle = time.time()

    try:
        def get_json_list(file_path):
            file_path = os.path.expanduser(file_path)
            with open(file_path, "r") as f:
                json_list = []
                for line in f:
                    json_list.append(json.loads(line))
                return json_list

        question_file_path = os.path.join(os.getcwd(), "question.jsonl")
        questions = get_json_list(question_file_path)
        
        for i, q in enumerate(questions):
            print(f"Generating sample {i}...")
            _data = generate_alpaca(model, tokenizer, q["turns"])
            print("=================")
            print(_data)
            data.append([_data])
            ans_id = shortuuid.uuid()
            answer = {
                "question_id": q["question_id"],
                "turns": _data,
                "answer_id": ans_id,
                "model_id": model_id,
                "metadata": {},
                "category": q["category"],
            }
            
            # Write the answer to the JSONL file immediately
            with open(answer_file_path, "a") as ans_file:
                ans_file.write(json.dumps(answer) + "\n")

    except Exception as e:
        print(e)
    print("=================")

    end = time.time()
    print(
        f"Generating samples took {end - start} seconds (first part {middle - start} seconds, second part {end - middle} seconds))"
    )

    return data


class GradientLogCallback(TrainerCallback):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_step = -1

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.last_step >= state.global_step or state.global_step % 10 != 0:
            return
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        print(f"gradient norm ({state.global_step}): {total_norm}")
        wandb.log({"gradient_norm": total_norm})
        self.last_step = state.global_step


class MetricEvalCallback(TrainerCallback):
    def __init__(self, _to_eval, tokenizer, model, batch_size):
        super().__init__()
        self._to_eval = _to_eval
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size
        self.last_step = -1

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        try:
            if self.last_step >= state.global_step:
                return
            with torch.autocast("cuda"):
                bleu = evaluate.load("bleu")
                rouge = evaluate.load("rouge")
                print("evaluating with metrics...")
                start = time.time()
                references = []
                predictions = []
                for i in range(0, len(self._to_eval["input"]), self.batch_size):
                    inputs = self._to_eval["input"][i : i + self.batch_size]
                    outputs = self._to_eval["output"][i : i + self.batch_size]
                    self.tokenizer.padding_side = "left"
                    input = self.tokenizer(
                        inputs, padding=True, return_tensors="pt"
                    ).to("cuda:0")
                    _references = [[s + outputs[j]] for j, s in enumerate(inputs)]
                    _predictions = self.tokenizer.batch_decode(
                        self.model.generate(
                            **input,
                            max_length=input["input_ids"].shape[-1] + 64,
                            pad_token_id=self.tokenizer.pad_token_id,
                        ),
                        skip_special_tokens=True,
                    )
                    references += _references
                    predictions += _predictions
                bleu_results = bleu.compute(
                    predictions=predictions, references=references
                )
                rouge_results = rouge.compute(
                    predictions=predictions, references=references
                )
                print(bleu_results, rouge_results)
                _step = state.global_step if state.global_step > 0 else None
                wandb.log(
                    {"rougeL": rouge_results["rougeL"], "bleu": bleu_results["bleu"]},
                    step=_step,
                )
                print(f"metric evaluation took {time.time() - start} seconds")
                self.last_step = state.global_step
        except Exception as e:
            print("metric eval error:", e)


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [
            f"{self.tokenizer.bos_token}{example['input']}" for example in instances
        ]
        # targets = [
        #     f"{example['output']}{self.tokenizer.eos_token}" for example in instances
        # ]
        full = [
            f"{self.tokenizer.bos_token}{example['input']}{example['output']}{self.tokenizer.eos_token}"
            for example in instances
        ]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
            padding='max_length',
        )
        # tokenized_targets = self.tokenizer(
        #     targets,
        #     max_length=self.target_max_len,
        #     truncation=True,
        #     add_special_tokens=False,
        # )
        tokenized_full_text = self.tokenizer(
            full,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
            padding='max_length',
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_full in zip(
            tokenized_sources_with_prompt["input_ids"], tokenized_full_text["input_ids"]
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_full))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + tokenized_full[len(tokenized_source) :]
                        )
                    )
                else:
                    labels.append(torch.tensor(tokenized_full))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict


def load_data(dataset_name):
    if dataset_name == "alpaca":
        return load_dataset("tatsu-lab/alpaca")
    elif dataset_name == "alpaca-clean":
        return load_dataset("yahma/alpaca-cleaned")
    elif dataset_name == "chip2":
        return load_dataset("laion/OIG", data_files="unified_chip2.jsonl")
    elif dataset_name == "self-instruct":
        return load_dataset("yizhongw/self_instruct", name="self_instruct")
    elif dataset_name == "hh-rlhf":
        return load_dataset("Anthropic/hh-rlhf")
    elif dataset_name == "longform":
        return load_dataset("akoksal/LongForm")
    elif dataset_name == "oasst1":
        return load_dataset("timdettmers/openassistant-guanaco")
    elif dataset_name == "vicuna":
        raise NotImplementedError("Vicuna data was not released.")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")


_ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}
ALPACA_PROMPT_DICT = {
    "prompt_input": "### Human:{instruction} Input: {input}\n### Assistant:",
    "prompt_no_input": "### Human:{instruction}\n### Assistant:",
}


def generate_alpaca(model, tokenizer, instruction, length=1024):
    prompt = tokenizer.bos_token + ALPACA_PROMPT_DICT["prompt_no_input"].format(
        instruction=instruction
    )

    input = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(
        "cuda:0"
    )
    response = tokenizer.batch_decode(
        model.generate(
            inputs=input,
            max_length=input.shape[-1] + length,
            pad_token_id=tokenizer.pad_token_id,
        ),
        skip_special_tokens=False,
    )[0]
    return response


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {"input": prompt_format.format(**example)}


def format_dataset(dataset, dataset_format):
    if (
        dataset_format == "alpaca"
        or dataset_format == "alpaca-clean"
        or (dataset_format is None and dataset in ["alpaca", "alpaca-clean"])
    ):
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=["instruction"])
    elif dataset_format == "chip2" or (dataset_format is None and dataset == "chip2"):
        dataset = dataset.map(
            lambda x: {
                "input": x["turns"].split("\n<bot>: ")[0].replace("<human>: ", ""),
                "output": x["turns"].split("\n<bot>: ")[1],
            }
        )
    elif dataset_format == "self-instruct" or (
        dataset_format is None and dataset == "self-instruct"
    ):
        for old, new in [["prompt", "input"], ["completion", "output"]]:
            dataset = dataset.rename_column(old, new)
    elif dataset_format == "hh-rlhf" or (
        dataset_format is None and dataset == "hh-rlhf"
    ):
        dataset = dataset.map(lambda x: {"input": "", "output": x["chosen"]})
    elif dataset_format == "oasst1" or (dataset_format is None and dataset == "oasst1"):
        dataset = dataset.map(
            lambda x: {
                "input": "",
                "output": x["text"],
            }
        )
    elif dataset_format == "input-output":
        # leave as is
        pass
    # Remove unused columns.
    # dataset = dataset.remove_columns(
    #     [
    #         col
    #         # for col in dataset.column_names["train"]
    #         for col in dataset.column_names
    #         if col not in ["input", "output"]
    #     ]
    # )
    return dataset
