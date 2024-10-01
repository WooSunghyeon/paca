# PaCA: Partial Connection Adaptation for Efficient Fine-Tuning
This is the official PyTorch implementation of **PaCA: Partial Connection Adaption for Efficient Fine-Tuning**.


## Abstract




## how to use
1. download custom peft library which supports PaCA.
```
cd peft
pip install -v -e .
```
2. use paca as follow
```
from peft import PacaConfig, get_peft_model

peft_config = PacaConfig(
                r=8,
                paca_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

model = get_peft_model(model, peft_config)
```



