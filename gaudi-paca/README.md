# PaCA: Partial Connection Adaptation for Efficient Fine-Tuning
**PaCA** (**Pa**rtial **C**onnection **A**daptation) is new parameter-efficient fine-tuning (PEFT) algorithm for enhancing performance. PaCA not only reduces activation memory by storing only partial activations for backward propagation, but also reduces training time by eliminating additional sequential process by additional adapter layers as below:

![image](https://github.com/user-attachments/assets/9b59b1b9-a4dd-4513-84e7-fc9e3551bbce)


# PaCA
**PaCA** (**Pa**rtial **C**onnection **A**daptation) is new parameter-efficient fine-tuning (PEFT) algorithm for enhancing performance. PaCA not only reduces activation memory by storing only partial activations for backward propagation, but also reduces training time by eliminating additional sequential process by additional adapter layers as below:

## Setup

1. Install the required dependencies
```bash
pip install -q git+https://github.com/HabanaAI/DeepSpeed.git@1.18.0
```

2.  Install the custom optimum-habana library
 ```bash
cd ./optimum-habana
pip install -v -e .
```

3. Install the PEFT library which supports PaCA.
```bash
cd ./peft
pip install -v -e .
```   


## How to apply PaCA for fine-tuning 
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

## Examples 

Once the environment is set up, you can begin the fine-tuning process with the provided scripts. The scripts for running DropBP are as follows:

```bash
sh ./examples/paca.sh # PaCA
```
