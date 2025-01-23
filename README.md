# PaCA: Partial Connection Adaptation for Efficient Fine-Tuning [ICLR2025]
This is the official PyTorch implementation of [PaCA: Partial Connection Adaption for Efficient Fine-Tuning](https://openreview.net/forum?id=iYkhxre0In&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions)).


## Abstract

![PaCA_FIG](https://github.com/user-attachments/assets/09667f50-abe5-42aa-9474-8053d1dcb3c4)


Prior parameter-efficient fine-tuning (PEFT) algorithms reduce memory usage and computational costs of fine-tuning large neural network models by training only a few additional adapter parameters, rather than the entire model. However, the reduction in computational costs due to PEFT does not necessarily translate to a reduction in training time; although the computational costs of the adapter layers are much smaller than the pretrained layers, it is well known that those two types of layers are processed sequentially on GPUs, resulting in significant latency overhead. LoRA and its variants avoid this latency overhead by merging the low-rank adapter matrices with the pretrained weights during inference. However, those layers cannot be merged during training since the pretrained weights must remain frozen while the low-rank adapter matrices are updated continuously over the course of training. Furthermore, LoRA and its variants do not reduce activation memory, as the first low-rank adapter matrix still requires the input activations to the pretrained weights to compute weight gradients. To mitigate this issue, we propose **Pa**rtial **C**onnection **A**aptation (**PaCA**), which fine-tunes randomly selected partial connections within the pretrained weights instead of introducing adapter layers in the model. PaCA not only enhances training speed by eliminating the time overhead due to the sequential processing of the adapter and pretrained layers but also reduces activation memory since only partial activations, rather than full activations, need to be stored for gradient computation. Compared to LoRA, PaCA reduces training time by 22% and total memory usage by 16%, while maintaining comparable accuracy across various fine-tuning scenarios, such as fine-tuning on the MMLU dataset and instruction tuning on the Oasst1 dataset. PaCA can also be combined with quantization, enabling the fine-tuning of large models such as LLaMA3.1-70B. In addition, PaCA enables training on 23% longer sequence data and improves throughput by 16% on both NVIDIA A100 and INTEL Gaudi 2 GPUs compared to LoRA.


## How to use
1. Download custom peft library which supports PaCA
```
cd peft
pip install -v -e .
```
1. Use paca as follows
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



