# Fine-tuning LLaMA3-8b using PaCA and QPaCA on the Oasst1 Dataset

Due to licensing issues, we cannot release the code at this time. Once the licensing issues are resolved, we will make the code publicly available.

In this directory, we apply PaCA and QPaCA to general instruction-tuning code using Huggingface.

## Setup

Please follow the steps below to set up the environment and dependencies for fine-tuning LLaMA3-8b using PaCA and QPaCA.

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Install the custom version of PEFT that supports PaCA and QPaCA:

The detailed implementation can be found in `../peft/src/peft/tuners/paca/layers`

## Fine-tuning & Merging

Once the environment is set up, you can begin the fine-tuning process with the provided scripts. The scripts for running PaCA and QPaCA are as follows:

```bash
sh ./scripts/run_paca.sh # PaCA
sh ./scripts/run_qpaca.sh # QPaCA
```

## Evaluation on MT-Bench

After fine-tuning the model, we use [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) for evaluation. For evaluation, we use **gpt4o-mini** as the judge.


## Acknowledgements

Our research has been greatly influenced by [Vera](https://github.com/neurotechcenter/VERA), [PEFT](https://github.com/huggingface/peft), and [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge). We sincerely appreciate the excellent repositories they have provided.

 