# Fine-tuning LLaMA2-7B/13B and LLaMA3-8b using PaCA on the MMLU Dataset

In this directory, we apply PaCA to [LMFLow](https://github.com/OptimalScale/LMFlow) for fine-tuning on MMLU dataset.

## Setup

Please follow the steps below to set up the environment and dependencies for using [LMFLow](https://github.com/OptimalScale/LMFlow) for fine-tuning.


1. Install the required dependencies:
```bash
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
bash install.sh
```

2. Install the custom version of PEFT that supports PaCA and QPaCA:

The detailed implementation can be found in `../peft/src/peft/tuners/paca/layers`


## Prepare Dataset

You can download processed MMLU datasets by [OwLore](https://github.com/pixeli99/OwLore) from Hugging Face [here](https://huggingface.co/datasets/pengxiang/OwLore_Dataset).

## Fine-tuning & Merging & Evaluation

We provide a quick scripts for fine-tuning, merging, and evaluating LLaMA2-7B on the MMLU dataset:
```bash
bash mmlu_scripts/run_paca.sh merge # PaCA
```

### Acknowledgement
This repository is based on the [LMFlow](https://github.com/OptimalScale/LMFlow) and [OwLore](https://github.com/pixeli99/OwLore) repositories. We are grateful for their excellent work.

## Citation
If you find our work helpful for your research, please consider citing the following BibTeX entry.
```
@misc{li2024owlore,
      title={OwLore: Outlier-weighed Layerwise Sampled Low-Rank Projection for Memory-Efficient LLM Fine-tuning}, 
      author={Pengxiang Li and Lu Yin and Xiaowei Gao and Shiwei Liu},
      year={2024},
      eprint={2405.18380},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```