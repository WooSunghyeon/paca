# Setup

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
# Examples 

Once the environment is set up, you can begin the fine-tuning process with the provided scripts. The scripts for running PaCA are as follows:

```bash
sh ./examples/paca.sh # PaCA
```
