import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Merge a base model with a PEFT model and save the merged model.")
    parser.add_argument('--base_model_id', type=str, default='meta-llama/Meta-Llama-3-8B', help='Base model ID to load.')
    parser.add_argument('--peft_model_id', type=str, default='output_models/merge_lora', help='PEFT model ID to load.')
    parser.add_argument('--output_dir', type=str, default='output_models/merge_paca', help='Directory to save the merged model.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Set output directory
    output_dir = os.path.join(os.getcwd(), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        args.peft_model_id,
    )
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, args.peft_model_id)
    
    # Merge and unload
    merged_model = model.half().merge_and_unload()
    
    # Save merged model
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
