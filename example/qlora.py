import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- Configuration ---
# 1. Model and Dataset
MODEL_ID = "facebook/opt-125m" # A relatively small but capable 2.7B model
DATASET_ID = "tatsu-lab/alpaca" # A small, high-quality instruction dataset
OUTPUT_DIR = "./phi-2-alpaca-qlora"
MAX_SEQ_LENGTH = 512 # Max length for training samples

# 2. QLoRA and PEFT Configuration (Memory-Efficient Fine-Tuning)
# 4-bit Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# LoRA (Low-Rank Adaptation) Configuration
peft_config = LoraConfig(
    r=16, # LoRA attention dimension
    lora_alpha=32, # Alpha parameter for LoRA scaling
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. Training Arguments (Adjusted for Small Scale and 2x 3060)
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,                     # Small number of epochs for quick testing
    per_device_train_batch_size=4,          # Adjust batch size based on VRAM (4 is safe)
    gradient_accumulation_steps=1,          # Accumulate gradients over N steps
    optim="paged_adamw_32bit",              # Optimized AdamW for QLoRA
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,                              # Use float16 for speed/memory (works with bnb_4bit_compute_dtype)
    save_strategy="epoch",
    report_to="none",                       # Change to "wandb" to track with Weights & Biases
    # --- Distributed/Multi-GPU Setup ---
    # Since you have multiple GPUs, 'accelerate' will handle DDP automatically.
    # We set these for clarity, but 'accelerate launch' takes care of the distribution.
)

# --- Main Script ---
def finetune_model():
    print(f"Loading Model: {MODEL_ID}...")
    
    # Load model with QLoRA configuration
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # Automatically distributes the model across all available GPUs
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Prepare model for k-bit training (important for QLoRA)
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters to the model
    model = get_peft_model(model, peft_config)
    
    print("Model ready for PEFT/QLoRA.")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Set padding token

    print(f"Loading Dataset: {DATASET_ID}...")
    # Load a small sample of the dataset
    dataset = load_dataset(DATASET_ID, split="train[:500]") # Use 500 examples for fast fine-tuning

    # The dataset needs a function to format the input for instruction tuning
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            # Format the instruction/response pair into a Chat-like format
            text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction'][i]}\n\n### Input:\n{example['input'][i]}\n\n### Response:\n{example['output'][i]}"
            output_texts.append(text)
        return output_texts

    # Set up the Supervised Fine-Tuning (SFT) Trainer
    trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_arguments,
    formatting_func=formatting_prompts_func
    )

    print("Starting Fine-Tuning...")
    trainer.train()
    
    # Save the final LoRA adapters and tokenizer
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuning complete. Model adapters saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    finetune_model()

# --- Post-Training Inference Example (run after the script completes) ---
"""
from peft import PeftModel
from transformers import pipeline

# Reload the base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
# Load the fine-tuned LoRA adapters
model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

# Merge LoRA layers with the base model for easy deployment (optional)
# model = model.merge_and_unload() 

# Create a text generation pipeline
generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    device=0 # Use one of your GPUs for inference
)

prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain the concept of quantum entanglement.\n\n### Response:\n"

result = generator(prompt, max_length=256, do_sample=True, temperature=0.7)
print(result[0]['generated_text'])
"""
