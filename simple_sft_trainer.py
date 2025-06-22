#!/usr/bin/env python3
"""
Simple SFT trainer for RLLA dataset.
Uses the Transformers library directly instead of the complex veRL SFT framework.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import pandas as pd
import os
from typing import Dict, List

class RLLASFTDataset(Dataset):
    """Simple dataset for RLLA SFT training."""
    
    def __init__(self, parquet_file: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        df = pd.read_parquet(parquet_file)
        self.prompts = df['prompt'].tolist()
        self.responses = df['response'].tolist()
        
        print(f"Loaded {len(self.prompts)} examples from {parquet_file}")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        
        # Format as conversation
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template for prompt
        prompt_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Full text = prompt + response + eos
        full_text = prompt_text + response + self.tokenizer.eos_token
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Create labels (same as input_ids, but we'll mask the prompt part)
        labels = input_ids.clone()
        
        # Find where the prompt ends by tokenizing just the prompt
        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        
        # Mask prompt tokens in labels (set to -100 so they're ignored in loss)
        labels[:prompt_length] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def main():
    print("Starting Simple SFT Training for RLLA")
    
    # Configuration
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    train_file = "dataset/rlla_sft_processed/train.parquet"
    val_file = "dataset/rlla_sft_processed/test.parquet"
    output_dir = "checkpoints/simple_sft_rlla"
    
    print(f"Model: {model_name}")
    print(f"Train file: {train_file}")
    print(f"Output dir: {output_dir}")
    
    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Creating datasets...")
    train_dataset = RLLASFTDataset(train_file, tokenizer)
    val_dataset = RLLASFTDataset(val_file, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,  # Reduced for smaller dataset
        per_device_train_batch_size=4,  # Small batch size for memory efficiency
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 4*4 = 16
        warmup_steps=100,
        learning_rate=2e-5,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        bf16=True,  # Use bfloat16 for efficiency
        gradient_checkpointing=True,  # Save memory
        report_to=[],  # Disable wandb
    )
    
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model()
    
    print(f"✅ SFT training completed! Model saved to {output_dir}")

if __name__ == "__main__":
    main() 