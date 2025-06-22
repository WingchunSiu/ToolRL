#!/usr/bin/env python3
"""
Process the RLLA SFT dataset to parquet format for veRL SFT training.
This uses the dedicated rlla_sft.json which has the proper instruction-input-output format.
"""

import json
import pandas as pd
import os
import numpy as np

def main():
    print("Processing RLLA SFT dataset...")
    
    # Load SFT dataset
    with open('dataset/rlla_4k_raw/rlla_sft.json', 'r') as f:
        sft_data = json.load(f)
    
    print(f"Loaded {len(sft_data)} SFT samples")
    
    # Process data
    processed_data = []
    
    for i, sample in enumerate(sft_data):
        instruction = sample['instruction']
        input_text = sample['input'] 
        output = sample['output']
        
        # Combine instruction and input as the prompt
        if input_text.strip():
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction
        
        processed_data.append({
            'prompt': prompt,
            'response': output,
            'data_source': 'rlla_sft',
            'original_index': i
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Split into train/val (90/10 split since it's small)
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    
    val_size = int(0.1 * len(df))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    
    # Create output directory
    output_dir = 'dataset/rlla_sft_processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to parquet
    train_df.to_parquet(f'{output_dir}/train.parquet', index=False)
    val_df.to_parquet(f'{output_dir}/test.parquet', index=False)
    
    print(f"Saved processed SFT data to {output_dir}/")
    
    # Show sample
    print("\nSample processed data:")
    print("PROMPT:", train_df.iloc[0]['prompt'][:200] + "...")
    print("RESPONSE:", train_df.iloc[0]['response'][:200] + "...")

if __name__ == "__main__":
    main() 