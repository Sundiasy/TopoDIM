from datasets import load_dataset
import os
import pandas as pd

def download():
    print("downloading TIGER-Lab/MMLU-Pro dataset...")
    try:
        # 需要安装: pip install datasets pyarrow
        ds = load_dataset("TIGER-Lab/MMLU-Pro")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please ensure the datasets library is installed: pip install datasets")
        return

    base_path = "datasets_/MMLU_PRO/data"
    os.makedirs(base_path, exist_ok=True)
    
    # MMLU-Pro usually contains 'test' and 'validation'
    for split in ds.keys():
        # Map validation to val to follow project conventions
        save_split = 'val' if split == 'validation' else split
        
        df = ds[split].to_pandas()
        save_path = os.path.join(base_path, f"{save_split}.parquet")
        
        # Save as parquet format to preserve list-type options column
        df.to_parquet(save_path)
        print(f"Saved {split} split to: {save_path} (total {len(df)} records)")

if __name__ == "__main__":
    download()