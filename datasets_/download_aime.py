import json
import re
import random  # 1. 导入 random 模块
from datasets import load_dataset

random.seed(42)  # 设置随机种子，确保可复现
# 定义输出文件名
OUTPUT_FILENAME = "aime_2023_2025_shuffled.jsonl"

def clean_answer(ans):
    if ans is None:
        return ""
    ans_str = str(ans).strip()
    
    try:
        if "." in ans_str:
            return str(int(float(ans_str)))
    except:
        pass
    
    return ans_str

def get_2025_data():
    print("--- downloading AIME 2025 (Source: yentinglin/aime_2025) ---")
    data = []
    try:
        ds = load_dataset("yentinglin/aime_2025", split="train")
        for item in ds:
            data.append({
                "question": item.get("problem", "").strip(),
                "answer": clean_answer(item.get("answer")),
                "year": "2025"
            })
    except Exception as e:
        print(f"2025 download fail: {e}")
    return data

def get_2024_data():
    """ 2024 """
    print("--- downloading AIME 2024 (Source: Maxwell-Jia/AIME_2024) ---")
    data = []
    try:
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        for item in ds:
            data.append({
                "question": item.get("Problem", "").strip(),
                "answer": clean_answer(item.get("Answer")),
                "year": "2024"
            })
    except Exception as e:
        print(f"2024 download fail: {e}")
    return data

def get_2023_data():
    """ 2023 """
    print("--- downloading AIME 2023 (Source: MathArena/aime_2023_I and aime_2023_II) ---")
    data = []
    try:
        ds = load_dataset("MathArena/aime_2023_I", split="train")
        for item in ds:
            data.append({
                "question": item.get("problem", "").strip(),
                "answer": clean_answer(item.get("answer")),
                "year": "2023"
            })
        ds2 = load_dataset("MathArena/aime_2023_II", split="train")
        for item in ds2:
            data.append({
                "question": item.get("problem", "").strip(),
                "answer": clean_answer(item.get("answer")),
                "year": "2023"
            })
    except Exception as e:
        print(f"2023 download fail (please check if Dataset ID is correct): {e}")
    return data

def main():
    all_records = []
    
    records_2023 = get_2023_data()
    records_2024 = get_2024_data()
    records_2025 = get_2025_data()

    all_records.extend(records_2023)
    all_records.extend(records_2024)
    all_records.extend(records_2025)
    
    total_count = len(all_records)
    print(f"\nTotal records obtained: {total_count}")

    if total_count > 0:
        # 3. Shuffle the data
        print("Shuffling data order...")
        random.shuffle(all_records) # In-place shuffle of the list

        print(f"Writing to local file: {OUTPUT_FILENAME} ...")

        # 4. Write to JSONL
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            for record in all_records:
                # Construct the final object, keeping only question and answer
                final_obj = {
                    "question": record["question"],
                    "answer": record["answer"]
                }
                f.write(json.dumps(final_obj, ensure_ascii=False) + "\n")
                
        print("Processing complete!")
        
        # Print shuffled sample checks
        print("\n--- Random Sample 1 (Head) ---")
        # Print year for verification, not included in the file
        print(f"[Year: {all_records[0]['year']}] " + json.dumps(all_records[0], ensure_ascii=False)[:100] + "...")
        
        print("\n--- Random Sample 2 (Tail) ---")
        print(f"[Year: {all_records[-1]['year']}] " + json.dumps(all_records[-1], ensure_ascii=False)[:100] + "...")
    else:
        print("No data obtained, skipping write.")

if __name__ == "__main__":
    main()