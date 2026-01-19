import pandas as pd
from typing import Union, List, Literal, Any, Dict
import numpy as np
from abc import ABC
import re
import os

class MMLUProDataset(ABC):
    def __init__(self,
        split: Union[Literal['dev'], Literal['val'], Literal['test']],
        ) -> None:

        self._split = split
        
        # 映射 split 名称
        file_split = 'val' if self._split in ['val', 'dev'] else self._split
        
        data_path = f"datasets_/MMLU_PRO/data/{file_split}.parquet"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"未找到数据文件: {data_path}。请先运行 datasets_/download_mmlu_pro.py")
            
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu_pro'

    @staticmethod
    def _load_data(data_path: str) -> pd.DataFrame:
        rng = np.random.default_rng(888)
        
        print(f"Loading MMLU-Pro from {data_path}...")
        total_df = pd.read_parquet(data_path)
        
        # 过滤掉选项数量超过10个的异常数据（MMLU-Pro标准是10个）
        total_df = total_df[total_df['options'].apply(len) <= 10]
        
        total_df = total_df.reset_index(drop=True)
        # Pseudorandom shuffle
        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of questions: ", len(total_df))
        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        # MMLU-Pro 的 options 是一个列表
        options = record['options']
        options_str = ""
        for i, opt in enumerate(options):
            letter = chr(65 + i) # A, B, C...
            options_str += f"Option {letter}: {opt}\n"
            
        demo_question = (
            f"{record['question']}\n"
            f"{options_str}"
        )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        """
        针对 MMLU-Pro (A-J) 适配的答案提取函数
        """
        if isinstance(answer, list):
            answer = " ".join(filter(None, answer))
        if not isinstance(answer, str) or not answer.strip():
            return ""

        text = answer.strip()
        
        # MMLU-Pro 最多支持 A-J (10个选项)
        OPTIONS = [chr(65+i) for i in range(10)] # ['A', 'B', ..., 'J']
        scores = {opt: 0 for opt in OPTIONS}
        evidence = []

        # 正则表达式适配 A-J
        strong_positive_patterns = [
            r'(?:the\s+answer\s+is|correct\s+answer\s+is|choice\s+is|my\s+answer\s+is|选择|答案\s*是)\s*[:：\s]*\s*([A-J])'
        ]
        weak_positive_patterns = [
            r'(?:option|选项)\s+([A-J])',
            r'[\(（\[【]([A-J])[\)）\]】]'
        ]
        final_char_patterns = [
            r'[.。,!！?？]\s*([A-J])$',
            r':\s*([A-J])$'
        ]
        negative_patterns = [
            r'(?:is\s+not|not|incorrect|wrong|错误|不正确|不是)\s+(?:option\s+|选项\s*)?([A-J])',
            r'排除\s*(?:option\s+|选项\s*)?([A-J])'
        ]

        # 1. 收集证据
        for pattern in strong_positive_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                opt = match.group(1).upper()
                if opt in OPTIONS:
                    evidence.append({'score': 10, 'pos': match.start(), 'opt': opt})

        for pattern in weak_positive_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                opt = match.group(1).upper()
                if opt in OPTIONS:
                    evidence.append({'score': 5, 'pos': match.start(), 'opt': opt})

        for pattern in final_char_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                opt = match.group(1).upper()
                if opt in OPTIONS:
                    evidence.append({'score': 5, 'pos': match.start(), 'opt': opt})

        eliminated_options = set()
        for pattern in negative_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                opt = match.group(1).upper()
                if opt in OPTIONS:
                    eliminated_options.add(opt)

        # 2. 计算得分
        final_evidence = [ev for ev in evidence if ev['opt'] not in eliminated_options]

        if not final_evidence:
            # 兜底策略：寻找最后一个孤立的大写字母 (A-J)
            last_resort_matches = re.findall(r'\b([A-J])\b', text)
            if last_resort_matches:
                return last_resort_matches[-1].upper()
            if len(text) == 1 and text.upper() in OPTIONS:
                return text.upper()
            return ""

        for ev in final_evidence:
            scores[ev['opt']] += ev['score'] + (ev['pos'] / len(text))
        
        # 3. 决策
        valid_scores = {opt: score for opt, score in scores.items() if score > 0}
        if not valid_scores:
            return ""

        max_score = max(valid_scores.values())
        top_candidates = [opt for opt, score in valid_scores.items() if score >= max_score]

        if len(top_candidates) == 1:
            return top_candidates[0]
        
        last_positions = {}
        for opt in top_candidates:
            last_positions[opt] = text.upper().rfind(opt)
        
        return max(last_positions, key=last_positions.get)

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        # MMLU-Pro 的 answer 字段通常直接就是字母，或者 answer_index
        if 'answer' in record and record['answer']:
            return record['answer']
        elif 'answer_index' in record:
            return chr(65 + record['answer_index'])
        raise ValueError(f"Cannot find answer in record: {record}")