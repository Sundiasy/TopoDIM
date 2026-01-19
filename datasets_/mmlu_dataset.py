import glob
import pandas as pd
from typing import Union, List, Literal, Any, Dict
import numpy as np
from abc import ABC
import re

class MMLUDataset(ABC):
    def __init__(self,
        split: Union[Literal['dev'], Literal['val'], Literal['test']],
        ) -> None:

        self._split = split

        data_path = f"datasets_/MMLU/data/{self._split}/"
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        csv_paths = glob.glob(data_path + "*.csv")
        csv_paths = sorted(csv_paths)
        print("Number of topics: ", len(csv_paths))

        names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None,
                            names=names,encoding='utf-8')
            total_df = pd.concat([total_df, single_df])

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
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    # def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
    #     if isinstance(answer, list):
    #         if len(answer) > 0:
    #             answer = answer[0]
    #         else:
    #             answer = ""
    #     if not isinstance(answer, str):
    #         raise Exception("Expected string")
    #     if len(answer) > 0:
    #         ans_pos = answer.find("answer is")
    #         if ans_pos != -1:
    #             answer = answer[ans_pos+len("answer is"):].strip(":").strip().strip("Option").strip()
    #         answer = answer[0] # Try to format the answer by taking the first letter
    #     return answer

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        """
        从大模型的复杂输出中稳健地提取单选题选项（A, B, C, D等）。

        该函数采用基于证据的计分策略，模拟人类的判断过程：
        1.  **预处理**: 清理和规范化输入文本。
        2.  **计分**: 
            - 强阳性证据 (加分多): 明确的答案指示词，如 "the answer is A", "答案是B"。
            - 弱阳性证据 (加分少): 结构性标志，如 "(C)" 或 "选项D"。
            - 位置权重: 越靠后的证据，权重越高，因为答案通常在最后总结。
        3.  **排除**: 
            - 强阴性证据 (直接淘汰): 明确的排除词，如 "A is incorrect", "排除B"。
        4.  **决策**:
            - 选出得分最高的选项。
            - 如果出现平分，选择在原文中最后被提及的那个作为最终答案。
            - 如果没有任何有效证据，返回空字符串。

        Args:
            answer: 模型的原始输出，可以是字符串或字符串列表。

        Returns:
            提取出的单个大写字母选项，如果无法可靠提取则返回空字符串。
        """
        # 步骤 0: 标准化输入
        if isinstance(answer, list):
            answer = " ".join(filter(None, answer))
        if not isinstance(answer, str) or not answer.strip():
            return ""

        text = answer.strip()
        
        # 假设我们处理的是A, B, C, D四个选项
        OPTIONS = ['A', 'B', 'C', 'D']
        scores = {opt: 0 for opt in OPTIONS}
        
        # 为了处理位置权重，我们记录每个证据出现的位置
        # (score, position, option)
        evidence = []

        # --- 步骤 1: 收集证据（阳性与阴性） ---

        # 强阳性证据模式: "answer is A", "选择 B" 等
        # 权重: +10
        strong_positive_patterns = [
            r'(?:the\s+answer\s+is|correct\s+answer\s+is|choice\s+is|my\s+answer\s+is|选择|答案\s*是)\s*[:：\s]*\s*([A-D])'
        ]
        
        # 弱阳性证据模式: "Option A", "(B)" 等
        # 权重: +3
        weak_positive_patterns = [
            r'(?:option|选项)\s+([A-D])',  # e.g., "Option A"
            r'[\(（\[【]([A-D])[\)）\]】]' # e.g., "(A)" or "[B]"
        ]

        # 结尾独立字母模式: 句子结尾的 A, B, C, D
        # 权重: +5 (比弱阳性强，因为通常是结论)
        final_char_patterns = [
            r'[.。,!！?？]\s*([A-D])$', # 结尾是 "... a."
            r':\s*([A-D])$' # 结尾是 "...: a"
        ]
        
        # 强阴性证据模式: "A is incorrect", "排除 B"
        # 这些模式应该让一个选项被直接排除
        negative_patterns = [
            r'(?:is\s+not|not|incorrect|wrong|错误|不正确|不是)\s+(?:option\s+|选项\s*)?([A-D])',
            r'排除\s*(?:option\s+|选项\s*)?([A-D])'
        ]

        # 收集强阳性证据
        for pattern in strong_positive_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                opt = match.group(1).upper()
                evidence.append({'score': 10, 'pos': match.start(), 'opt': opt})

        # 收集弱阳性证据
        for pattern in weak_positive_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                opt = match.group(1).upper()
                evidence.append({'score': 5, 'pos': match.start(), 'opt': opt})

        # 收集结尾独立字母证据
        for pattern in final_char_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                opt = match.group(1).upper()
                evidence.append({'score': 5, 'pos': match.start(), 'opt': opt})

        # 收集并应用强阴性证据
        eliminated_options = set()
        for pattern in negative_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                eliminated_options.add(match.group(1).upper())

        # --- 步骤 2: 计算最终得分 ---

        # 过滤掉被明确排除的选项的证据
        final_evidence = [ev for ev in evidence if ev['opt'] not in eliminated_options]

        if not final_evidence:
            # 如果没有任何正面证据，尝试最后的兜底策略：寻找文本中最后一个孤立的大写字母
            last_resort_matches = re.findall(r'\b([A-D])\b', text)
            if last_resort_matches:
                return last_resort_matches[-1].upper()
            # 如果模型只输出一个字母
            if len(text) == 1 and text.upper() in OPTIONS:
                return text.upper()
            return ""

        # 计算每个选项的总分，考虑位置权重
        # 简单实现：将位置作为小数加到分数上，保证同样分数时，位置靠后的胜出
        for ev in final_evidence:
            scores[ev['opt']] += ev['score'] + (ev['pos'] / len(text))
        
        # --- 步骤 3: 决策 ---

        # 移除得分为0的选项
        valid_scores = {opt: score for opt, score in scores.items() if score > 0}
        if not valid_scores:
            return ""

        # 找到最高分
        max_score = max(valid_scores.values())

        # 找到所有获得最高分的候选人
        top_candidates = [opt for opt, score in valid_scores.items() if score >= max_score]

        # 如果只有一个最高分，直接返回
        if len(top_candidates) == 1:
            return top_candidates[0]
        
        # 如果有多个最高分（平分），这是一个棘手的情况
        # 策略：选择在原文中最后一次被提及的那个
        last_positions = {}
        for opt in top_candidates:
            # 使用 rfind 找到每个候选者最后出现的位置
            last_positions[opt] = text.upper().rfind(opt)
        
        # 返回位置最靠后的那个
        return max(last_positions, key=last_positions.get)

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer
