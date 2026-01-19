import json
from typing import Union, List, Any, Dict, Literal
from abc import ABC
import re

class HumanEvalDataset(ABC):
    def __init__(self, 
                 split: Union[Literal['test']] = 'test',
                 data_path: str = "datasets/humaneval/humaneval-py.jsonl") -> None:
        """
        Initialize HumanEval dataset.
        
        Args:
            split: Dataset split. HumanEval only has 'test' split.
            data_path: Path to the JSONL file containing HumanEval problems.
        """
        self._split = split
        self._data_path = data_path
        self._data: List[Dict[str, Any]] = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'humaneval'

    @staticmethod
    def _load_data(data_path: str) -> List[Dict[str, Any]]:
        """Load HumanEval dataset from JSONL file."""
        data = []
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    data.append(json.loads(line))
        
        print(f"Total number of problems: {len(data)}")
        return data

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._data[index]

    @staticmethod
    def record_to_input(record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a HumanEval record to input format for the model.
        
        Args:
            record: A dictionary containing 'prompt' (function signature + docstring)
                   and other metadata.
        
        Returns:
            A dictionary with 'task' key containing the problem description.
        """
        # The 'prompt' field contains the function signature and docstring
        task_description = record.get('prompt', '')
        
        # Optionally include test cases as examples if they exist
        if 'test' in record and record['test']:
            task_description += f"\n\n# Test cases are provided internally for validation."
        
        input_dict = {"task": task_description}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        """Extract Python code from the model's response.
        
        This function extracts code from markdown code blocks or raw responses.
        It handles various formats like:
        - ```python ... ```
        - ``` ... ```
        - Raw code without markers
        
        Args:
            answer: The model's output, either a string or list of strings.
        
        Returns:
            Extracted Python code as a string.
        """
        if isinstance(answer, list):
            answer = "\n".join(filter(None, answer))
        
        if not isinstance(answer, str):
            return ""
        
        answer = answer.strip()
        
        if not answer:
            return ""
        
        # Try to extract code from markdown code blocks
        # Pattern 1: ```python ... ```
        python_code_pattern = r'```python\s*\n(.*?)```'
        matches = re.findall(python_code_pattern, answer, re.DOTALL)
        if matches:
            # Return the last code block (usually the final answer)
            return matches[-1].strip()
        
        # Pattern 2: ``` ... ``` (without language specifier)
        generic_code_pattern = r'```\s*\n(.*?)```'
        matches = re.findall(generic_code_pattern, answer, re.DOTALL)
        if matches:
            return matches[-1].strip()
        
        # Pattern 3: If no code blocks, try to extract code heuristically
        # Look for function definitions
        if 'def ' in answer:
            # Find the first 'def ' and take everything from there
            def_pos = answer.find('def ')
            if def_pos != -1:
                code = answer[def_pos:]
                # Clean up any trailing explanatory text after the code
                # This is a simple heuristic - take until we see markdown or explanatory patterns
                lines = code.split('\n')
                code_lines = []
                for line in lines:
                    # Stop if we encounter obvious non-code patterns
                    if line.strip() and not line.strip().startswith('#'):
                        # Check for explanatory text patterns
                        if any(pattern in line.lower() for pattern in [
                            'explanation:', 'note:', 'example:', 'usage:', 
                            'this function', 'this code', 'the above'
                        ]):
                            break
                    code_lines.append(line)
                return '\n'.join(code_lines).strip()
        
        # Pattern 4: Return as-is if it looks like valid Python code
        # Check if it starts with common Python keywords
        if any(answer.lstrip().startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ']):
            return answer
        
        # Last resort: return the original answer
        return answer

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        """Extract the canonical solution from the record.
        
        Args:
            record: A dictionary containing 'canonical_solution' or 'solution'.
        
        Returns:
            The canonical solution code as a string.
        """
        # HumanEval records typically have 'canonical_solution' field
        if 'canonical_solution' in record:
            return record['canonical_solution'].strip()
        elif 'solution' in record:
            return record['solution'].strip()
        else:
            return ""

    @staticmethod
    def record_to_test_cases(record: Dict[str, Any]) -> str:
        """Extract test cases from the record.
        
        Args:
            record: A dictionary containing 'test' field with test cases.
        
        Returns:
            Test cases as a string.
        """
        return record.get('test', '')

    @staticmethod
    def record_to_entry_point(record: Dict[str, Any]) -> str:
        """Extract the entry point (function name) from the record.
        
        Args:
            record: A dictionary containing 'entry_point' field.
        
        Returns:
            The function name to be tested.
        """
        return record.get('entry_point', '')
