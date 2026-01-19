from typing import Optional
from class_registry import ClassRegistry

from Topodim.llm.llm import LLM

class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name == "":
            model_name = "qwen-local"

        if model_name == 'mock':
            model = cls.registry.get('mock')
        
        elif 'deepseek' in model_name:
            model = cls.registry.get('deepseek', model_name)

        elif 'qwen' in model_name.lower():
            model = cls.registry.get('QwenLocalChat', model_name)
            
        else:
            model = cls.registry.get('QwenLocalChat', model_name)

        return model