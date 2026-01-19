import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed
from typing import Dict, Any
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI, RateLimitError
import async_timeout
from transformers import AutoTokenizer
from aiolimiter import AsyncLimiter
from Topodim.llm.format import Message
from Topodim.llm.price import cost_count, cost_count_deepseek
from Topodim.llm.llm import LLM
from Topodim.llm.llm_registry import LLMRegistry
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')
MAX_CONCURRENT_REQUESTS = 3
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
RPM_LIMIT = 10
rate_limiter = AsyncLimiter(RPM_LIMIT, time_period=60)

aclient = AsyncOpenAI(api_key=MINE_API_KEYS, base_url=MINE_BASE_URL)

@retry(reraise=True, stop=stop_after_attempt(8), wait=wait_exponential_jitter(exp_base=2, max=8))
async def achat_deepseek(model: str, msg: List[Dict],):
    async with rate_limiter:
        try:
            async with async_timeout.timeout(1000):
                completion = await aclient.chat.completions.create(model=model,messages=msg)
            response_message = completion.choices[0].message.content
            
            if isinstance(response_message, str):
                prompt = "".join([item['content'] for item in msg])
                cost_count_deepseek(prompt, response_message, model)
                return response_message
        except RateLimitError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to complete the async chat request: {e}")


    
@LLMRegistry.register('deepseek')
class DeepseekChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        msg_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        return await achat_deepseek(self.model_name,msg_dicts)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

async def guarded_call(model, msg):
    async with semaphore:
        return await achat_deepseek(model, msg)