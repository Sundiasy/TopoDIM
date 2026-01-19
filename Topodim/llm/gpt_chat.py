import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os

from Topodim.llm.format import Message
from Topodim.llm.price import cost_count
from Topodim.llm.llm import LLM
from Topodim.llm.llm_registry import LLMRegistry


OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    

    headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {authorization_key}"
    }
    data = {
        "model": model,
        "messages": msg,  
        "stream": False,
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(request_url, headers=headers, json=data) as response:
                response_text = await response.text()
                
                if response.status != 200:
                    raise Exception(f"API request failed with status {response.status}: {response_text}")
                
                response_data = await response.json()

                if 'choices' in response_data and len(response_data['choices']) > 0:
                    content = response_data['choices'][0]['message']['content']
                elif 'data' in response_data:
                    content = response_data['data']
                else:
                    raise Exception(f"Unexpected response format: {response_data}")
                
                prompt = "".join([item['content'] for item in msg])
                cost_count(prompt, content, model)
                return content
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

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
        
        return await achat(self.model_name, msg_dicts)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass