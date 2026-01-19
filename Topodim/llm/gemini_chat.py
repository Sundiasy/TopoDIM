import aiohttp
from typing import List, Union, Optional, Dict, Any
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import os
import json

from Topodim.llm.format import Message
from Topodim.llm.llm import LLM
from Topodim.llm.llm_registry import LLMRegistry
from Topodim.llm.price import cost_count

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_API_BASE = os.getenv('GEMINI_API_BASE')
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY not found in environment variables.")
# if not GEMINI_API_BASE:
#     raise ValueError("GEMINI_API_BASE not found in environment variables.")


@retry(wait=wait_random_exponential(max=60), stop=stop_after_attempt(3))
async def agemini_chat(
    model_name: str,
    history: List[Dict],
    system_instruction: Optional[str],
    generation_config: Dict[str, Any]
) -> str:
    request_url = GEMINI_API_BASE
    authorization_key = GOOGLE_API_KEY

    headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {authorization_key}"
    }

    data = {
        "model": model_name,
        "messages": history, 
        "temperature": generation_config.get("temperature"),
        "max_tokens": generation_config.get("max_output_tokens"),
        "n": generation_config.get("candidate_count"),
        "stream": False,
    }
    
    if system_instruction:
        data["messages"].insert(0, {"role": "system", "content": system_instruction})

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(request_url, headers=headers, json=data) as response:
                response_text = await response.text()

                if response.status != 200:
                    raise Exception(f"API request failed with status {response.status}: {response_text}")

                response_data = json.loads(response_text)
                
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    content = response_data['choices'][0]['message']['content']
                else:
                    print(f"API did not return 'choices'. Full response: {response_data}")
                    return f"An error occurred with the API: No 'choices' returned. Response: {response_data}"

                prompt = "".join([item['content'] for item in history])
                cost_count(prompt, content, model_name)
                
                return content
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        import traceback
        traceback.print_exc()
        raise


@LLMRegistry.register('GeminiChat')
class GeminiChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Union[Message, dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> str:
        
        generation_config = {
            "temperature": temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
            "max_output_tokens": max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS,
            "candidate_count": num_comps if num_comps is not None else self.DEFUALT_NUM_COMPLETIONS,
        }
        
        system_instruction = None
        history = []
        
        processed_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                processed_messages.append(Message(role=msg.get('role'), content=msg.get('content')))
            else:
                processed_messages.append(msg)

        for msg in processed_messages:
            if msg.role == "system":
                system_instruction = msg.content
                continue
            
            history.append({"role": msg.role, "content": msg.content})

        response = await agemini_chat(
            model_name=self.model_name,
            history=history,
            system_instruction=system_instruction,
            generation_config=generation_config
        )

        return response

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> str:
        pass