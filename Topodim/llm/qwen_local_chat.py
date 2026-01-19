import asyncio
import ollama
from typing import List, Union, Optional, Dict, Any
from tenacity import retry, wait_random_exponential, stop_after_attempt

import tiktoken

from Topodim.llm.format import Message
from Topodim.llm.llm import LLM
from Topodim.llm.llm_registry import LLMRegistry
from Topodim.utils.globals import PromptTokens, CompletionTokens


# =========================================================
# Token utilities
# =========================================================

def cal_token_qwen(text: str) -> int:
    """
    Approximate token count for Qwen.
    """
    try:
        encoder = tiktoken.encoding_for_model("gpt-4.1")
        return len(encoder.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)


def token_count_qwen(prompt: str, response: str):
    prompt_len = cal_token_qwen(prompt)
    completion_len = cal_token_qwen(response)

    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len

    return prompt_len, completion_len


# =========================================================
# Global Ollama resources (CRITICAL)
# =========================================================

# 1. Single shared AsyncClient (prevents FD / socket leak)
_OLLAMA_CLIENT = ollama.AsyncClient()

# 2. Concurrency limiter (Ollama is NOT high-concurrency safe)
#    Empirically: 1–2 is safest for qwen / qwen3
_OLLAMA_SEMAPHORE = asyncio.Semaphore(1)


@retry(
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _ollama_chat(
    model_name: str,
    history: List[Dict[str, str]],
    generation_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Low-level Ollama call with:
    - retry
    - concurrency control
    - strict error semantics
    """
    async with _OLLAMA_SEMAPHORE:
        response = await _OLLAMA_CLIENT.chat(
            model=model_name,
            messages=history,
            options={
                "temperature": generation_config["temperature"],
                "num_predict": generation_config["max_output_tokens"],
            },
        )

    # if not isinstance(response, dict):
    #     raise RuntimeError(f"Invalid Ollama response type: {type(response)}")

    return response


# =========================================================
# QwenLocalChat
# =========================================================

@LLMRegistry.register("QwenLocalChat")
class QwenLocalChat(LLM):
    """
    Stable local Qwen chat wrapper for long-running experiments.
    """

    def __init__(self, model_name: str = "qwen"):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Union[Message, dict]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> str:

        generation_config = {
            "temperature": (
                temperature if temperature is not None else self.DEFAULT_TEMPERATURE
            ),
            "max_output_tokens": (
                max_tokens if max_tokens is not None else self.DEFAULT_MAX_TOKENS
            ),
        }

        history: List[Dict[str, str]] = []
        prompt_text_parts: List[str] = []

        # -----------------------------------------------------
        # Build history safely
        # -----------------------------------------------------
        no_think_injected = False

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = msg.role
                content = msg.content

            # Inject /no_think ONLY ONCE and ONLY in system
            if (
                self.model_name.startswith("qwen3")
                and role == "system"
                and not no_think_injected
            ):
                content = content.rstrip() + "\n/no_think"
                no_think_injected = True

            history.append({"role": role, "content": content})
            prompt_text_parts.append(content)

        prompt_text = "\n".join(prompt_text_parts)

        # -----------------------------------------------------
        # Call Ollama
        # -----------------------------------------------------
        response_dict = await _ollama_chat(
            model_name=self.model_name,
            history=history,
            generation_config=generation_config,
        )

        # content = response_dict.get('message', {}).get('content', '')
        message = response_dict.get("message")
        if not message or "content" not in message:
            raise RuntimeError(f"Malformed Ollama response: {response_dict}")

        content = message["content"]

        # -----------------------------------------------------
        # Token accounting (prefer Ollama native stats)
        # -----------------------------------------------------
        if "prompt_eval_count" in response_dict and "eval_count" in response_dict:
            PromptTokens.instance().value += response_dict["prompt_eval_count"]
            CompletionTokens.instance().value += response_dict["eval_count"]
        else:
            token_count_qwen(prompt_text, content)

        return content

    def gen(self, *args, **kwargs):
        raise NotImplementedError(
            "Use agen() for QwenLocalChat (async only)."
        )
