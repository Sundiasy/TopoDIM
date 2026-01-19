from typing import List,Any,Dict

from Topodim.graph.node import Node
from Topodim.agents.agent_registry import AgentRegistry
from Topodim.llm.llm_registry import LLMRegistry
from Topodim.prompt.prompt_set_registry import PromptSetRegistry
from Topodim.tools.coding.python_executor import PyExecutor
from Topodim.llm.format import Message
@AgentRegistry.register('CodeWriting')
class CodeWriting(Node):
    def __init__(self, id: str | None =None, role:str = None ,domain: str = "", llm_name: str = "",):
        super().__init__(id, "CodeWriting" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role) 
        
    async def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], query_info:Dict[str,Dict], debate_info:Dict[str,Dict], evaluation_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """             
        system_prompt = self.constraint

        if self.role == 'Fake':
            user_prompt = self.prompt_set.get_adversarial_answer_prompt(raw_inputs['task'])
            return system_prompt, user_prompt
        
        user_prompt = f"The task is:\n\n{raw_inputs['task']}\n"
        
        spatial_str = ""
        temporal_str = ""
        query_str = ""
        debate_str = ""
        evaluation_str = ""
        
        for id, info in spatial_info.items():
            if info['output'].startswith("```python") and info['output'].endswith("```") and self.role != 'Normal Programmer' and self.role != 'Stupid Programmer' and self.role != 'Fake':
                output = info['output'].lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                if is_solved and len(self.internal_tests):
                    return "is_solved", info['output']
                spatial_str += f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"

        for id, info in temporal_info.items():
            if info['output'].startswith("```python") and info['output'].endswith("```") and self.role != 'Normal Programmer' and self.role != 'Stupid Programmer' and self.role != 'Fake':
                output = info['output'].lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                if is_solved and len(self.internal_tests):
                    return "is_solved", info['output']
                temporal_str += f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"

        for id, info in query_info.items():
            query_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"

        for id, info in debate_info.items():
            debate_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"

        for id, info in evaluation_info.items():
            evaluation_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"

        user_prompt += (
            "\n\nPeers evaluated your code. You can refer to their feedback critically and improve your implementation:\n\n"
            f"{evaluation_str}\n\n"
        ) if len(evaluation_str) else ""
        user_prompt += (
            "\n\nEvaluate the provided code from another agent. The evaluated agent's code is provided below:\n\n"
            f"{query_str}\n\n"
        ) if len(query_str) else ""
        user_prompt += (
            "\n\nPeers entered a debate to stress-test your code assumptions. Their feedback is as follows:\n\n"
            f"{debate_str}\n\n"
        ) if len(debate_str) else ""
        user_prompt += (
            "\n\nAt the same time, the outputs and feedbacks of other agents are as follows:\n\n"
            f"{spatial_str}\n\n"
        ) if len(spatial_str) else ""
        user_prompt += (
            "\n\nIn the last round of dialogue, the outputs and feedbacks of some agents were:\n\n"
            f"{temporal_str}"
        ) if len(temporal_str) else ""
        
        return system_prompt, user_prompt

    def extract_example(self, prompt: str) -> list:
        prompt = prompt['task']
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results
    
    def _execute(self, input:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], query_info:Dict[str,Any], debate_info:Dict[str,Any], evaluation_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        self.internal_tests = self.extract_example(input)
        # Note: _process_inputs is now async, but this sync version is kept for compatibility
        import asyncio
        loop = asyncio.get_event_loop()
        system_prompt, user_prompt = loop.run_until_complete(
            self._process_inputs(input, spatial_info, temporal_info, query_info, debate_info, evaluation_info)
        )
        messages = [Message(role='system', content=system_prompt), Message(role='user', content=user_prompt)]
        response = self.llm.gen(messages)
        return response

    async def _async_execute(self, input:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], query_info:Dict[str,Any], debate_info:Dict[str,Any], evaluation_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        self.internal_tests = self.extract_example(input)
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info, query_info, debate_info, evaluation_info)
        ## test
        if system_prompt == "is_solved":
            return user_prompt
        messages = [Message(role='system', content=system_prompt), Message(role='user', content=user_prompt)]
        response = await self.llm.agen(messages)
        print(f"################system prompt:{system_prompt}")
        print(f"################user prompt:{user_prompt}")
        print(f"################response:{response}")
        return response