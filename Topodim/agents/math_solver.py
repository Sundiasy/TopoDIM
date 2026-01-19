from typing import List,Any,Dict
from Topodim.llm.format import Message
from Topodim.graph.node import Node
from Topodim.agents.agent_registry import AgentRegistry
from Topodim.llm.llm_registry import LLMRegistry
from Topodim.prompt.prompt_set_registry import PromptSetRegistry
from Topodim.tools.coding.python_executor import execute_code_get_return
from datasets_.gsm8k_dataset import gsm_get_predict

@AgentRegistry.register('MathSolver')
class MathSolver(Node):
    def __init__(self, id: str | None =None, role:str = None ,domain: str = "", llm_name: str = "",):
        super().__init__(id, "MathSolver" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role) 
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], query_info:Dict[str,Dict], debate_info:Dict[str,Dict], evaluation_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """             
        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""
        query_str = ""
        debate_str = ""
        evaluation_str = ""
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"],role=self.role)
        if self.role == "Math Solver":
            user_prompt += "(Hint: The answer is near to"
            for id, info in spatial_info.items():
                user_prompt += " "+gsm_get_predict(info["output"])
            for id, info in temporal_info.items():
                user_prompt += " "+gsm_get_predict(info["output"])
            user_prompt += ")."
        else:
            for id, info in spatial_info.items():
                spatial_str += f"Agent {id} as a {info['role']} his answer to this question is:\n\n{info['output']}\n\n"
            for id, info in temporal_info.items():
                temporal_str += f"Agent {id} as a {info['role']} his answer to this question was:\n\n{info['output']}\n\n"

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
                "\n\nEvaluate the provided code from anosther agent. The evaluated agent's code is provided below:\n\n"
                f"{query_str}\n\n"
            ) if len(query_str) else ""
            user_prompt += (
                "\n\nPeers entered a debate to stress-test your code assumptions. Their feedback is as follows:\n\n"
                f"{debate_str}\n\n"
            ) if len(debate_str) else ""
            user_prompt += f"\n\nAt the same time, there are the following responses to the same question for your reference:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
            user_prompt += f"\n\nIn the last round of dialogue, there were the following responses to the same question for your reference: \n\n{temporal_str}" if len(temporal_str) else ""
        return system_prompt, user_prompt
    
    def _execute(self, input:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], query_info:Dict[str,Any], debate_info:Dict[str,Any], evaluation_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, query_info, debate_info, evaluation_info)
        messages = [Message(role='system', content=system_prompt), Message(role='user', content=user_prompt)]
        response = self.llm.gen(messages)
        return response

    async def _async_execute(self, input:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], query_info:Dict[str,Any], debate_info:Dict[str,Any], evaluation_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, query_info, debate_info, evaluation_info)

        messages = [Message(role='system', content=system_prompt), Message(role='user', content=user_prompt)]
        response = await self.llm.agen(messages)
        if self.role == "Programming Expert":
            answer = execute_code_get_return(response.lstrip("```python\n").rstrip("\n```"))
            response += f"\nthe answer is {answer}"
        print(f"#################system_prompt:{system_prompt}")
        print(f"#################user_prompt:{user_prompt}")
        print(f"#################response:{response}")
        return response