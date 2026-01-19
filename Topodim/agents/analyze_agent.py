from typing import List,Any,Dict
import re
from Topodim.llm.format import Message
from Topodim.graph.node import Node
from Topodim.agents.agent_registry import AgentRegistry
from Topodim.llm.llm_registry import LLMRegistry
from Topodim.prompt.prompt_set_registry import PromptSetRegistry
from Topodim.tools.search.wiki import search_wiki_main

def find_strings_between_pluses(text):
    return re.findall(r'\@(.*?)\@', text)

@AgentRegistry.register('AnalyzeAgent')
class AnalyzeAgent(Node):
    def __init__(self, id: str | None =None, role:str = None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "AnalyzeAgent" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.wiki_summary = ""

    async def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], query_info:Dict[str,Dict], debate_info:Dict[str,Dict], evaluation_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """              
        system_prompt = f"{self.constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n" if self.role != 'Fake' else self.prompt_set.get_adversarial_answer_prompt(raw_inputs['task'])
        spatial_str = ""
        temporal_str = ""
        query_str = ""
        debate_str = ""
        evaluation_str = ""
        for id, info in spatial_info.items():
            if self.role == 'Wiki Searcher' and info['role']=='Knowlegable Expert':
                queries = find_strings_between_pluses(info['output'])
                wiki = await search_wiki_main(queries)
                if len(wiki):
                    self.wiki_summary = ".\n".join(wiki)
                    user_prompt += f"The key entities of the problem are explained in Wikipedia as follows:{self.wiki_summary}"
            spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        for id, info in query_info.items():
            query_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        for id, info in debate_info.items():
            debate_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        for id, info in evaluation_info.items():
            evaluation_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"

        user_prompt += (
            "\nPeers evaluated your answers. You can refer to their response critically and determine the correct option of the task:\n\n"
            f"{evaluation_str}\n\n"
        ) if len(evaluation_str) else ""
        user_prompt += (
            "\nEvaluate the provided answer from another agent and determine the correct option of the task. The evaluated agent's response is provided below:\n\n"
            f"{query_str}\n\n"
        ) if len(query_str) else ""
        user_prompt += (
            "\nPeers entered a debate to stress-test your assumptions. Their answers are as follows:\n\n"
            f"{debate_str}\n\n"
        ) if len(debate_str) else ""
        user_prompt += (
            "\nAt the same time, the outputs of other agents are as follows:\n\n"
            f"{spatial_str}\n\n"
        ) if len(spatial_str) else ""
        user_prompt += (
            "\nIn the last round of dialogue, the outputs of other agents were:\n\n"
            f"{temporal_str}"
        ) if len(temporal_str) else ""
        return system_prompt, user_prompt

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], query_info:Dict[str,Dict], debate_info:Dict[str,Dict], evaluation_info:Dict[str,Dict], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, query_info, debate_info, evaluation_info)
        # message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        messages = [Message(role='system', content=system_prompt), Message(role='user', content=user_prompt)]
        response = self.llm.gen(messages)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], query_info:Dict[str,Dict], debate_info:Dict[str,Dict], evaluation_info:Dict[str,Dict], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = await self._process_inputs(input, spatial_info, temporal_info, query_info, debate_info, evaluation_info)
        # message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        # response = await self.llm.agen(message)
        messages = [Message(role='system', content=system_prompt), Message(role='user', content=user_prompt)]
        response = await self.llm.agen(messages)
        if self.wiki_summary != "":
            response += f"\n\n{self.wiki_summary}"
            self.wiki_summary = ""
        print(f"################system prompt:{system_prompt}")
        print(f"################user prompt:{user_prompt}")
        print(f"################response:{response}")
        return response