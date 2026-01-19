import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio

from Topodim.graph.node import Node
from Topodim.agents.agent_registry import AgentRegistry
from Topodim.prompt.prompt_set_registry import PromptSetRegistry
from Topodim.llm.profile_embedding import get_sentence_embedding
from Topodim.gnn.rgcn import GraphGenerationPolicy
from torch_geometric.utils import dense_to_sparse

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        
        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role_adj_matrix,self.role_relation = self.construct_relation_adj_matrix()
        self.features = self.construct_features()
        self.rgcn = GraphGenerationPolicy(
                    num_nodes=len(self.agent_names), 
                    node_in_dim=16, 
                    hid_dim=16,
                    num_base_rels=3, 
                    num_predict_rels=3,
                    prune_strategy='balanced_top_k', 
                    edges_per_type=2,                
                    keep_start_node=True,
                )
            
    def construct_relation_adj_matrix(self):
        role_connections: List[Tuple[str, str, str]] = self.prompt_set.get_role_connection()

        role_to_ids: Dict[str, List[int]] = {}
        all_node_roles = {self.nodes[node_id].role for node_id in self.nodes}
        
        for role in all_node_roles:
            role_to_ids[role] = []

        sorted_node_ids = sorted(self.nodes.keys())
        for i, node_id in enumerate(sorted_node_ids):
            role = self.nodes[node_id].role
            if role in role_to_ids:
                role_to_ids[role].append(i)

        relation_to_idx: Dict[str, int] = {}
        all_relation_types = sorted(list(set(conn[1] for conn in role_connections)))
        for relation_type in all_relation_types:
            if relation_type not in relation_to_idx:
                relation_to_idx[relation_type] = len(relation_to_idx)
        
        edge_list = []
        edge_type_list = []

        for source_role, relation, target_role in role_connections:
            source_node_indices = role_to_ids.get(source_role, [])
            target_node_indices = role_to_ids.get(target_role, [])
            relation_idx = relation_to_idx[relation]
            
            for source_idx in source_node_indices:
                for target_idx in target_node_indices:
                    edge_list.append([source_idx, target_idx])
                    edge_type_list.append(relation_idx)
        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)

        return edge_index, edge_type
    
    def construct_features(self):
        features = []
        for node_id in self.nodes:
            role = self.nodes[node_id].role
            profile = self.prompt_set.get_description(role)
            feature = get_sentence_embedding(profile)
            features.append(feature)
        features = torch.tensor(np.array(features))
        return features
    
    def construct_new_features(self, query):
        query_embedding = torch.tensor(get_sentence_embedding(query))
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_nodes,1))
        new_features = torch.cat((self.features,query_embedding),dim=1)
        return new_features
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        # node_kwargs can contain per-node overrides, e.g. {'llm_name': 'gemini-1.5-pro-latest'}
        for agent_name, raw_kwargs in zip(self.agent_names, self.node_kwargs):
            # Work on a shallow copy to avoid mutating the provided list in-place
            kwargs = {} if raw_kwargs is None else dict(raw_kwargs)
            if agent_name in AgentRegistry.registry:
                # only set domain/llm_name defaults if not already provided per-node
                kwargs.setdefault("domain", self.domain)
                # allow per-node llm_name override; fall back to graph-level llm_name
                if "llm_name" not in kwargs or kwargs.get("llm_name") in (None, ""):
                    kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []

    def clear_query_connection(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].query_predecessors = []
            self.nodes[node_id].query_successors = []

    def clear_debate_connection(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].debate_predecessors = []
            self.nodes[node_id].debate_successors = []



    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        
        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f" of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                    num_rounds:int = 3, 
                    max_tries: int = 3, 
                    max_time: int = 600,) -> List[Any]:
            log_probs = 0
            
            task_query = input.get('task', '') 
            if not task_query:
                raise ValueError("Input dictionary must contain a 'task' key with the query string.")
            
            task_embedding = torch.tensor(get_sentence_embedding(task_query))

            temperature = getattr(self, 'temperature', 1.0)

            total_diversity_score = torch.tensor(0.0)
            total_entropy = torch.tensor(0.0)
            
            self.execution_info = {
                'node_execution_order': [], 
                'node_roles': {},
                'node_outputs': {},  
                'adjacency_info': [] 
            }

            for round in range(num_rounds):
                self.clear_spatial_connection()
                self.clear_debate_connection()
                self.clear_query_connection()

                sorted_node_ids = sorted(self.nodes.keys())
                id_to_idx = {node_id: i for i, node_id in enumerate(sorted_node_ids)}

                adj_matrices, total_log_prob_edges, diversity_score, avg_entropy = self.rgcn(
                    task_embedding, self.role_adj_matrix, self.role_relation, temperature=temperature
                )
                log_probs += total_log_prob_edges
                total_diversity_score += diversity_score
                total_entropy += avg_entropy 


                self.execution_info['adjacency_info'].append({
                    'round': round,
                    'spatial': adj_matrices[0].detach().cpu().numpy().tolist(),
                    'query': adj_matrices[1].detach().cpu().numpy().tolist(),
                    'debate': adj_matrices[2].detach().cpu().numpy().tolist(),
                })
                

                active_nodes_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
                for rel_idx in range(adj_matrices.shape[0]): 
                    has_outgoing = adj_matrices[rel_idx].sum(dim=1) > 0.5
                    has_incoming = adj_matrices[rel_idx].sum(dim=0) > 0.5 
                    active_nodes_mask |= (has_outgoing | has_incoming)

                active_node_indices = torch.where(active_nodes_mask)[0].tolist()
                if len(active_node_indices) < self.num_nodes:

                    sorted_node_ids = [sorted_node_ids[i] for i in active_node_indices]
                    id_to_idx = {node_id: i for i, node_id in enumerate(sorted_node_ids)}

                    active_indices_tensor = torch.tensor(active_node_indices, dtype=torch.long)
                    adj_matrices = adj_matrices[:, active_indices_tensor, :][:, :, active_indices_tensor]
                

                forward_adj_matrix = (adj_matrices[0] + adj_matrices[2]) > 0.5 
                num_active_nodes = len(sorted_node_ids)
                for i in range(num_active_nodes):
                    for j in range(num_active_nodes):
                        if forward_adj_matrix[i, j]:
                            out_node = self.nodes[sorted_node_ids[i]]
                            in_node = self.nodes[sorted_node_ids[j]]
                            out_node.add_successor(in_node, 'spatial')
                        if adj_matrices[1][i,j]:
                            out_node = self.nodes[sorted_node_ids[i]]
                            in_node = self.nodes[sorted_node_ids[j]]
                            out_node.add_successor(in_node, 'query')
                        if adj_matrices[2][i,j]:
                            out_node = self.nodes[sorted_node_ids[i]]
                            in_node = self.nodes[sorted_node_ids[j]]
                            out_node.add_successor(in_node, 'debate')

                in_degree = {node_id: len(self.nodes[node_id].spatial_predecessors) for node_id in sorted_node_ids}
                zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

                while zero_in_degree_queue:
                    current_node_id = zero_in_degree_queue.pop(0)
                    current_node = self.nodes[current_node_id]
                    current_node_idx = id_to_idx[current_node_id]
                    
                    if current_node_id not in self.execution_info['node_roles']:
                        self.execution_info['node_roles'][current_node_id] = current_node.role
                    self.execution_info['node_execution_order'].append({
                        'round': round,
                        'node_id': current_node_id,
                        'node_role': current_node.role,
                        'topo_position': len(self.execution_info['node_execution_order'])
                    })

                    await self._execute_node_with_retry(current_node, input, max_tries, max_time)
                    
                    if current_node_id not in self.execution_info['node_outputs']:
                        self.execution_info['node_outputs'][current_node_id] = []
                    self.execution_info['node_outputs'][current_node_id].append({
                        'round': round,
                        'output': current_node.outputs[-1] if current_node.outputs else None
                    })

                    for successor_idx in range(num_active_nodes):
                        successor_node = self.nodes[sorted_node_ids[successor_idx]]
                        
                        if adj_matrices[1, current_node_idx, successor_idx] == 1:
                            await self._execute_node_with_retry(successor_node, input, max_tries, max_time, mode=1, pre_node=current_node.role)
                            await self._execute_node_with_retry(current_node, input, max_tries, max_time, mode=3, pre_node=successor_node.role)

                        elif adj_matrices[2, current_node_idx, successor_idx] == 1:
                            await self._execute_node_with_retry(successor_node, input, max_tries, max_time)
                            await self._execute_node_with_retry(current_node, input, max_tries, max_time, mode=2, pre_node=successor_node.role)

                    for successor_node in current_node.spatial_successors:
                        in_degree[successor_node.id] -= 1
                        if in_degree[successor_node.id] == 0:
                            zero_in_degree_queue.append(successor_node.id)

                self.update_memory()

            self.connect_decision_node()
            await self.decision_node.async_execute(input)
            final_answers = self.decision_node.outputs
            
            if len(final_answers) == 0:
                final_answers.append("No answer of the decision node")

            self.execution_info['final_answer'] = final_answers[0] if final_answers else None
            self.execution_info['question'] = task_query

            avg_diversity_score = total_diversity_score / num_rounds
            avg_entropy_score = total_entropy / num_rounds

            return final_answers, log_probs, avg_diversity_score, avg_entropy_score

    async def _execute_node_with_retry(self, node: Node, input: Dict[str, str], max_tries: int, max_time: int, mode: int = 0, pre_node: any = None):
        tries = 0
        while tries < max_tries:
            try:
                if pre_node is None:
                    await asyncio.wait_for(node.async_execute(input, mode), timeout=max_time)
                else:
                    await asyncio.wait_for(node.async_execute(input, mode, pre_node=pre_node), timeout=max_time)
                break
            except Exception as e:
                print(f"Error during execution of node {node.id} (try {tries+1}/{max_tries}): {e}")
                tries += 1
                if tries == max_tries:
                    print(f"Failed to execute node {node.id} after {max_tries} attempts.")
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks
    
    def get_execution_summary(self) -> str:
        if not hasattr(self, 'execution_info'):
            return "No execution info available"
        
        exec_info = self.execution_info
        summary_lines = []

        rounds_dict: Dict[int, List[str]] = {}
        for exec_item in exec_info.get('node_execution_order', []):
            round_num = exec_item['round']
            if round_num not in rounds_dict:
                rounds_dict[round_num] = []
            rounds_dict[round_num].append((exec_item['topo_position'], exec_item['node_id']))

        for round_num in sorted(rounds_dict.keys()):
            # sort by topo position
            ordered = [node_id for _, node_id in sorted(rounds_dict[round_num], key=lambda x: x[0])]

            # build a line that includes edge types between consecutive nodes
            parts: List[str] = []
            for i, node_id in enumerate(ordered):
                node = self.nodes.get(node_id)
                role = node.role if node is not None else 'Unknown'
                parts.append(f"{role}({node_id})")
                # annotate edge type to the next node if exists
                if i + 1 < len(ordered):
                    next_id = ordered[i + 1]
                    next_node = self.nodes.get(next_id)
                    edge_types = []
                    if node is not None and next_node is not None:
                        if next_node in node.spatial_successors:
                            edge_types.append('spatial')
                        if next_node in node.query_successors:
                            edge_types.append('query')
                        if next_node in node.debate_successors:
                            edge_types.append('debate')
                    if edge_types:
                        parts.append(f"- [{' ,'.join(edge_types)}] ->")
                    else:
                        parts.append("->")

            # Additionally, list non-consecutive edges (extra connections) for this round
            extra_edges: List[str] = []
            for src in ordered:
                src_node = self.nodes.get(src)
                if src_node is None:
                    continue
                for succ in src_node.spatial_successors:
                    if succ.id in ordered and ordered.index(succ.id) == ordered.index(src) + 1:
                        # already represented as consecutive
                        continue
                    if succ.id in ordered:
                        extra_edges.append(f"{src_node.role}({src}) -> {succ.role}({succ.id}) [spatial]")
                for succ in src_node.query_successors:
                    if succ.id in ordered and ordered.index(succ.id) == ordered.index(src) + 1:
                        continue
                    if succ.id in ordered:
                        extra_edges.append(f"{src_node.role}({src}) -> {succ.role}({succ.id}) [query]")
                for succ in src_node.debate_successors:
                    if succ.id in ordered and ordered.index(succ.id) == ordered.index(src) + 1:
                        continue
                    if succ.id in ordered:
                        extra_edges.append(f"{src_node.role}({src}) -> {succ.role}({succ.id}) [debate]")

            line = f"Round {round_num}: {' '.join(parts)}"
            if extra_edges:
                line += " | Extra: " + ", ".join(extra_edges)
            summary_lines.append(line)

        return '\n'.join(summary_lines) if summary_lines else "No topology info"

def min_max_norm(tensor:torch.Tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1
    