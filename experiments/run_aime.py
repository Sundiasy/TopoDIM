import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import copy
from typing import List,Union,Literal
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from Topodim.utils.const import Topodim_ROOT
from Topodim.graph.graph import Graph
from Topodim.tools.reader.readers import JSONLReader
from Topodim.utils.globals import Time
from Topodim.utils.globals import Cost, PromptTokens, CompletionTokens
from datasets_.gsm8k_dataset import gsm_data_process,gsm_get_predict

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_temperature(iteration: int, total_iterations: int, init_temp: float = 2.0, final_temp: float = 0.5) -> float:
    if total_iterations <= 0:
        return final_temp
    progress = max(0.0, min(1.0, iteration / total_iterations))
    return init_temp - (init_temp - final_temp) * progress
    
def parse_args():
    parser = argparse.ArgumentParser(description="Topodim Experiments on aime")
    parser.add_argument("--dataset_json", type=str, default="datasets_/aime/aime_23_25.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-oss:20b")  # gpt-oss:120b 
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star','hetero'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.01,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default=10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="gsm8k",help="Domain (the same as dataset name), default 'gsm8k'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the Topodim')
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')

    parser.add_argument('--diversity_weight', type=float, default=0.8)
    parser.add_argument('--entropy_weight', type=float, default=0.01)
    parser.add_argument('--baseline_momentum', type=float, default=0.9)
    parser.add_argument('--use_temperature_annealing', action='store_true')
    parser.add_argument('--init_temperature', type=float, default=2.0)
    parser.add_argument('--final_temperature', type=float, default=0.5)

    args = parser.parse_args()
    result_path = Topodim_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args

async def main():
    args = parse_args()
    result_file = None
    dataset = JSONLReader.parse_file(args.dataset_json)
    dataset = gsm_data_process(dataset)
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{Topodim_ROOT}/result/gsm8k")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(args.mode,len(agent_names))
    graph = Graph(domain="gsm8k",
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    graph.rgcn.train()
    optimizer = torch.optim.Adam(graph.rgcn.parameters(), lr=args.lr)   
    
    num_batches = int(len(dataset)/args.batch_size)
    total_solved, total_executed = (0, 0)
    
    baseline_reward = None

    for i_batch in range(num_batches):
        print(f"Batch {i_batch}",80*'-')
        start_ts = time.time()

        if args.use_temperature_annealing:
            temperature = get_temperature(i_batch, args.num_iterations, args.init_temperature, args.final_temperature) \
                if i_batch < args.num_iterations else args.final_temperature
        else:
            temperature = 1.0
        print(f"🌡️  Temperature: {temperature:.3f}")

        answer_log_probs = []
        answers = []
        
        current_batch = dataloader(dataset,args.batch_size,i_batch)
        if current_batch is None:
            print("No more data available.")
            break
        
        answer_tasks = []
        realized_graphs = []
        input_dicts = []
        for i_record, record in enumerate(current_batch):
            realized_graph = copy.deepcopy(graph)
            realized_graph.rgcn = graph.rgcn
            realized_graph.temperature = temperature 
            task = record["task"]
            step = record["step"]
            answer = record["answer"]
            answers.append(answer)
            input_dict = {"task": task}
            answer_tasks.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds)))
            realized_graphs.append(realized_graph)
            input_dicts.append(input_dict)
        
        raw_results = await asyncio.gather(*answer_tasks)
        raw_answers, log_probs, diversity_scores, entropy_scores = zip(*raw_results)
        
        answers_text: List[str] = []
        for ra in raw_answers:
            answers_text.append(ra)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        for task, answer, true_answer, step in zip(current_batch, raw_answers, answers, [r["step"] for r in current_batch]):
            predict_answer = gsm_get_predict(answer[0])
            is_solved = float(predict_answer)==float(true_answer)
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            accuracy = total_solved/ total_executed
            utility = 1.0 if is_solved else 0.0
            utilities.append(utility)
        
        diversity_vals = [(float(d.item()) if hasattr(d, "item") else float(d)) for d in diversity_scores]
        entropy_vals = [(float(e.item()) if hasattr(e, "item") else float(e)) for e in entropy_scores]
        
        total_rewards = [u + args.diversity_weight * d for u, d in zip(utilities, diversity_vals)]
        batch_mean_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        
        if baseline_reward is None:
            baseline_reward = batch_mean_reward
        else:
            baseline_reward = args.baseline_momentum * baseline_reward + (1 - args.baseline_momentum) * batch_mean_reward

        for tr, lp, ent in zip(total_rewards, log_probs, entropy_scores):
            advantage = tr - baseline_reward
            policy_loss = -lp * advantage
            entropy_reg = -args.entropy_weight * ent
            loss_list.append(policy_loss + entropy_reg)

        total_loss = torch.mean(torch.stack(loss_list)) if loss_list else torch.tensor(0.0)

        if args.optimized_spatial or args.optimized_temporal:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")
        print(f"Utilities: {[f'{u:.2f}' for u in utilities]}")
        print(f"Diversity: {[f'{d:.2f}' for d in diversity_vals]}")
        print(f"Total Rewards: {[f'{r:.2f}' for r in total_rewards]}")
        print(f"Baseline: {baseline_reward:.3f}")
        print("loss:", total_loss.item())
        for i, (inp, ans, solved, realized_graph) in enumerate(zip(input_dicts, answers_text, utilities, realized_graphs)):
            status = '✅' if solved else '❌'
            print(f"\n  {status} Sample {i}:")
            print(f"     Question: {inp['task'][:120]}...")
            print(f"     Topology Execution Order:")
            
            topo_summary = realized_graph.get_execution_summary()
            for line in topo_summary.split('\n'):
                print(f"       {line}")
            print(f"     Passed: {bool(solved)}")        

        if i_batch+1 == args.num_iterations:
            args.optimized_spatial = False
            args.optimized_temporal = False
            total_solved = 0
            total_executed = 0
            graph.rgcn.eval()
            print("Start Eval")
            
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")


def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']]
               ,N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Programming Expert'}]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'hetero':
        node_kwargs = [ {"llm_name": "gemma3:12b"},
                        {"llm_name": "gpt-oss:20b"}, 
                        {"llm_name": "gemma3:12b"}, 
                        {"llm_name": "gpt-oss:20b"},
                        {"llm_name": "gemma3:12b"}]  
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs} 

if __name__ == '__main__':
    asyncio.run(main())