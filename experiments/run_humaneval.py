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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from Topodim.graph.graph import Graph
from Topodim.tools.reader.readers import JSONLReader
from Topodim.tools.coding.python_executor import PyExecutor
from Topodim.utils.globals import Time
from Topodim.utils.const import Topodim_ROOT
from Topodim.utils.globals import Cost, PromptTokens, CompletionTokens

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
    parser = argparse.ArgumentParser(description="Topodim Experiments on HumanEval")
    parser.add_argument("--dataset_json", type=str, default="datasets_/humaneval/humaneval-py.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-oss:120b")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star','hetero'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.01,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--num_rounds',type=int,default=2,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default = 10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="humaneval",help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalWriteCode',
                        help='The decison method of the Topodim')
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    parser.add_argument('--diversity_weight', type=float, default=0.5,
                        help='Weight for diversity reward added to utility (recommended 0.3-0.8)')
    parser.add_argument('--entropy_weight', type=float, default=0.01,
                        help='Weight for entropy regularization (recommended 0.01-0.05)')

    parser.add_argument('--baseline_momentum', type=float, default=0.9,
                        help='EMA momentum for baseline reward (0.8-0.95 recommended)')

    parser.add_argument('--use_temperature_annealing', action='store_true',
                        help='Enable temperature annealing during training')
    parser.add_argument('--init_temperature', type=float, default=2.0,
                        help='Initial temperature for annealing (exploration)')
    parser.add_argument('--final_temperature', type=float, default=0.5,
                        help='Final temperature for annealing (exploitation)')



    args = parser.parse_args()
    result_path = Topodim_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args

async def main():
    args = parse_args()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()
    Cost.instance().reset()

    result_file = None
    dataset = JSONLReader.parse_file(args.dataset_json)
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{Topodim_ROOT}/result/eval")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.llm_name}_{current_time}.json"
    
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(args.mode,len(agent_names))
    graph = Graph(domain="humaneval",
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
    print("\n" + "=" * 80)
    print("HumanEval Training (with diversity and entropy regularization)")
    print("=" * 80)
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Diversity Weight: {args.diversity_weight}")
    print(f"  - Entropy Weight: {args.entropy_weight}")
    print(f"  - Temperature Annealing: {args.use_temperature_annealing}")
    if args.use_temperature_annealing:
        print(f"  - Temperature Range: [{args.final_temperature}, {args.init_temperature}]")
    print("=" * 80 + "\n")

    for i_batch in range(num_batches):
        print(f"{'='*80}")
        print(f"Batch {i_batch}")
        print(f"{'='*80}")
        start_ts = time.time()
        if args.use_temperature_annealing:
            if i_batch < args.num_iterations:
                temperature = get_temperature(
                    i_batch, args.num_iterations, args.init_temperature, args.final_temperature
                )
            else:
                temperature = args.final_temperature
        else:
            temperature = 1.0
        print(f"🌡️  Temperature: {temperature:.3f}")
        answer_log_probs = []
        tests = []
        realized_graphs = []
        input_dicts = []
        
        current_batch = dataloader(dataset,args.batch_size,i_batch)
        if current_batch is None:
            print("No more data available.")
            break
        
        for i_record, record in enumerate(current_batch):
            realized_graph = copy.deepcopy(graph)
            realized_graph.rgcn = graph.rgcn
            realized_graph.temperature = temperature
            task = record["prompt"]
            test = record["test"]
            tests.append(test)
            input_dict = {"task": task}
            input_dicts.append(input_dict)
            realized_graphs.append(realized_graph)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds)))
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs, diversity_scores, entropy_scores = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        total_rewards: List[float] = []
        data = load_result(result_file)
        answers_text: List[str] = []
        solved_flags: List[bool] = []
        
        for record, raw_answer, log_prob, test, div_score, ent_score in zip(
            current_batch, raw_answers, log_probs, tests, diversity_scores, entropy_scores
        ):
            if not isinstance(raw_answer,list):
                raise TypeError(f"Expected a list for the answer, but got {type(raw_answer).__name__}")
            answer = raw_answer[0].lstrip("```python\n").rstrip("\n```").strip()
            answers_text.append(answer)
            is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
            solved_flags.append(bool(is_solved))
            total_solved += is_solved
            total_executed += 1
            accuracy = total_solved / total_executed
            utility = float(is_solved)
            utilities.append(utility)
            diversity_term = float(div_score.detach().item()) if hasattr(div_score, "detach") else float(div_score)
            total_reward = utility + args.diversity_weight * diversity_term
            total_rewards.append(total_reward)

        batch_mean_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        if baseline_reward is None:
            baseline_reward = batch_mean_reward
        else:
            baseline_reward = args.baseline_momentum * baseline_reward + \
                              (1 - args.baseline_momentum) * batch_mean_reward
        for total_reward, log_prob, ent_score in zip(total_rewards, log_probs, entropy_scores):
            advantage = total_reward - baseline_reward
            policy_loss = -log_prob * advantage
            entropy_reg = -args.entropy_weight * ent_score
            single_loss = policy_loss + entropy_reg
            loss_list.append(single_loss)

        with open(result_file, 'w',encoding='utf-8') as file:
            for record, ans, is_ok in zip(current_batch, answers_text, solved_flags):
                data.append({
                    "Question": record["prompt"],
                    "Tests": record["test"],
                    "Attempt answer": ans,
                    "Solved": bool(is_ok),
                    "Solution": ans,
                    "Total solved": total_solved,
                    "Total executed": total_executed,
                    "Accuracy": accuracy
                })
            json.dump(data, file, indent=4)

        total_loss = torch.mean(torch.stack(loss_list))
        if args.optimized_spatial or args.optimized_temporal:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        mean_util = sum(utilities)/len(utilities) if utilities else 0.0
        diversity_vals = [(float(d.item()) if hasattr(d, "item") else float(d)) for d in diversity_scores]
        mean_div = sum(diversity_vals) / len(diversity_vals)
        entropy_vals = [(float(e.item()) if hasattr(e, "item") else float(e)) for e in entropy_scores]
        mean_ent = sum(entropy_vals) / len(entropy_vals)
        mean_tr = sum(total_rewards)/len(total_rewards) if total_rewards else 0.0
        advantages_vals = [tr - baseline_reward for tr in total_rewards]
        diversity_str = [f"{v:.3f}" for v in diversity_vals]

        print(f"\n📊 Results:")
        print(f"  Batch time: {time.time() - start_ts:.2f}s")
        print(f"  Utilities (Task Reward): {[f'{u:.3f}' for u in utilities]} (mean={mean_util:.3f})")
        print(f"  Diversity Scores: {diversity_str} (mean={mean_div:.3f})")
        print(f"  Total Rewards: {[f'{r:.3f}' for r in total_rewards]} (mean={mean_tr:.3f})")
        print(f"  Baseline (Reward): {baseline_reward:.3f}")
        print(f"  Advantages: {[f'{a:.3f}' for a in advantages_vals]}")
        print(f"  Entropy: {mean_ent:.3f} (weight={args.entropy_weight})")
        print(f"  loss: {float(total_loss.item()):.3f}")

        print(f"\n{'='*80}")
        print(f"🎯 Detailed Predictions:")
        print(f"{'='*80}")
        for i, (inp, ans, solved, realized_graph) in enumerate(zip(input_dicts, answers_text, solved_flags, realized_graphs)):
            status = "✅" if solved else "❌"
            print(f"\n  {status} Sample {i}:")
            print(f"     Question: {inp['task'][:120]}...")
            print(f"     Topology Execution Order:")
            topo_summary = realized_graph.get_execution_summary()
            for line in topo_summary.split('\n'):
                print(f"       {line}")
            print(f"     Passed: {bool(solved)}")

        print(f"\n💰 Cost:")
        print(f"  Total Cost: ${Cost.instance().value:.4f}")
        print(f"  Prompt Tokens: {int(PromptTokens.instance().value):,}")
        print(f"  Completion Tokens: {int(CompletionTokens.instance().value):,}")

        if i_batch+1 == args.num_iterations:
            args.optimized_spatial = False
            args.optimized_temporal = False
            total_solved = 0
            total_executed = 0
            graph.rgcn.eval()
            print("Start Eval")
            
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")
        print("utilities:", utilities)
        print("loss:", total_loss.item())

        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    print("\n" + "="*60)
    print("Token Usage Summary:")
    print("="*60)
    print(f"Total Prompt Tokens:     {int(PromptTokens.instance().value):,}")
    print(f"Total Completion Tokens: {int(CompletionTokens.instance().value):,}")
    print(f"Total Tokens:            {int(PromptTokens.instance().value + CompletionTokens.instance().value):,}")
    if Cost.instance().value > 0:
        print(f"Total Cost:              ${Cost.instance().value:.4f}")
    print("="*60)



def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']],
               N:int):
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
        node_kwargs = [ {"llm_name": "gpt-oss:20b"},
                        {"llm_name": "gpt-oss:120b"}, 
                        {"llm_name": "gpt-oss:20b"}, 
                        {"llm_name": "gpt-oss:120b"},
                        {"llm_name": "gpt-oss:20b"}]


    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == '__main__':
    asyncio.run(main())
