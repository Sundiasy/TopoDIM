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
from typing import List, Union, Literal
import random
import numpy as np
class ProblemAdapter:
    def __init__(self, d: dict):
        self._d = d
        self.question_id = d.get('question_id') or d.get('QuestionID') or d.get('id')
        self.question_content = (
            d.get('question_content')
            or d.get('question')
            or d.get('prompt')
            or d.get('description')
            or ''
        )
        self._evaluation = d.get('evaluation') or d.get('evaluation_sample')
        self._tests = d.get('input_output') or d.get('tests') or d.get('test')

    def get_evaluation_sample(self):
        if isinstance(self._evaluation, dict):
            if "input_output" in self._evaluation or {"inputs", "outputs"} <= set(self._evaluation.keys()):
                return normalize_lcb_sample(self._evaluation)
        if self._tests is not None:
            return normalize_lcb_sample({"input_output": self._tests})

        pub = self._d.get("public_test_cases")
        priv = self._d.get("private_test_cases")
        metadata = self._d.get("metadata")
        if pub and priv:
            try:
                import json, zlib, pickle, base64
                pub_list = json.loads(pub)
                try:
                    priv_list = json.loads(priv)
                except Exception:
                    priv_list = json.loads(
                        pickle.loads(zlib.decompress(base64.b64decode(priv.encode("utf-8"))))
                    )
                tests = pub_list + priv_list
                inputs = [t["input"] for t in tests]
                outputs = [t["output"] for t in tests]
                fn_name = None
                if metadata:
                    try:
                        meta_obj = json.loads(metadata)
                        fn_name = meta_obj.get("func_name")
                    except:
                        pass
                obj = {"inputs": inputs, "outputs": outputs}
                if fn_name:
                    obj["fn_name"] = fn_name
                return {"input_output": json.dumps(obj, ensure_ascii=False)}
            except Exception:
                pass
        return {}

    def __getitem__(self, key):
        return self._d.get(key)

    def __getattr__(self, key):
        try:
            return self._d[key]
        except Exception as e:
            raise AttributeError(key) from e

def normalize_lcb_sample(sample) -> dict:

    if isinstance(sample, dict) and "evaluation" in sample and isinstance(sample["evaluation"], dict):
        sample = sample["evaluation"]

    if isinstance(sample, dict):
        if "input_output" in sample:
            raw = sample["input_output"]
        elif {"inputs", "outputs"} <= set(sample.keys()):
            raw = {k: sample[k] for k in ("inputs", "outputs") if k in sample}
            if "fn_name" in sample:
                raw["fn_name"] = sample["fn_name"]
        elif "tests" in sample:
            raw = sample["tests"]
        elif "test" in sample:
            raw = sample["test"]
        else:
            raise ValueError(f"Evaluation sample missing 'input_output' or 'inputs'/'outputs': keys={list(sample.keys())}")

        if isinstance(raw, str):
            obj = json.loads(raw) 
        elif isinstance(raw, dict):
            obj = raw
        else:
            raise TypeError(f"Unsupported type for 'input_output': {type(raw)}")

        if "inputs" in obj and isinstance(obj["inputs"], str):
            obj["inputs"] = [obj["inputs"]]
        if "outputs" in obj and isinstance(obj["outputs"], str):
            obj["outputs"] = [obj["outputs"]]

        if not {"inputs", "outputs"} <= set(obj.keys()):
            raise ValueError(f"'input_output' JSON missing 'inputs' or 'outputs': keys={list(obj.keys())}")

        return {"input_output": json.dumps(obj, ensure_ascii=False)}

    elif isinstance(sample, str):
        obj = json.loads(sample) 
        if isinstance(obj, dict) and {"inputs", "outputs"} <= set(obj.keys()):
            if isinstance(obj["inputs"], str): obj["inputs"] = [obj["inputs"]]
            if isinstance(obj["outputs"], str): obj["outputs"] = [obj["outputs"]]
            return {"input_output": json.dumps(obj, ensure_ascii=False)}
        return {"input_output": sample}

    else:
        raise TypeError(f"Unsupported evaluation sample type: {type(sample)}")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

lcb_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets_', 'LiveCodeBench'))
if lcb_repo_root not in sys.path:
    sys.path.insert(0, lcb_repo_root)
from Topodim.graph.graph import Graph
from Topodim.tools.coding.python_executor import PyExecutor  
from Topodim.utils.globals import Time
from Topodim.utils.const import Topodim_ROOT
from Topodim.utils.globals import Cost, PromptTokens, CompletionTokens

from lcb_runner.prompts.code_generation import PromptConstants, get_generic_question_template_answer
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from datasets import load_dataset

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def get_temperature(iteration: int, total_iterations: int, init_temp: float = 2.0, final_temp: float = 0.5) -> float:
    if total_iterations <= 0:
        return final_temp
    progress = max(0.0, min(1.0, iteration / total_iterations))
    return init_temp - (init_temp - final_temp) * progress

def lcb_build_prompt(problem) -> str:
    """
    Use LCB’s official “Python Program” format (Generic System + Generic Question Template). Return the prompt as plain text and pass it to Topodim’s CodeWriting agent as the task.
    """
    # SYSTEM + Question template（generate Python program, read stdin write stdout）
    sys_msg = PromptConstants.SYSTEM_MESSAGE_GENERIC
    try:
        q = get_generic_question_template_answer(problem)
    except Exception:
        q = problem.question_content if hasattr(problem, "question_content") else str(problem)
    return f"\n{q}"

def strip_code_fence(text: str) -> str:
    return text.lstrip("```python\n").rstrip("\n```").strip()

def parse_args():
    parser = argparse.ArgumentParser(description="Topodim Experiments on LiveCodeBench (release_v1)")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-oss:20b") 
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star','hetero'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_rounds', type=int, default=2)
    parser.add_argument('--pruning_rate', type=float, default=0.25)
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--domain', type=str, default="humaneval") 
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'])
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5])
    parser.add_argument('--decision_method', type=str, default='FinalWriteCode')
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')
    parser.add_argument('--local_lcb_dataset_dir', type=str, default='./datasets_/LiveCodeBench/code_generation_lite', help="Path to local LCB dataset repo (code_generation_lite)")

    parser.add_argument('--diversity_weight', type=float, default=0.8)
    parser.add_argument('--entropy_weight', type=float, default=0.01)
    parser.add_argument('--baseline_momentum', type=float, default=0.9)


    parser.add_argument('--use_temperature_annealing', action='store_true')
    parser.add_argument('--init_temperature', type=float, default=2.0)
    parser.add_argument('--final_temperature', type=float, default=0.5)


    parser.add_argument('--release_version', type=str, default='release_v1', help="LCB dataset version")
    parser.add_argument('--not_fast', action='store_true', help="Use full (slow) tests instead of lite set")
    parser.add_argument('--lcb_timeout', type=int, default=6, help="Timeout per test in seconds")
    parser.add_argument('--lcb_num_process_evaluate', type=int, default=8, help="Processes for LCB evaluation")
    parser.add_argument('--start_date', type=str, default=None, help="Filter problems by date >= YYYY-MM-DD")
    parser.add_argument('--end_date', type=str, default=None, help="Filter problems by date <= YYYY-MM-DD")

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

    # result file
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{Topodim_ROOT}/result/eval")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.llm_name}_LCB_{args.release_version}_{current_time}.json"

    # Load LCB dataset (release_v1 defaults to code_generation_lite)
    # Note: Depends on 'datasets' library and requires network download (first time)
    benchmark = load_dataset(args.local_lcb_dataset_dir,
                                split="test",
                                version_tag='release_v1',
                                trust_remote_code=True)
    # Convert Dataset to list and adapt if necessary
    raw_benchmark = list(benchmark)
    if len(raw_benchmark) > 0 and isinstance(raw_benchmark[0], dict):
        benchmark = [ProblemAdapter(x) for x in raw_benchmark]
    else:
        benchmark = raw_benchmark

    # Sort by question_id for stable order
    benchmark = sorted(benchmark, key=lambda x: x.question_id or "")

    # Construct graph
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode, len(agent_names))
    graph = Graph(domain="humaneval", 
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=args.decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    graph.rgcn.train()
    optimizer = torch.optim.Adam(graph.rgcn.parameters(), lr=args.lr)

    num_batches = int(len(benchmark) / args.batch_size)
    total_solved, total_executed = 0, 0

    baseline_reward = None
    print("\n" + "=" * 80)
    print(f"LiveCodeBench Training (release={args.release_version})")
    print("=" * 80)
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Diversity Weight: {args.diversity_weight}")
    print(f"  - Entropy Weight: {args.entropy_weight}")
    print(f"  - Temperature Annealing: {args.use_temperature_annealing}")
    if args.use_temperature_annealing:
        print(f"  - Temperature Range: [{args.final_temperature}, {args.init_temperature}]")
    print(f"  - LCB Timeout: {args.lcb_timeout}s, Processes: {args.lcb_num_process_evaluate}")
    print("=" * 80 + "\n")

    for i_batch in range(num_batches): 
        print(f"{'=' * 80}")
        print(f"Batch {i_batch}")
        print(f"{'=' * 80}")
        start_ts = time.time()

        if args.use_temperature_annealing:
            temperature = get_temperature(i_batch, args.num_iterations, args.init_temperature, args.final_temperature) \
                if i_batch < args.num_iterations else args.final_temperature
        else:
            temperature = 1.0
        print(f"🌡️  Temperature: {temperature:.3f}")

        current_batch = dataloader(benchmark, args.batch_size, i_batch)
        if not current_batch:
            print("No more data available.")
            break
        answer_tasks = []
        realized_graphs = []
        input_dicts = []
        for problem in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.rgcn = graph.rgcn
            realized_graph.temperature = temperature

            prompt_text = lcb_build_prompt(problem)
            input_dict = {"task": prompt_text}
            input_dicts.append(input_dict)
            realized_graphs.append(realized_graph)
            answer_tasks.append(asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds)))

        raw_results = await asyncio.gather(*answer_tasks)
        raw_answers, log_probs, diversity_scores, entropy_scores = zip(*raw_results)

        answers_text: List[str] = []
        for ra in raw_answers:
            if not isinstance(ra, list):
                raise TypeError(f"Expected a list for the answer, but got {type(ra).__name__}")
            ans = strip_code_fence(ra[0])
            answers_text.append(ans)

        eval_samples = [p.get_evaluation_sample() for p in current_batch]
        generations = [[c] for c in answers_text] 

        metrics, results, metadatas = codegen_metrics(
            eval_samples,
            generations,
            num_process_evaluate=args.lcb_num_process_evaluate,
            timeout=args.lcb_timeout,
            debug=False,
        )

        utilities: List[float] = []
        total_rewards: List[float] = []
        loss_list: List[torch.Tensor] = []
        solved_flags: List[bool] = []
        # data = load_result(result_file)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

        # results {idx_in_batch: [[bool,bool,...]]}
        for local_idx in range(len(current_batch)):

            raw_res = results.get(local_idx) if isinstance(results, dict) else results[local_idx]
            if not raw_res:
                case_results = [False]
            elif isinstance(raw_res[0], list):
                case_results = raw_res[0]
            else:
                case_results = raw_res
            # case_results is a list of booleans for all tests of a single generation
            is_solved = bool(np.all(np.array(case_results) > 0))
            solved_flags.append(is_solved)
            total_solved += int(is_solved)
            total_executed += 1
            utilities.append(1.0 if is_solved else 0.0)

        diversity_vals = [(float(d.item()) if hasattr(d, "item") else float(d)) for d in diversity_scores]
        entropy_vals = [(float(e.item()) if hasattr(e, "item") else float(e)) for e in entropy_scores]
        total_rewards = [u + args.diversity_weight * d for u, d in zip(utilities, diversity_vals)]
        batch_mean_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        baseline_reward = batch_mean_reward if baseline_reward is None else \
            args.baseline_momentum * baseline_reward + (1 - args.baseline_momentum) * batch_mean_reward

        for tr, lp, ent in zip(total_rewards, log_probs, entropy_scores):
            advantage = tr - baseline_reward
            policy_loss = -lp * advantage
            entropy_reg = -args.entropy_weight * ent
            loss_list.append(policy_loss + entropy_reg)

        accuracy = total_solved / total_executed if total_executed > 0 else 0.0

        total_loss = torch.mean(torch.stack(loss_list)) if loss_list else torch.tensor(0.0)
        if args.optimized_spatial or args.optimized_temporal:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        mean_util = sum(utilities) / len(utilities) if utilities else 0.0
        mean_div = sum(diversity_vals) / len(diversity_vals) if diversity_vals else 0.0
        mean_ent = sum(entropy_vals) / len(entropy_vals) if entropy_vals else 0.0
        mean_tr = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
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
        try:
            print(f"  LCB pass@1 (batch): {metrics['pass@1']:.4f}")
        except Exception:
            pass

        print(f"\n{'='*80}")
        print(f"🎯 Detailed Predictions:")
        print(f"{'='*80}")
        # print("eval_samples\n",eval_samples)
        for i, (inp, ans, solved, realized_graph) in enumerate(zip(input_dicts, answers_text, solved_flags, realized_graphs)):
            status = '✅' if solved else '❌'
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

        if i_batch + 1 == args.num_iterations:
            args.optimized_spatial = False
            args.optimized_temporal = False
            total_solved = 0
            total_executed = 0
            graph.rgcn.eval()
            print("Start Eval")

    # 汇总 Token
    print("\n" + "=" * 60)
    print("Token Usage Summary:")
    print("=" * 60)
    print(f"Total Prompt Tokens:     {int(PromptTokens.instance().value):,}")
    print(f"Total Completion Tokens: {int(CompletionTokens.instance().value):,}")
    print(f"Total Tokens:            {int(PromptTokens.instance().value + CompletionTokens.instance().value):,}")
    if Cost.instance().value > 0:
        print(f"Total Cost:              ${Cost.instance().value:.4f}")
    print("=" * 60)

def get_kwargs(mode: Union[Literal['DirectAnswer'], Literal['FullConnected'], Literal['Random'], Literal['Chain'], Literal['Debate'], Literal['Layered'], Literal['Star']], N: int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

    def generate_layered_graph(N, layer_num=2):
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
            for j in range(i + 1, n):
                matrix[i][j] = 1
        return matrix

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Programming Expert'}]
    elif mode == 'FullConnected':
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Random':
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == 'Chain':
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'hetero':
        node_kwargs = [ {"llm_name": "gpt-oss:20b"},
                        {"llm_name": "gpt-oss:120b"}, 
                        {"llm_name": "gpt-oss:20b"}, 
                        {"llm_name": "gpt-oss:120b"},
                        {"llm_name": "gpt-oss:20b"}]
 
    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs
    }

if __name__ == '__main__':
    asyncio.run(main())