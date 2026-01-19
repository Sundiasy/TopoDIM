
import torch
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy
from collections import defaultdict

from Topodim.graph.graph import Graph
from experiments.accuracy import Accuracy
from Topodim.utils.globals import Cost, PromptTokens, CompletionTokens

def get_temperature(iteration, total_iterations, init_temp=2.0, final_temp=0.5):

    progress = iteration / total_iterations
    return init_temp - (init_temp - final_temp) * progress


async def train_stage2(
    graph: Graph,
    dataset,
    num_iters: int = 100,
    num_rounds: int = 1,
    lr: float = 0.01,  
    batch_size: int = 8,  
    diversity_weight: float = 0.5,  
    entropy_weight: float = 0.01, 
    use_temperature_annealing: bool = True, 
    init_temperature: float = 2.0,  
    final_temperature: float = 0.5, 
    baseline_momentum: float = 0.9, 
    print_edge_distribution: bool = True,  
) -> None:

    def infinite_data_loader() -> Iterator[pd.DataFrame]:
        perm = np.random.permutation(len(dataset))
        while True:
            for idx in perm:
                record = dataset[idx.item()]
                yield record

    loader = infinite_data_loader()

    optimizer = torch.optim.Adam(graph.rgcn.parameters(), lr=lr)


    baseline_utility = 0.0
    edge_type_history = []
    diversity_history = []
    entropy_history = []
    utility_history = []
    total_reward_history = []

    graph.rgcn.train()

    print("\n" + "=" * 80)
    print("=" * 80)
    print(f"Settings:")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Diversity Regularization Weight: {diversity_weight} ")
    print(f"  - Entropy Weight: {entropy_weight}")
    print(f"  - Temperature Annealing: {use_temperature_annealing}")
    if use_temperature_annealing:
        print(f"  - Temperature Range: [{final_temperature}, {init_temperature}]")
    print("=" * 80 + "\n")

    for i_iter in range(num_iters):
        print(f"\n{'=' * 80}")
        print(f"Iter {i_iter}/{num_iters}")
        print(f"{'=' * 80}")
        
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []

        if use_temperature_annealing:
            temperature = get_temperature(i_iter, num_iters, init_temperature, final_temperature)
            print(f"🌡️  Temperature: {temperature:.3f}")
        else:
            temperature = 1.0

        realized_graphs = []
        input_dicts = []
        
        for i_record, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.rgcn = graph.rgcn
            
            realized_graph.temperature = temperature
            
            input_dict = dataset.record_to_input(record)
            input_dicts.append(input_dict)
            
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)
            realized_graphs.append(realized_graph)

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs, diversity_scores, entropy_scores = zip(*raw_results)
        
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        total_rewards = []
        answers: List[str] = []

        for idx, (raw_answer, correct_answer, diversity_score) in enumerate(zip(raw_answers, correct_answers, diversity_scores)):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            utility = accuracy.get()
            utilities.append(utility)
            
            total_reward = utility + diversity_weight * diversity_score.detach().item()
            total_rewards.append(total_reward)

        batch_mean_reward = np.mean(total_rewards)
        if i_iter == 0:
            baseline_reward = batch_mean_reward
        else:
            baseline_reward = baseline_momentum * baseline_reward + \
                              (1 - baseline_momentum) * batch_mean_reward

        for total_reward, log_prob, entropy_score in zip(total_rewards, log_probs, entropy_scores):
            advantage = total_reward - baseline_reward
            
            policy_loss = -log_prob * advantage
            
            entropy_regularization = -entropy_weight * entropy_score
            
            single_loss = policy_loss + entropy_regularization
            loss_list.append(single_loss)

        total_loss = torch.mean(torch.stack(loss_list))
        
        optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(graph.rgcn.parameters(), max_norm=1.0)

        optimizer.step()

        avg_diversity = np.mean([d.item() for d in diversity_scores])
        avg_entropy = np.mean([e.item() for e in entropy_scores])
        batch_mean_utility = np.mean(utilities) 
        diversity_history.append(avg_diversity)
        entropy_history.append(avg_entropy)
        utility_history.append(batch_mean_utility)
        total_reward_history.append(batch_mean_reward)
        print(f"\n📊 Results:")
        print(f"  Batch time: {time.time() - start_ts:.2f}s")
        print(f"  Utilities (Task Reward): {[f'{u:.3f}' for u in utilities]} (mean={batch_mean_utility:.3f})")
        print(f"  Diversity Scores: {[f'{d.item():.3f}' for d in diversity_scores]} (mean={avg_diversity:.3f})")
        print(f"  Total Rewards: {[f'{r:.3f}' for r in total_rewards]} (mean={batch_mean_reward:.3f})")
        print(f"  Baseline (Reward): {baseline_reward:.3f}")
        print(f"  Advantages: {[f'{r - baseline_reward:.3f}' for r in total_rewards]}")
        print(f"  Entropy: {avg_entropy:.3f} (weight={entropy_weight})")
        print(f"  loss: {total_loss:.3f}")
        
        print(f"\n{'='*80}")
        print(f"🎯 Detailed Predictions:")
        print(f"{'='*80}")
        for i, (ans, correct, util, realized_graph, input_dict) in enumerate(zip(
            answers, correct_answers, utilities, realized_graphs, input_dicts
        )):
            status = "✅" if util > 0.5 else "❌"
            print(f"\n  {status} Sample {i}:")
            print(f"     Question: {input_dict['task'][:120]}...")

            print(f"     Topology Execution Order:")
            topo_summary = realized_graph.get_execution_summary()
            for line in topo_summary.split('\n'):
                print(f"       {line}")
            
            print(f"     Predicted Answer: '{ans}'")
            print(f"     Correct Answer:   '{correct}'")

        if (i_iter + 1) % 10 == 0:
            recent_utilities = utility_history[-10:]
            recent_diversities = diversity_history[-10:]
            recent_entropies = entropy_history[-10:]
            recent_edge_usage = edge_type_history[-10:] if edge_type_history else []
            
            print(f"\n📈 Last 10 Iterations Summary:")
            print(f"  Avg Utility: {np.mean(recent_utilities):.3f} (±{np.std(recent_utilities):.3f})")
            print(f"  Avg Diversity: {np.mean(recent_diversities):.3f} (±{np.std(recent_diversities):.3f})")
            print(f"  Avg Entropy: {np.mean(recent_entropies):.3f} (±{np.std(recent_entropies):.3f})")
            if recent_edge_usage:
                print(
                    f"  Avg Edge Distribution: Type0={np.mean([e[0] for e in recent_edge_usage]):.3f}, "
                    f"Type1={np.mean([e[1] for e in recent_edge_usage]):.3f}, "
                    f"Type2={np.mean([e[2] for e in recent_edge_usage]):.3f}"
                )

        print(f"\n💰 Cost:")
        print(f"  Total Cost: ${Cost.instance().value:.4f}")
        print(f"  Prompt Tokens: {PromptTokens.instance().value}")
        print(f"  Completion Tokens: {CompletionTokens.instance().value}")

    print(f"\n{'=' * 80}")
    print("🎉 Train Complete!")
    print(f"{'=' * 80}")
    print(f"Iter Num: {num_iters}")
    print(f"Final Baseline: {baseline_utility:.3f}")
    print(f"Mean Utility: {np.mean(utility_history):.3f}")
    print(f"Mean Diversity: {np.mean(diversity_history):.3f}")
    print(f"Mean Entropy: {np.mean(entropy_history):.3f}")
    if edge_type_history:
        print(
            f"Edge Type Distribution: Type0={np.mean([e[0] for e in edge_type_history]):.3f}, "
            f"Type1={np.mean([e[1] for e in edge_type_history]):.3f}, "
            f"Type2={np.mean([e[2] for e in edge_type_history]):.3f}"
        )
    print(f"Total Cost: ${Cost.instance().value:.4f}")
    print(f"{'=' * 80}\n")

    import os
    from pathlib import Path

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "rgcn_best.pt"
    
    final_utility = float(np.mean(utility_history[-10:])) if len(utility_history) >= 10 else float(np.mean(utility_history))
    
    torch.save({
        'model_state_dict': graph.rgcn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_iters': int(num_iters),
        'final_utility': final_utility,
        'utility_history': [float(x) for x in utility_history],
        'diversity_history': [float(x) for x in diversity_history],
        'entropy_history': [float(x) for x in entropy_history],
        'training_config': {
            'lr': float(lr),
            'batch_size': int(batch_size),
            'diversity_weight': float(diversity_weight),
            'entropy_weight': float(entropy_weight),
            'use_temperature_annealing': bool(use_temperature_annealing),
            'init_temperature': float(init_temperature),
            'final_temperature': float(final_temperature),
        }
    }, checkpoint_path)
    
    print(f"✅ Model checkpoint saved to: {checkpoint_path}")
    print(f"   Final Utility: {final_utility:.3f}")


async def train(
    graph: Graph,
    dataset,
    num_iters: int = 100,
    num_rounds: int = 1,
    lr: float = 0.01,
    batch_size: int = 8,
    **kwargs 
) -> None:
    return await train_stage2(
        graph, dataset, num_iters, num_rounds, lr, batch_size, **kwargs
    )
