import os
import json
import math
import time
import asyncio
from typing import Union,Literal,Optional,Iterator,List,Any,Dict
from tqdm import tqdm
import copy

from Topodim.graph.graph import Graph
from experiments.accuracy import Accuracy
from Topodim.utils.globals import Cost, PromptTokens, CompletionTokens

async def evaluate(
        graph:Graph,
        dataset,
        num_rounds:int = 1,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 4,
        ) -> float:

    print(f"Evaluating gdesigner on {dataset.__class__.__name__} split {dataset.split}")
    
    graph.rgcn.eval()
    accuracy = Accuracy()
    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None:
                if i_record >= limit_questions:
                    break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records
        return
    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size))

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        print(80*'-')

        start_ts = time.time()
        answer_log_probs = []
        realized_graphs = []
        input_dicts = []
        
        for record in record_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.rgcn = graph.rgcn
            input_dict = dataset.record_to_input(record)
            input_dicts.append(input_dict)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
            realized_graphs.append(realized_graph)
            
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs,*_ = zip(*raw_results)
        print(f"Batch time {time.time() - start_ts:.3f}")
        
        print("\n" + "="*80)
        print("📊 Detailed Evaluation Results:")
        print("="*80)
        
        for idx, (raw_answer, record, realized_graph, input_dict) in enumerate(zip(
            raw_answers, record_batch, realized_graphs, input_dicts
        )):
            answer = dataset.postprocess_answer(raw_answer)
            correct_answer = dataset.record_to_target_answer(record)
            is_correct = answer == correct_answer
            status = "✅" if is_correct else "❌"
            
            print(f"\n{status} Sample {i_batch * eval_batch_size + idx}:")
            print(f"  Question: {input_dict['task'][:150]}...")
            
            print(f"  Topology Execution Order:")
            topo_summary = realized_graph.get_execution_summary()
            for line in topo_summary.split('\n'):
                print(f"    {line}")
            
            print(f"  Predicted Answer: '{answer}'")
            print(f"  Correct Answer:   '{correct_answer}'")
            
            accuracy.update(answer, correct_answer)
        
        print("\n" + "="*80)
        print(f"Current Accuracy: {accuracy.get():.4f} ({accuracy._num_correct}/{accuracy._num_total})")
        print(f"Cost: ${Cost.instance().value:.4f}")
        print(f"Tokens: {int(PromptTokens.instance().value):,} prompt + {int(CompletionTokens.instance().value):,} completion")
        print("="*80)
    accuracy.print()

    return accuracy.get()


def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)
