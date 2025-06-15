import asyncio
import csv
import json
import os
from typing import Optional, List

from dotenv import load_dotenv
from fire import Fire

from weave_utils.models import LiteLLMModel, MajorityVoteModel
from weave_utils.scorers import (
    eval_majority_vote,
    eval_multi_choice,
    eval_multi_choice_confidence,
)

load_dotenv()


def load_dataset(file_path: str) -> List[dict]:
    with open(file_path, "r") as file:
        data = json.load(file)
        return data["eval_data"]


async def _evaluate_question(
    model: LiteLLMModel,
    prompt: str,
    answer: str,
    num_responses: int,
    confidence: bool,
):
    outputs = []
    if num_responses == 1:
        out = await model.predict(prompt)
        outputs.append(out)
        correct = eval_multi_choice(out, answer)
    else:
        outputs = await model.predict(prompt)
        correct = eval_majority_vote(outputs, answer)

    if confidence:
        scores = [eval_multi_choice_confidence(o, answer) for o in outputs]
        score = sum(scores) / len(scores)
    else:
        score = float(bool(correct))

    return outputs, int(bool(correct)), float(score)


def run_benchmark(
    model_name: str = "gpt-4o-mini",
    dataset_path: str = "simple_bench_public.json",
    num_responses: int = 1,
    temp: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    max_retries: int = 3,
    confidence: Optional[bool] = False,
    system_prompt_path: str = "system_prompt.txt",
    results_dir: str = "results",
):
    """Run a benchmark and save per-question results to a CSV file."""
    if confidence:
        system_prompt_path = "system_prompt_confidence_question_parallel.txt"

    os.makedirs(results_dir, exist_ok=True)

    with open(system_prompt_path, "r") as f:
        system_prompt = f.read().strip()

    model = LiteLLMModel(
        model_name=model_name,
        temp=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        max_retries=max_retries,
        system_prompt=system_prompt,
    )

    if num_responses > 1:
        model = MajorityVoteModel(model=model, num_responses=num_responses)

    dataset = load_dataset(dataset_path)

    async def _run():
        results = []
        for row in dataset:
            outputs, correct, score = await _evaluate_question(
                model,
                row["prompt"],
                row["answer"],
                num_responses,
                confidence,
            )
            results.append(
                {
                    "question_id": row.get("question_id"),
                    "correct": correct,
                    "score": score,
                    "output": " ||| ".join(outputs),
                }
            )
        return results

    results = asyncio.run(_run())

    result_file = os.path.join(results_dir, f"results_{model_name}.csv")
    with open(result_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["question_id", "correct", "score", "output"]
        )
        writer.writeheader()
        writer.writerows(results)

    num_correct = sum(r["correct"] for r in results)
    total_score = sum(r["score"] for r in results)
    print(
        f"Model {model_name}: {num_correct}/{len(results)} correct. Total score: {total_score}"
    )


if __name__ == "__main__":
    Fire(run_benchmark)
