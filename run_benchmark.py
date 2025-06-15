import csv
import os
import json
import weave
import asyncio
from fire import Fire
import openai

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


def get_openai_models() -> List[str]:
    """Fetch available model names from the OpenAI API."""
    try:
        client = openai.OpenAI()
        response = client.models.list()
        return [model.id for model in response.data]
    except Exception as exc:
        print(f"Failed to fetch OpenAI models: {exc}")
        return []


def run_benchmark(
    model_name: str = "gpt-4o-mini",
    dataset_path: str = "simple_bench_public.json",
    num_responses: int = 1,
    temp: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    max_retries: int = 3,
    confidence: Optional[bool] = True,
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



def run_all_benchmarks(
    dataset_path: str = "simple_bench_public.json",
    num_responses: int = 1,
    entity: str = "simplebench",
    project: str = "simple_bench_public",
    temp: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    max_retries: int = 3,
    confidence: Optional[bool] = False,
    system_prompt_path: str = "system_prompt.txt",
    repeats: int = 3,
) -> None:
    """Run benchmarks for all available OpenAI models."""
    models = get_openai_models()
    all_results = []
    for model_name in models:
        for idx in range(repeats):
            res = run_benchmark(
                model_name=model_name,
                dataset_path=dataset_path,
                num_responses=num_responses,
                entity=entity,
                project=project,
                temp=temp,
                max_tokens=max_tokens,
                top_p=top_p,
                max_retries=max_retries,
                confidence=confidence,
                system_prompt_path=system_prompt_path,
            )
            all_results.append({
                "model_name": model_name,
                "run_index": idx + 1,
                "result": res,
            })

    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    Fire({
        "run_benchmark": run_benchmark,
        "run_all_benchmarks": run_all_benchmarks,
    })
