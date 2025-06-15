# Simple Bench

https://simple-bench.com/

## Run Instructions

Run benchmark:
```
python run_benchmark.py --model_name=gpt-4o --dataset_path=simple_bench_public.json
```

After running benchmarks for each model, a CSV file will be written to the
`results/` directory containing the score and whether each answer was correct.
To generate a performance graph from these CSV files run:

```
python analyze_results.py
```

## Setup Instructions

Clone the github repo and cd into it.

Make sure you have the correct python version (3.10.11) as a venv:
```
pyenv local 3.10.11
python -m venv llm_env
source llm_env/bin/activate
```

Install dependencies:

The best way to install dependencies is to use `uv`. If you don't have it installed in your environment, you can install it with `pip install uv`.

``` 
uv pip install -r pyproject.toml
```

Create a `.env` file with the following:
```
OPENAI_API_KEY=<your key>
ANTHROPIC_API_KEY=<your key>
...
```
