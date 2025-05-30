from typing import List
import weave
import re

model_score_correct = {
    "1": 1.0,
    "2": 1.5,
    "3": 2.0
}
model_score_incorrect = {
    "1": 0.0,
    "2": -0.5,
    "3": -2.0
}

@weave.op()
def extract_answer(output: str) -> str:
    match = re.search(r"Final Answer:\s*([A-F])", output.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    else:
        raise ValueError("No answer found in model output")

def extract_confidence(output: str) -> str:
    match = re.search(r"Final Answer:\s*([A-F]),\s*([1-3])", output.strip(), re.IGNORECASE)
    if match:
        return match.group(2).upper()
    else:
        raise ValueError("No answer found in model output")


@weave.op()
def eval_majority_vote(output: List[str], answer: str):
    model_answers = []
    for _output in output:
        try:
            model_answers.append(extract_answer(_output))
        except ValueError:
            continue  # Skip this output if extraction fails
    
    if not model_answers:
        raise ValueError("Failed to extract any valid answers from model outputs")
    
    return model_answers.count(answer) > len(model_answers) / 2


@weave.op()
def eval_multi_choice(output: str, answer: str):
    model_answer = extract_answer(output)
    return model_answer == answer

@weave.op()
def eval_multi_choice_confidence(output: str, answer: str):
    model_answer = extract_answer(output)
    model_confidence = extract_confidence(output)
    if model_answer == answer:
        return model_score_correct[model_confidence]
    else:
        return model_score_incorrect[model_confidence]
    return model_answer == answer
