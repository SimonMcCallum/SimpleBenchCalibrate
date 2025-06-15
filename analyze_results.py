import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def analyze_results(results_dir: str = "results", output_file: str = "performance.png"):
    pattern = os.path.join(results_dir, "results_*.csv")
    result_files = glob.glob(pattern)
    if not result_files:
        print("No result files found.")
        return

    models = []
    num_correct = []
    scores = []
    for path in result_files:
        model_name = os.path.splitext(os.path.basename(path))[0].split("results_")[1]
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        models.append(model_name)
        num_correct.append(np.sum(data["correct"]))
        scores.append(np.sum(data["score"]))

    x = np.arange(len(models))
    fig, ax1 = plt.subplots()
    ax1.bar(x - 0.2, num_correct, width=0.4, color="tab:blue", label="Number Correct")
    ax1.set_ylabel("Number Correct", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, scores, width=0.4, color="tab:orange", label="Total Score")
    ax2.set_ylabel("Total Score", color="tab:orange")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    plt.title("AI Performance")
    fig.tight_layout()
    plt.savefig(output_file)
    print(f"Saved performance graph to {output_file}")


if __name__ == "__main__":
    analyze_results()
