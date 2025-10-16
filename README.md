# MemoryBench

**MemoryBench** aims to provide a standardized and extensible benchmark for evaluating memory and continual learning in LLM systems — encouraging future work toward more adaptive, feedback-driven, and efficient LLM systems.

## Introduction

Scaling up data, parameters, and test-time computation has been the mainstream methods to improve LLM systems (LLMsys), but their upper bounds are almost reached due to the gradual depletion of high-quality data and marginal gains obtained from larger computational resource consumption. Inspired by the abilities of human and traditional AI systems in learning from practice, constructing memory and continual learning frameworks for LLMsys has become an important and popular research direction in recent literature. 

Yet, existing benchmarks for LLM memory often focus on evaluating the system on homogeneous reading comprehension tasks with long-form inputs rather than testing their abilities to learn from accumulated user feedback in service time. Therefore, we propose a user feedback simulation framework and a comprehensive benchmark covering multiple domains, languages, and types of tasks to evaluate the continual learning abilities of LLMsys. 
Experiments show that the effectiveness and efficiency of state-ofthe-art baselines are far from satisfying, and we hope this benchmark could pave the way for future studies on LLM memory and optimization algorithms.

> This repository provides a lightweight interface for **loading the MemoryBench dataset** and **evaluations**. For full baseline and experiment implementations, please refer to [https://github.com/LittleDinoC/MemoryBench-code](https://github.com/LittleDinoC/MemoryBench-code).

## Repository Structure

Below is the directory structure of MemoryBench.
Please maintain this structure to ensure proper execution.

```plain
configs/
    datasets/           # Dataset configuration files
    final_evaluate_summary_wo_details.json # Normalization data
raw/                    # Raw datasets
src/
    datasets/               # Dataset classes
    llms/                   # LLM interfaces, including OpenAI and vLLM
    agents/                 # Evaluation Agents
memorybench.py          # Main entry point
.env                    # Environment variables for evaluation configuration
```

Dataset configurations are located in `configs/datasets/`:

* `each.json` — metadata for each dataset
* `domain.json` — datasets grouped by domain
* `task.json` — datasets grouped by task

The full dataset is publicly available on Hugging Face:
👉 [https://huggingface.co/datasets/THUIR/MemoryBench](https://huggingface.co/datasets/THUIR/MemoryBench)

We also provide lightweight dataset loading and evaluation utilities in `memorybench.py`.

## Using MemoryBench

### Environment Setup

Use the following commands to set up the conda environment:

```
conda create -n memorybench python=3.10
conda activate memorybench
pip install -r requirements.txt
```

Please set up the `.env` file to specify evaluation models and optional OpenAI API configurations.
These evaluation models are used for all LLM-as-judge evaluations and integrated scoring across multiple metrics.

### Load Dataset

You can load datasets using the `load_memory_bench` function.

**Parameters:**

* `dataset_type` (`single` | `domain` | `task`):
  Choose to load a single dataset, or merge datasets by domain or by task.
* `name` (str):
  The name of the dataset/domain/task.

  * Datasets are listed on the Hugging Face page.
  * Domains include `Open-Domain`, `Academic&Knowledge`, and `Legal` (see `configs/datasets/domain.json`).
  * Tasks include `Long-Short`, `Long-Long`, `Short-Long`, and `Short-Short` (see `configs/datasets/task.json`).
* `return_instance` (bool, default=True):
  Whether to return the dataset class instance.
  Set to `False` if you only need the raw data for inference.
* `eval_mode` (bool, default=False):
  Whether to enable evaluation mode.
  When `True`, this automatically enables `return_instance=True` and loads the evaluation models.

If `dataset_type` is `single`, the function returns a dictionary; if `domain` or `task`, it returns a list of dictionaries.
Each dictionary has the following structure:

```python
{
    "dataset": dataset,             # HuggingFace dataset object with "train", "test", and optional "corpus"
    "dataset_name": str,            # Dataset name
    "dataset_instance": dataset_instance or None  # Dataset class instance (None if return_instance=False)
}
```

**Example usage:**

```python
from memorybench import load_memory_bench

# Load a single dataset (JRE-L)
dataset_item = load_memory_bench(dataset_type='single', name='JRE-L')

# Load a domain (Open-Domain) without dataset instances
dataset_list = load_memory_bench(dataset_type='domain', name='Open-Domain', return_instance=False)

# Load a task (Long-Short, LiSo) with evaluation mode
dataset_list = load_memory_bench(dataset_type='task', name='Long-Short', return_instance=True, eval_mode=True)
```

### Evaluation

You can evaluate model predictions using the `evaluate` function.

**Parameters:**

* `dataset_type` (`single` | `domain` | `task`): same as above.
* `name` (str): dataset/domain/task name.
* `predicts` (list of dict): list of model predictions.
  Each element must include:

  * `dataset` (str): dataset name.
  * `test_idx` (int): index of the test sample.
  * `response` (str): model’s response.

The function loads the datasets, runs the evaluation, and returns a list of results:

```python
{
    "dataset": str,      # Dataset name
    "test_idx": int,     # Sample index
    "metrics": {         # Evaluation metrics
        "metric_name_1": value_1,
    }
}
```

**Example usage:**

```python
from memorybench import evaluate

evaluate_details = evaluate(
    dataset_type='domain',
    name='Open-Domain',
    predicts=[
        {"test_idx": 0, "response": "Your model's response here.", "dataset": "WritingPrompts"},
        {"test_idx": 1, "response": "Another response.", dataset: "DialSim-friends"},
        # Add more predictions as needed
    ]
)
```

### Summary and Normalization

The `evaluate` function produces per-sample metrics.
To compute overall performance scores, please use the `summary_results` function.

For single datasets, it computes the mean of each metric directly.
For domains or tasks, it additionally performs normalization across datasets using precomputed statistics.

**Parameters:**

* `dataset_type` (`single` | `domain` | `task`): same as above.
* `name` (str): dataset/domain/task name.
* `predicts` (list of dict): model predictions.
* `evaluate_details` (list of dict): detailed results from the `evaluate` function.
* `min_max_config_file` (str, default=`configs/final_evaluate_summary_wo_details.json`):
  Configuration file containing normalization parameters (min, max, mean, std).

The function returns a dictionary whose core field is `summary`, containing the averaged or normalized metrics.

**Example usage:**

```python
from memorybench import evaluate, summary_results

predicts = [
    {"test_idx": 0, "response": "Your model's response here.", "dataset": "WritingPrompts"},
    {"test_idx": 1, "response": "Another response.", dataset: "DialSim-friends"},
    # Add more predictions as needed
]

evaluate_details = evaluate(
    dataset_type='domain',
    name='Open-Domain',
    predicts=predicts
)

summary = summary_results(
    dataset_type='domain',
    name='Open-Domain',
    predicts=predicts,
    evaluate_details=evaluate_details
)
```