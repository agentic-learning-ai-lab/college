# CoLLEGe [COLM 2024]

Code release for the CoLLEGe few-shot learning experiments.

## Links

- [OpenReview](https://openreview.net/forum?id=Fkr1yVUb9G)
- [arXiv](https://arxiv.org/abs/2403.15362)
- [Project website](https://agenticlearning.ai/college/)

## Training

The Llama training entrypoint lives in `college.train_with_llama` and is exposed as:

```bash
college-train-llama --epochs 1 --logging_step 100 --data_path /path/to/dataset
```

For local development without installing the package:

```bash
PYTHONPATH=src python scripts/train_with_llama.py --epochs 1 --logging_step 100 --data_path /path/to/dataset
```

The refactor keeps the original Llama-2 path as the default `--second_lm`, but it can now be overridden from the command line.

Supported final-paper settings:

- `--first_lm` must be a RoBERTa checkpoint, such as `roberta-base` or `roberta-large`.
- Memory aggregation is fixed to mean.
- Each batch samples a random number of context examples from `1..num_examples`.
