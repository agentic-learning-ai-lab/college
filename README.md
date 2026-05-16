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

## Evaluation

Run the release evaluators against a saved training checkpoint:

```bash
college-eval-gre --checkpoint /path/to/checkpoint --dataset_path processed_kaplan_v0 --sentence_path gre_examples_gpt4_v2.json
college-eval-twitter --checkpoint /path/to/checkpoint --dataset_path new_twitter_large_v3
college-eval-oxford --checkpoint /path/to/checkpoint --dataset_path merged_oxford_test_set.csv
```

For local development without installing the package, use `PYTHONPATH=src python scripts/eval_gre.py`, `scripts/eval_twitter.py`, or `scripts/eval_oxford.py` with the same arguments.

## Checkpoints

Evaluation code loads models through `CollegeLlama.from_checkpoint(...)`, which reconstructs frozen RoBERTa/Llama backbones from their model paths, loads `tokenizerMLM` and `tokenizerTask` from the checkpoint directory, and restores the trained embedding generator from `pytorch_model.bin` or `model.safetensors`.

Training resume still uses Accelerate state restoration via `--resume_from_checkpoint`, because it must also restore optimizer, scheduler, and dataloader position.
