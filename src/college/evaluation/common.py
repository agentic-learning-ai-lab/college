from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from college.arguments import DEFAULT_LLAMA_PATH
from college.modeling_llama import CollegeLlama


def load_college_model(
    checkpoint_path: str,
    device: Optional[str] = None,
    first_lm: str = "roberta-large",
    second_lm: str = DEFAULT_LLAMA_PATH,
    layer: int = 1,
    num_feature_layers: int = 1,
    num_layers: int = 1,
    strict: bool = True,
):
    return CollegeLlama.from_checkpoint(
        checkpoint_path,
        device=device,
        first_lm=first_lm,
        second_lm=second_lm,
        layer=layer,
        num_feature_layers=num_feature_layers,
        num_layers=num_layers,
        strict=strict,
        return_tokenizers=True,
    )


def evaluate_top_1(probs: Sequence[float], labels: Sequence[int]) -> bool:
    return int(np.argmax(np.array(probs))) == labels.index(1)


def evaluate_top_2(probs: Sequence[float], labels: Sequence[int]) -> bool:
    predicted = set(np.argsort(np.array(probs), axis=0)[-2:].tolist())
    gold = {index for index, value in enumerate(labels) if value == 1}
    return predicted == gold


def score_answer_suffix(logits, tokenizer, seq: str, suffix: str, vocab_size: Optional[int] = None) -> float:
    if vocab_size is None:
        vocab_size = logits.shape[-1]
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)

    labels = tokenizer(seq, return_tensors="pt")["input_ids"].clone()
    suffix_tokens = tokenizer(suffix)
    answer_length = len(suffix_tokens["input_ids"]) - 1

    answer_labels = labels[:, -answer_length:]
    answer_logits = logits[:, -answer_length:, :]
    shift_logits = answer_logits[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = answer_labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
    loss = CrossEntropyLoss()(shift_logits, shift_labels)
    return -loss.item()


def require_min_examples(examples: Sequence[str], k: int, label: str) -> None:
    if len(examples) < k:
        raise ValueError(f"Need {k} examples for {label!r}, but only found {len(examples)}.")


def sample_examples(examples: Sequence[str], k: int, label: str) -> List[str]:
    require_min_examples(examples, k, label)
    return np.random.choice(list(examples), size=k, replace=False).tolist()


def rows_from_dataset(dataset) -> Iterable:
    if hasattr(dataset, "keys") and "train" in dataset:
        return dataset["train"]
    return dataset
