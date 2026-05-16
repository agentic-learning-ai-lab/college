import json
import re
from argparse import ArgumentParser
from typing import Optional

import numpy as np
import torch
from datasets import load_from_disk

from college.arguments import DEFAULT_LLAMA_PATH
from college.evaluation.common import evaluate_top_1, load_college_model, rows_from_dataset, sample_examples, score_answer_suffix

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **_: x


EXAMPLE_PROMPT = 'The following are examples using a new word <nonce>:\n{}\nThe definition of <nonce> is "{}"'
DEFINITION_PROMPT = 'The definition of <nonce> is "{}"'
BASE_PROMPT = ' "{}"'

NORMALIZED_TERMS = {
    "take the l": "take the L",
    "goblin era": "goblin mode",
    "menty b": "menty B",
    "caught in 4k": "caught in 4K",
    "trade": "trad",
}


def normalize_term(term: str) -> str:
    return NORMALIZED_TERMS.get(term, term)


def replace_term_with_nonce(text: str, term: str) -> str:
    variants = {
        term,
        term.lower(),
        term.capitalize(),
        term.upper(),
        " ".join([word.capitalize() for word in term.split(" ")]),
    }
    if term == "beyhive":
        variants.add("BeyHive")
    if term == "goblin mode":
        variants.add("GOBLIN mode")
    if term == "l's":
        variants.update({"L's", "l's"})

    output = text
    for variant in sorted(variants, key=len, reverse=True):
        output = output.replace(variant, "<nonce>")
    return output


def sample_term_examples(ex, term: str, field: str, k: int):
    examples = [replace_term_with_nonce(text, term) for text in ex[field] if term.lower() in text.lower()]
    return sample_examples(examples, k, term)


def prepare_twitter_example(ex, k: int, with_prompt: bool = False):
    labels = [1, 0, 0, 0]
    definition = replace_term_with_nonce(ex["definition"], normalize_term(ex["word"]))

    samples = []
    sequences = []
    word = normalize_term(ex["word"])
    word_samples = sample_term_examples(ex, word, "word_examples", k)
    samples.append(word_samples)

    if with_prompt:
        sequence = EXAMPLE_PROMPT.format("\n".join(word_samples), definition)
    else:
        sequence = DEFINITION_PROMPT.format(definition)
    sequences.append((sequence, BASE_PROMPT.format(definition)))

    for index in range(3):
        negative = normalize_term(ex[f"negative_choice_{index}"])
        negative_samples = sample_term_examples(ex, negative, f"negative_choice_examples_{index}", k)
        samples.append(negative_samples)
        if with_prompt:
            negative_sequence = EXAMPLE_PROMPT.format("\n".join(negative_samples), definition)
        else:
            negative_sequence = DEFINITION_PROMPT.format(definition)
        sequences.append((negative_sequence, BASE_PROMPT.format(definition)))

    return samples, sequences, labels


@torch.no_grad()
def evaluate_twitter_example(ex, model, tokenizer_mlm, tokenizer_task, k: int, with_prompt: bool = False) -> bool:
    samples, sequences, labels = prepare_twitter_example(ex, k, with_prompt=with_prompt)
    probs = []
    for sample, (sequence, base_sequence) in zip(samples, sequences):
        context = tokenizer_mlm(sample, truncation=True, padding="longest", return_tensors="pt").to(model.device)
        tokens = tokenizer_task(sequence, truncation=True, return_tensors="pt").to(model.device)
        batch = {
            "contexts": [context],
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["input_ids"].clone(),
        }
        outputs = model(batch)
        probs.append(score_answer_suffix(outputs.logits, tokenizer_task, sequence, base_sequence))
    return evaluate_top_1(probs, labels)


def run_twitter_evaluation(
    checkpoint_path: str,
    dataset_path: str = "new_twitter_large_v3",
    trials: int = 3,
    max_k: int = 4,
    with_prompt: bool = False,
    output_path: Optional[str] = None,
    device: Optional[str] = None,
    first_lm: str = "roberta-large",
    second_lm: str = DEFAULT_LLAMA_PATH,
    layer: int = 1,
    num_feature_layers: int = 1,
    num_layers: int = 1,
):
    model, tokenizer_mlm, tokenizer_task = load_college_model(
        checkpoint_path,
        device=device,
        first_lm=first_lm,
        second_lm=second_lm,
        layer=layer,
        num_feature_layers=num_feature_layers,
        num_layers=num_layers,
    )
    dataset = rows_from_dataset(load_from_disk(dataset_path))
    scores = {}
    for _ in range(trials):
        for k in range(1, max_k + 1):
            outputs = [
                evaluate_twitter_example(ex, model, tokenizer_mlm, tokenizer_task, k, with_prompt=with_prompt)
                for ex in tqdm(dataset, desc=f"Twitter k={k}")
            ]
            scores.setdefault(k, []).append(sum(outputs) / len(outputs))

    if output_path is not None:
        with open(output_path, "w") as fp:
            json.dump(scores, fp, indent=2)
    return scores


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", default="new_twitter_large_v3")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max_k", type=int, default=4)
    parser.add_argument("--with_prompt", action="store_true")
    parser.add_argument("--output_path", default="twitter_scores.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--first_lm", default="roberta-large")
    parser.add_argument("--second_lm", default=DEFAULT_LLAMA_PATH)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--num_feature_layers", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    scores = run_twitter_evaluation(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_path,
        trials=args.trials,
        max_k=args.max_k,
        with_prompt=args.with_prompt,
        output_path=args.output_path,
        device=args.device,
        first_lm=args.first_lm,
        second_lm=args.second_lm,
        layer=args.layer,
        num_feature_layers=args.num_feature_layers,
        num_layers=args.num_layers,
    )
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
