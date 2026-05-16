import itertools
import json
import re
from argparse import ArgumentParser
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk

from college.arguments import DEFAULT_LLAMA_PATH
from college.evaluation.common import (
    evaluate_top_1,
    evaluate_top_2,
    load_college_model,
    sample_examples,
    score_answer_suffix,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **_: x


def prepare_for_top_1_selection(ex):
    multi_blank_values = ["(i)", "(ii)", "(iii)"]
    question = ex["QUESTION"]
    answers = ex["ANSWERS"]
    task_sequences = []

    if "_____" in question:
        answers = answers[0]
        for answer in answers:
            task_sequences.append(question.replace("_____", answer))
    elif "(i)" in question:
        to_replace = [value for value in multi_blank_values if value in question]
        if len(to_replace) != len(ex["LABELS"]):
            raise ValueError("Multi-blank question does not match label count.")
        for combination in itertools.product(*answers):
            task_sequence = question
            for slot, answer in zip(to_replace, combination):
                task_sequence = task_sequence.replace(slot, answer)
            task_sequences.append(task_sequence)
    else:
        raise NotImplementedError("GRE questions must contain blanks.")

    labels = ex["LABELS"]
    return task_sequences, [1 if all(label in seq for label in labels) else 0 for seq in task_sequences]


def prepare_for_top_2_selection(ex):
    question = ex["QUESTION"]
    if "_____" not in question:
        raise NotImplementedError("Top-2 GRE questions must contain a single blank.")

    task_sequences = [question.replace("_____", answer) for answer in ex["ANSWERS"][0]]
    labels = ex["LABELS"]
    return task_sequences, [1 if any(label in seq for label in labels) else 0 for seq in task_sequences]


def prepare_embedding_generator_batch(ex, sentence_dict, with_prompt: bool = False):
    if ex["ANSWER_TYPE"] == "top_1":
        task_sequences, labels = prepare_for_top_1_selection(ex)
        answers = ex["ANSWERS"][0] if "_____" in ex["QUESTION"] else ex["ANSWERS"]
    elif ex["ANSWER_TYPE"] == "top_2":
        task_sequences, labels = prepare_for_top_2_selection(ex)
        answers = ex["ANSWERS"][0]
    else:
        raise NotImplementedError(f"Unsupported answer type: {ex['ANSWER_TYPE']}")

    task_samples = []
    task_inputs = []
    base_inputs = []
    for answer, task_sequence in zip(answers, task_sequences):
        if not isinstance(answer, str):
            raise NotImplementedError("Composite GRE answers are not supported in the release evaluator.")

        samples = [re.sub(r"\b({})\b".format(re.escape(answer)), "<nonce>", s, flags=re.I) for s in sentence_dict[answer]]
        task_input = re.sub(r"\b({})\b".format(re.escape(answer)), "<nonce>", task_sequence, flags=re.I)
        task_samples.append(samples)
        base_inputs.append(task_input)
        if with_prompt:
            task_inputs.append('Here are some sentences for a new word "<nonce>":\n{}'.format("\n".join(samples + [task_input])))
        else:
            task_inputs.append(task_input)

    if with_prompt:
        return task_samples, task_inputs, base_inputs, labels
    return task_samples, task_inputs, labels


@torch.no_grad()
def score_gre_sequences(model, tokenizer_mlm, tokenizer_task, contexts, sequences, base_sequences=None):
    probs = []
    for index, sequence in enumerate(sequences):
        context = tokenizer_mlm(contexts[index], padding="longest", return_tensors="pt").to(model.device)
        tokens = tokenizer_task(sequence, return_tensors="pt").to(model.device)
        batch = {
            "contexts": [context],
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["input_ids"].clone(),
        }
        output = model(batch)
        if base_sequences is None:
            probs.append(-output.loss.item())
        else:
            probs.append(score_answer_suffix(output.logits, tokenizer_task, sequence, base_sequences[index]))
    return probs


@torch.no_grad()
def evaluate_gre_example(model, tokenizer_mlm, tokenizer_task, ex, sentence_dict, with_prompt: bool = False) -> bool:
    if with_prompt:
        samples, sequences, base_sequences, labels = prepare_embedding_generator_batch(ex, sentence_dict, with_prompt=True)
        probs = score_gre_sequences(model, tokenizer_mlm, tokenizer_task, samples, sequences, base_sequences)
    else:
        samples, sequences, labels = prepare_embedding_generator_batch(ex, sentence_dict, with_prompt=False)
        probs = score_gre_sequences(model, tokenizer_mlm, tokenizer_task, samples, sequences)

    if ex["ANSWER_TYPE"] == "top_1":
        return evaluate_top_1(probs, labels)
    if ex["ANSWER_TYPE"] == "top_2":
        return evaluate_top_2(probs, labels)
    raise NotImplementedError(f"Unsupported answer type: {ex['ANSWER_TYPE']}")


def select_gre_sentences(ex, all_sentences, max_k: int, definitions: Optional[Dict[str, str]] = None):
    selected = {}
    source = all_sentences[ex["QUESTION"]]
    for answer, sentences in source.items():
        candidates = [s for s in sentences if re.search(r"\b({})\b".format(re.escape(answer)), s, flags=re.I)]
        if definitions is None:
            selected[answer] = sample_examples(candidates, max_k, answer)
            continue

        sample_count = max(max_k - 1, 0)
        sampled = sample_examples(candidates, sample_count, answer) if sample_count else []
        definition = definitions.get(answer, definitions.get(answer.lower()))
        if definition is None:
            raise KeyError(f"No definition found for GRE answer {answer!r}.")
        selected[answer] = [f"The word <nonce> is defined as {definition}"] + sampled
    return selected


def run_gre_evaluation(
    checkpoint_path: str,
    dataset_path: str = "processed_kaplan_v0",
    sentence_path: str = "gre_examples_gpt4_v2.json",
    definitions_path: str = "",
    trials: int = 3,
    max_k: int = 5,
    with_prompt: bool = False,
    output_path: Optional[str] = None,
    device: Optional[str] = None,
    first_lm: str = "roberta-large",
    second_lm: str = DEFAULT_LLAMA_PATH,
    layer: int = 2,
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
    gre = load_from_disk(dataset_path)
    examples = gre.filter(lambda ex: "(i)" not in ex["QUESTION"])["train"]
    with open(sentence_path, "r") as fp:
        all_sentences = json.load(fp)

    definitions = None
    if definitions_path:
        with open(definitions_path, "r") as fp:
            definitions = json.load(fp)

    scores = {}
    for _ in range(trials):
        selected_by_question = {
            ex["QUESTION"]: select_gre_sentences(ex, all_sentences, max_k=max_k, definitions=definitions) for ex in examples
        }
        for k in range(1, max_k + 1):
            outputs = []
            for ex in tqdm(examples, desc=f"GRE k={k}"):
                selected = selected_by_question[ex["QUESTION"]]
                current = {answer: samples[:k] for answer, samples in selected.items()}
                outputs.append(evaluate_gre_example(model, tokenizer_mlm, tokenizer_task, ex, current, with_prompt=with_prompt))
            scores.setdefault(k, []).append(sum(outputs) / len(outputs))

    if output_path is not None:
        with open(output_path, "w") as fp:
            json.dump(scores, fp, indent=2)
    return scores


def gre_eval(
    emb_gen_model,
    tokenizer_mlm,
    tokenizer_task,
    device,
    dataset_path: str = "processed_kaplan_v0",
    sentence_path: str = "gre_examples_gpt4_v2.json",
    max_k: int = 3,
):
    emb_gen_model.device = device
    emb_gen_model.eval()
    gre = load_from_disk(dataset_path)
    examples = gre.filter(lambda ex: "(i)" not in ex["QUESTION"])["train"]
    with open(sentence_path, "r") as fp:
        all_sentences = json.load(fp)

    selected_by_question = {
        ex["QUESTION"]: select_gre_sentences(ex, all_sentences, max_k=max_k, definitions=None) for ex in examples
    }
    scores = {}
    with torch.no_grad():
        for k in range(1, max_k + 1):
            outputs = []
            for ex in examples:
                selected = selected_by_question[ex["QUESTION"]]
                current = {answer: samples[:k] for answer, samples in selected.items()}
                outputs.append(evaluate_gre_example(emb_gen_model, tokenizer_mlm, tokenizer_task, ex, current))
            scores[k] = sum(outputs) / len(outputs)
    return scores


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", default="processed_kaplan_v0")
    parser.add_argument("--sentence_path", default="gre_examples_gpt4_v2.json")
    parser.add_argument("--definitions_path", default="")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max_k", type=int, default=5)
    parser.add_argument("--with_prompt", action="store_true")
    parser.add_argument("--output_path", default="gre_scores.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--first_lm", default="roberta-large")
    parser.add_argument("--second_lm", default=DEFAULT_LLAMA_PATH)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--num_feature_layers", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    scores = run_gre_evaluation(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_path,
        sentence_path=args.sentence_path,
        definitions_path=args.definitions_path,
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
