import uuid
from argparse import ArgumentParser
from typing import Optional

import torch
from datasets import Dataset, load_dataset, load_from_disk

from college.arguments import DEFAULT_LLAMA_PATH
from college.evaluation.common import load_college_model, rows_from_dataset
from college.generation import generate

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **_: x


DEFINITION_PROMPT = 'Given the following examples: {}, the word "{}" is defined as'


def load_oxford_dataset(path: str):
    if path.endswith(".csv"):
        return load_dataset("csv", data_files=path)["train"]
    return rows_from_dataset(load_from_disk(path))


def get_replaced_examples(ex):
    examples = ex["replaced_examples"]
    if isinstance(examples, str):
        return [examples]
    return list(examples)


@torch.no_grad()
def generate_oxford_definition(model, ex, tokenizer_mlm, tokenizer_task, with_prompt: bool, max_new_tokens: int = 30):
    examples = get_replaced_examples(ex)
    context = tokenizer_mlm(examples, truncation=True, padding="longest", return_tensors="pt").to(model.device)
    nonce = "<nonce>"
    if with_prompt:
        prompt = DEFINITION_PROMPT.format("\n".join(examples), nonce)
    else:
        prompt = f'The word "{nonce}" is defined as'

    inputs = tokenizer_task(prompt, truncation=True, return_tensors="pt", max_length=256).to(model.device)
    outputs = generate(
        model,
        context,
        inputs["input_ids"],
        inputs["attention_mask"],
        max_new_tokens,
        mask_new_tokens=True,
    )
    generated_definition = tokenizer_task.decode(outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)
    return {
        "definition": ex["definition"],
        "word": ex["word"],
        "generated definition": generated_definition,
        "examples": examples,
        "prompt": prompt,
    }


def run_oxford_definition_generation(
    checkpoint_path: str,
    dataset_path: str = "merged_oxford_test_set.csv",
    output_path: Optional[str] = None,
    prompt_mode: str = "both",
    max_new_tokens: int = 30,
    device: Optional[str] = None,
    first_lm: str = "roberta-large",
    second_lm: str = DEFAULT_LLAMA_PATH,
    layer: int = 1,
    num_feature_layers: int = 1,
    num_layers: int = 1,
):
    if prompt_mode not in {"both", "with", "without"}:
        raise ValueError("--prompt_mode must be one of: both, with, without.")

    model, tokenizer_mlm, tokenizer_task = load_college_model(
        checkpoint_path,
        device=device,
        first_lm=first_lm,
        second_lm=second_lm,
        layer=layer,
        num_feature_layers=num_feature_layers,
        num_layers=num_layers,
    )
    dataset = load_oxford_dataset(dataset_path)
    outputs = []
    for ex in tqdm(dataset, desc="Oxford definitions"):
        if prompt_mode in {"both", "with"}:
            outputs.append(
                generate_oxford_definition(
                    model,
                    ex,
                    tokenizer_mlm,
                    tokenizer_task,
                    with_prompt=True,
                    max_new_tokens=max_new_tokens,
                )
            )
        if prompt_mode in {"both", "without"}:
            outputs.append(
                generate_oxford_definition(
                    model,
                    ex,
                    tokenizer_mlm,
                    tokenizer_task,
                    with_prompt=False,
                    max_new_tokens=max_new_tokens,
                )
            )

    if output_path is None:
        output_path = f"oxford_task_outputs/emb_gen_generations_{uuid.uuid4()}"
    Dataset.from_list(outputs).save_to_disk(output_path)
    return outputs


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", default="merged_oxford_test_set.csv")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--prompt_mode", choices=["both", "with", "without"], default="both")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--first_lm", default="roberta-large")
    parser.add_argument("--second_lm", default=DEFAULT_LLAMA_PATH)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--num_feature_layers", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    outputs = run_oxford_definition_generation(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        prompt_mode=args.prompt_mode,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        first_lm=args.first_lm,
        second_lm=args.second_lm,
        layer=args.layer,
        num_feature_layers=args.num_feature_layers,
        num_layers=args.num_layers,
    )
    print(f"Wrote {len(outputs)} generations.")


if __name__ == "__main__":
    main()
