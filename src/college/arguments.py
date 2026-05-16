from argparse import ArgumentParser


DEFAULT_LLAMA_PATH = "/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf"


def build_llama_training_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--word_path", type=str, default="")
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--negative_examples", action="store_true")
    parser.add_argument("--negative_data_path", type=str, default="")
    parser.add_argument("--regression_objective", action="store_true")
    parser.add_argument("--regression_alpha", type=float, default=1.0)
    parser.add_argument("--distillation_temp", type=float, default=1.0)
    parser.add_argument("--logging_step", type=int, required=True)
    parser.add_argument("--num_eval_steps", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--num_feature_layers", type=int, default=1)
    parser.add_argument("--l2", type=float, default=None)
    parser.add_argument("--first_lm", type=str, default="roberta-base")
    parser.add_argument("--second_lm", type=str, default=DEFAULT_LLAMA_PATH)
    parser.add_argument("--definition_training", action="store_true")
    parser.add_argument("--ablate_cosine", action="store_true")
    parser.add_argument("--ablate_logits", action="store_true")
    parser.add_argument("--saving", action="store_true")
    parser.add_argument("--checkpoint_root", type=str, default="model_checkpoints")
    parser.add_argument("--project_name", type=str, default="fewshot_llama")
    parser.add_argument("--run_gre_eval", action="store_true")
    parser.add_argument("--lm_alpha", type=float, default=1.0)
    parser.add_argument("--negatives_alpha", type=float, default=1.0)
    parser.add_argument("--cosine_alpha", type=float, default=1.0)
    parser.add_argument("--logits_alpha", type=float, default=1.0)
    return parser
