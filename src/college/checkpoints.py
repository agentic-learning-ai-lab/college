import os


def dataset_name_from_path(data_path: str) -> str:
    if "interleaved" in data_path:
        return "interleaved"
    if "generated" in data_path:
        return "generated"
    if "redone_pile" in data_path:
        return "redone_pile"
    return "pile"


def create_checkpoint_directories(args) -> str:
    if args.negative_examples and args.regression_objective:
        objective_name = "with_negatives_and_regression"
    elif args.negative_examples:
        objective_name = "with_negatives"
    elif args.regression_objective:
        objective_name = "with_regression"
    else:
        objective_name = "vanilla"

    try:
        import torch

        device_count = max(torch.cuda.device_count(), 1)
    except ModuleNotFoundError:
        device_count = 1
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * device_count
    path = os.path.join(
        args.checkpoint_root,
        "layers",
        "no_mp",
        "llama",
        "input_and_output",
        "filtered",
        dataset_name_from_path(args.data_path),
        "layernorm",
        args.first_lm,
        f"{args.num_layers}_layers",
        f"layer_{args.num_feature_layers}",
        f"last_{args.layer}",
        f"{effective_batch_size}_batch_size",
        "mean_agg",
        f"{args.num_examples}_examples",
        f"lr_{args.lr}",
        f"weight_decay_{args.weight_decay}",
        objective_name,
    )

    if args.regression_objective:
        path = os.path.join(path, f"distillation_weight_{args.regression_alpha}_temp_{args.distillation_temp}")
        path = os.path.join(path, "output_embedding_cosine")
    if args.definition_training:
        path = os.path.join(path, "definition_training")
    if args.l2 is not None:
        path = os.path.join(path, "l2")
    if args.ablate_cosine:
        path = os.path.join(path, "ablate_cosine")
    if args.ablate_logits:
        path = os.path.join(path, "ablate_logits")

    path = os.path.join(path, "checkpoints")
    os.makedirs(path, exist_ok=True)
    return path
