import os
import re
from functools import partial

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaForCausalLM,
    LlamaTokenizer,
    RobertaForMaskedLM,
    get_linear_schedule_with_warmup,
)

from .arguments import build_llama_training_parser
from .checkpoints import create_checkpoint_directories
from .configs import AggregatorConfig
from .modeling_llama import CollegeLlama
from .utils import seed_worker


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")


def validate_args(args) -> None:
    if "roberta" not in args.first_lm.lower():
        raise ValueError(f"--first_lm must be a RoBERTa checkpoint, got {args.first_lm!r}.")
    if args.negative_examples and not args.negative_data_path:
        raise ValueError("--negative_data_path is required when --negative_examples is set.")


def tokenize_example(ex, tokenizer_task):
    return tokenizer_task(ex["text"], truncation=True, max_length=256, padding="max_length", return_tensors=None)


def tokenize_regression_example(ex, tokenizer_task, definition_training: bool):
    text = "<def>" + ex["text"] if definition_training else ex["text"]
    inputs = tokenizer_task(text, truncation=True, max_length=256, padding="max_length", return_tensors=None)
    base_inputs = tokenizer_task(
        ex["base text"],
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors=None,
    )
    row = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "base_input_ids": base_inputs["input_ids"],
        "base_attention_mask": base_inputs["attention_mask"],
    }
    if definition_training:
        row["definition"] = ex["definition"]
    return row


def sample_context(k: int, ex, tokenizer_mlm):
    if len(ex["sentences"]) < k:
        raise ValueError(f"Requested {k} context examples but row only has {len(ex['sentences'])}.")
    sentences = np.random.choice(ex["sentences"], size=k, replace=False).tolist()
    return tokenizer_mlm(sentences, max_length=256, truncation=True, padding="longest", return_tensors="pt")


def choose_context_count(max_num_examples: int):
    return np.random.choice(max_num_examples) + 1


def regular_collate(max_num_examples, batch, tokenizer_mlm, data_collator):
    num_examples = choose_context_count(max_num_examples)
    contexts = [sample_context(num_examples, row, tokenizer_mlm) for row in batch]
    input_batch = [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in batch]
    final_collate = dict(data_collator(input_batch))
    final_collate["contexts"] = contexts
    return final_collate


def regression_collate(
    max_num_examples,
    batch,
    tokenizer_mlm,
    data_collator,
    definition_training=False,
):
    num_examples = choose_context_count(max_num_examples)
    contexts = [sample_context(num_examples, row, tokenizer_mlm) for row in batch]
    input_batch = [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in batch]
    base_batch = [{"input_ids": row["base_input_ids"], "attention_mask": row["base_attention_mask"]} for row in batch]

    input_collate = dict(data_collator(input_batch))
    base_collate = {"base_" + key: value for key, value in data_collator(base_batch).items()}
    final_collate = {**input_collate, **base_collate, "contexts": contexts}
    if definition_training:
        final_collate["definitions"] = [
            tokenizer_mlm(row["definition"], max_length=256, truncation=True, return_tensors="pt") for row in batch
        ]
    return final_collate


def build_memory_config():
    return AggregatorConfig()


def load_tokenizers(args):
    if args.resume_from_checkpoint is not None:
        tokenizer_mlm = AutoTokenizer.from_pretrained(os.path.join(args.resume_from_checkpoint, "tokenizerMLM"), use_fast=False)
        tokenizer_task = LlamaTokenizer.from_pretrained(
            os.path.join(args.resume_from_checkpoint, "tokenizerTask"),
            legacy=True,
            use_fast=False,
        )
        return tokenizer_mlm, tokenizer_task

    tokenizer_mlm = AutoTokenizer.from_pretrained(args.first_lm, use_fast=False)
    tokenizer_task = LlamaTokenizer.from_pretrained(args.second_lm, legacy=True, use_fast=False)
    tokenizer_task.add_bos_token = True
    tokenizer_task.pad_token = tokenizer_task.unk_token
    return tokenizer_mlm, tokenizer_task


def build_nonces(args):
    if args.word_path:
        word_dict = load_from_disk(args.word_path)
        words = word_dict["train"]["words"] + word_dict["test"]["words"]
        return sorted(set(f"<{word.lower()}_new>" for word in words))

    nonces = ["<nonce>"]
    if args.definition_training:
        nonces.append("<def>")
    return nonces


def load_models(args, accelerator):
    with accelerator.main_process_first():
        first_lm = RobertaForMaskedLM.from_pretrained(args.first_lm, low_cpu_mem_usage=True).to(accelerator.device)
        second_lm = LlamaForCausalLM.from_pretrained(args.second_lm, low_cpu_mem_usage=True).to(accelerator.device)

    first_lm.eval()
    return first_lm, second_lm


def build_train_eval_dataloaders(args, dataset, tokenizer_mlm, tokenizer_task, data_collator):
    if args.regression_objective:
        train = dataset["train"].map(
            partial(tokenize_regression_example, tokenizer_task=tokenizer_task, definition_training=args.definition_training),
            remove_columns=[name for name in dataset["train"].column_names if name != "sentences"],
            num_proc=2,
        ).with_format("torch")
        test = dataset["test"].map(
            partial(tokenize_regression_example, tokenizer_task=tokenizer_task, definition_training=args.definition_training),
            remove_columns=[name for name in dataset["test"].column_names if name != "sentences"],
            num_proc=2,
        ).with_format("torch")
        collate_fn = partial(
            regression_collate,
            args.num_examples,
            tokenizer_mlm=tokenizer_mlm,
            data_collator=data_collator,
            definition_training=args.definition_training,
        )
    else:
        train = dataset["train"].map(
            partial(tokenize_example, tokenizer_task=tokenizer_task),
            remove_columns=[name for name in dataset["train"].column_names if name != "sentences"],
            num_proc=2,
        ).with_format("torch")
        test = dataset["test"].map(
            partial(tokenize_example, tokenizer_task=tokenizer_task),
            remove_columns=[name for name in dataset["test"].column_names if name != "sentences"],
            num_proc=2,
        ).with_format("torch")
        collate_fn = partial(
            regular_collate,
            args.num_examples,
            tokenizer_mlm=tokenizer_mlm,
            data_collator=data_collator,
        )

    train_dl = DataLoader(
        train,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        pin_memory=True,
    )
    return train, test, train_dl, test_dl, collate_fn


def build_negative_dataloaders(args, dataset, tokenizer_task, data_collator):
    if not args.negative_examples:
        return None, None

    negative_dataset = load_from_disk(args.negative_data_path)
    negative_train = negative_dataset["train"]
    negative_test = negative_dataset["test"]
    if negative_train.num_rows > dataset["train"].num_rows:
        negative_train = negative_train.select(list(range(dataset["train"].num_rows)))
    if negative_test.num_rows > dataset["test"].num_rows:
        negative_test = negative_test.select(list(range(dataset["test"].num_rows)))

    negative_train = negative_train.map(
        partial(tokenize_example, tokenizer_task=tokenizer_task),
        remove_columns=negative_dataset["train"].column_names,
        num_proc=2,
    ).with_format("torch")
    negative_test = negative_test.map(
        partial(tokenize_example, tokenizer_task=tokenizer_task),
        remove_columns=negative_dataset["test"].column_names,
        num_proc=2,
    ).with_format("torch")

    negative_train_dl = DataLoader(
        negative_train,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        pin_memory=True,
    )
    negative_test_dl = DataLoader(
        negative_test,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        pin_memory=True,
    )
    return negative_train_dl, negative_test_dl


def next_or_restart(iterator, dataloader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def compute_training_loss(args, out):
    if not args.regression_objective:
        return out.loss

    if getattr(out, "positive_loss", None) is not None and getattr(out, "negative_loss", None) is not None:
        loss = args.lm_alpha * out.positive_loss + args.negatives_alpha * out.negative_loss
    else:
        loss = args.lm_alpha * out.loss

    if not args.ablate_cosine:
        loss = loss + args.cosine_alpha * out.regression_loss
    if not args.ablate_logits:
        loss = loss + args.logits_alpha * out.distillation_loss
    return loss


def add_l2_loss(args, loss, out):
    if args.l2 is None:
        return loss

    input_l2_losses = []
    output_l2_losses = []
    for mem_dict in out.memories:
        for memory_type in ["input_memory", "output_memory"]:
            memory = mem_dict[memory_type]
            new_ids = list(memory.memory.keys())
            if len(new_ids) != 1:
                raise ValueError("Expected exactly one generated token per memory entry.")
            if memory_type == "input_memory":
                input_l2_losses.append(memory.retrieve(new_ids[0]).norm())
            else:
                output_l2_losses.append(memory.retrieve(new_ids[0]).norm())

    input_l2_loss = torch.stack(input_l2_losses).mean()
    output_l2_loss = torch.stack(output_l2_losses).mean()
    return loss + args.l2 * (input_l2_loss + output_l2_loss)


def zero_metric(device):
    return torch.tensor(0.0, device=device)


def maybe_add(value, increment):
    if increment is None:
        return value
    return value + increment.detach().float()


def parse_checkpoint_progress(checkpoint_path: str):
    match = re.search(r"checkpoint_(\d+)_(\d+)", checkpoint_path)
    if match is None:
        return 0, 0
    epoch, step = match.groups()
    return int(epoch), int(step)


def save_checkpoint(accelerator, tokenizer_mlm, tokenizer_task, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(save_dir)
    tokenizer_mlm.save_pretrained(os.path.join(save_dir, "tokenizerMLM"))
    tokenizer_task.save_pretrained(os.path.join(save_dir, "tokenizerTask"))


def main(argv=None):
    torch.manual_seed(0)
    np.random.seed(0)

    args = build_llama_training_parser().parse_args(argv)
    validate_args(args)

    checkpoint_path = create_checkpoint_directories(args)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    accelerator.wait_for_everyone()

    tokenizer_mlm, tokenizer_task = load_tokenizers(args)
    nonces = build_nonces(args)
    tokenizer_mlm.add_tokens(nonces)
    tokenizer_task.add_tokens(nonces)
    mask_token_id = tokenizer_mlm.mask_token_id

    first_lm, second_lm = load_models(args, accelerator)
    memory_config = build_memory_config()
    layers = [-index for index in range(args.layer, args.layer + args.num_feature_layers)]
    model = CollegeLlama(
        first_lm,
        second_lm,
        len(nonces),
        layers,
        mask_token_id,
        memory_config,
        args.num_layers,
        args.distillation_temp,
    ).to(accelerator.device)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_task, mlm=False, return_tensors="pt")
    dataset = load_from_disk(args.data_path)
    _, _, train_dl, test_dl, _ = build_train_eval_dataloaders(
        args,
        dataset,
        tokenizer_mlm,
        tokenizer_task,
        data_collator,
    )
    negative_train_dl, negative_test_dl = build_negative_dataloaders(args, dataset, tokenizer_task, data_collator)

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                param
                for name, param in model.emb_gen.named_parameters()
                if not any(nd in name for nd in no_decay) and param.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                param
                for name, param in model.emb_gen.named_parameters()
                if any(nd in name for nd in no_decay) and param.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    opt = AdamW(optimizer_grouped_parameters, eps=1e-8, lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = int(args.epochs * (len(train_dl) / args.gradient_accumulation_steps) * 0.03)
    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, args.epochs * len(train_dl))

    if args.negative_examples:
        model.emb_gen, opt, train_dl, test_dl, scheduler, negative_train_dl, negative_test_dl = accelerator.prepare(
            model.emb_gen,
            opt,
            train_dl,
            test_dl,
            scheduler,
            negative_train_dl,
            negative_test_dl,
        )
    else:
        model.emb_gen, opt, train_dl, test_dl, scheduler = accelerator.prepare(
            model.emb_gen,
            opt,
            train_dl,
            test_dl,
            scheduler,
        )

    accelerator.register_for_checkpointing(opt)
    accelerator.register_for_checkpointing(scheduler)
    accelerator.wait_for_everyone()
    accelerator.init_trackers(
        project_name=args.project_name,
        config={
            "num_examples": args.num_examples,
            "learning_rate": args.lr,
            "aggregation": memory_config.agg_method,
            "batch_size": args.batch_size,
            "negative_examples": args.negative_examples,
            "regression": args.regression_objective,
            "alpha": args.regression_alpha,
        },
    )

    base_epoch = 0
    global_step = 0
    active_train_dl = train_dl
    active_negative_train_dl = negative_train_dl
    if args.resume_from_checkpoint is not None:
        base_epoch, global_step = parse_checkpoint_progress(args.resume_from_checkpoint)
        within_batch_step = global_step if base_epoch == 0 else args.gradient_accumulation_steps * (
            global_step - (base_epoch * len(train_dl)) + 1
        )
        active_train_dl = accelerator.skip_first_batches(train_dl, within_batch_step)
        if args.negative_examples:
            active_negative_train_dl = accelerator.skip_first_batches(negative_train_dl, within_batch_step)
        accelerator.load_state(args.resume_from_checkpoint)

    best_test_loss = float("inf")
    best_new_token_loss = float("inf")

    for epoch in range(base_epoch, args.epochs):
        if epoch > base_epoch and args.resume_from_checkpoint is not None:
            active_train_dl = train_dl
            active_negative_train_dl = negative_train_dl

        negative_train_iter = iter(active_negative_train_dl) if args.negative_examples else None
        total_loss = zero_metric(accelerator.device)
        total_new_token_loss = zero_metric(accelerator.device)
        total_positive_loss = zero_metric(accelerator.device)
        total_negative_loss = zero_metric(accelerator.device)
        total_regression_loss = zero_metric(accelerator.device)
        total_distillation_loss = zero_metric(accelerator.device)

        for batch_index, batch in enumerate(active_train_dl):
            with accelerator.accumulate(model):
                log_dict = {}
                model.train()
                model.firstLM.eval()

                if args.negative_examples:
                    neg_train_batch, negative_train_iter = next_or_restart(negative_train_iter, active_negative_train_dl)
                    batch["negative_input_ids"] = neg_train_batch["input_ids"]
                    batch["negative_attention_mask"] = neg_train_batch["attention_mask"]
                    batch["negative_labels"] = neg_train_batch["labels"]

                out = model(batch)
                loss = add_l2_loss(args, compute_training_loss(args, out), out)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    for name, param in model.named_parameters():
                        if param.grad is not None and param.requires_grad:
                            grad_norm = torch.norm(param.grad.view(-1))
                            log_dict[f"gradients/post_{name}_grad_norm"] = grad_norm.item()
                            if torch.isnan(grad_norm):
                                raise FloatingPointError(f"NaN gradient for {name}.")

                opt.step()
                scheduler.step()
                opt.zero_grad()
                model.zero_grad()

                total_loss = total_loss + loss.detach().float()
                total_new_token_loss = maybe_add(total_new_token_loss, out.new_token_loss)
                if args.negative_examples:
                    total_positive_loss = maybe_add(total_positive_loss, out.positive_loss)
                    total_negative_loss = maybe_add(total_negative_loss, out.negative_loss)
                if args.regression_objective:
                    total_regression_loss = maybe_add(total_regression_loss, out.regression_loss)
                    total_distillation_loss = maybe_add(total_distillation_loss, out.distillation_loss)

            if accelerator.sync_gradients:
                global_step += 1
                log_dict["global_step"] = global_step
                log_dict["train/loss"] = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps
                log_dict["train/new_token_loss"] = (
                    accelerator.gather(total_new_token_loss).mean().item() / args.gradient_accumulation_steps
                )
                total_loss = zero_metric(accelerator.device)
                total_new_token_loss = zero_metric(accelerator.device)

                if args.negative_examples:
                    log_dict["train/positive_loss"] = (
                        accelerator.gather(total_positive_loss).mean().item() / args.gradient_accumulation_steps
                    )
                    log_dict["train/negative_loss"] = (
                        accelerator.gather(total_negative_loss).mean().item() / args.gradient_accumulation_steps
                    )
                    total_positive_loss = zero_metric(accelerator.device)
                    total_negative_loss = zero_metric(accelerator.device)

                if args.regression_objective:
                    log_dict["train/regression_loss"] = (
                        accelerator.gather(total_regression_loss).mean().item() / args.gradient_accumulation_steps
                    )
                    log_dict["train/distillation_loss"] = (
                        accelerator.gather(total_distillation_loss).mean().item() / args.gradient_accumulation_steps
                    )
                    total_regression_loss = zero_metric(accelerator.device)
                    total_distillation_loss = zero_metric(accelerator.device)

                with torch.no_grad():
                    memory_norms = {"input_memory": [], "output_memory": []}
                    for mem_dict in out.memories:
                        for memory_type in ["input_memory", "output_memory"]:
                            memory = mem_dict[memory_type]
                            new_id = next(iter(memory.memory))
                            memory_norms[memory_type].append(memory.retrieve(new_id).norm().detach())
                    for memory_type, norms in memory_norms.items():
                        log_dict[f"embed_norms/{memory_type}_token_embedding_norm"] = torch.stack(norms).mean().item()

                accelerator.log(log_dict)

            should_eval = (
                global_step != 0
                and global_step % args.logging_step == 0
                and batch_index % args.gradient_accumulation_steps == 0
                and batch_index != 0
            ) or (batch_index % len(active_train_dl) == 0 and batch_index != 0 and epoch != 0)

            if should_eval:
                opt.zero_grad(set_to_none=True)
                model.eval()
                negative_test_iter = iter(negative_test_dl) if args.negative_examples else None
                total_test_loss = zero_metric(accelerator.device)
                total_test_nonce_loss = zero_metric(accelerator.device)
                total_test_negative_loss = zero_metric(accelerator.device)
                total_test_positive_loss = zero_metric(accelerator.device)
                total_test_regression_loss = zero_metric(accelerator.device)
                total_test_distillation_loss = zero_metric(accelerator.device)
                test_count = 0

                with torch.no_grad():
                    for test_index, test_batch in enumerate(test_dl):
                        if test_index >= args.num_eval_steps:
                            break
                        if args.negative_examples:
                            neg_test_batch, negative_test_iter = next_or_restart(negative_test_iter, negative_test_dl)
                            test_batch["negative_input_ids"] = neg_test_batch["input_ids"]
                            test_batch["negative_attention_mask"] = neg_test_batch["attention_mask"]
                            test_batch["negative_labels"] = neg_test_batch["labels"]

                        test_out = model(test_batch)
                        test_loss = compute_training_loss(args, test_out)
                        total_test_loss = total_test_loss + test_loss.detach().float()
                        total_test_nonce_loss = maybe_add(total_test_nonce_loss, test_out.new_token_loss)
                        if args.negative_examples:
                            total_test_positive_loss = maybe_add(total_test_positive_loss, test_out.positive_loss)
                            total_test_negative_loss = maybe_add(total_test_negative_loss, test_out.negative_loss)
                        if args.regression_objective:
                            total_test_regression_loss = maybe_add(total_test_regression_loss, test_out.regression_loss)
                            total_test_distillation_loss = maybe_add(total_test_distillation_loss, test_out.distillation_loss)
                        test_count += 1

                denom = max(test_count, 1)
                test_log = {
                    "test/average_loss": accelerator.gather(total_test_loss).mean().item() / denom,
                    "test/average_new_token_loss": accelerator.gather(total_test_nonce_loss).mean().item() / denom,
                    "epoch": epoch,
                    "eval_step": batch_index // args.logging_step,
                }
                if args.negative_examples:
                    test_log["test/average_positive_loss"] = accelerator.gather(total_test_positive_loss).mean().item() / denom
                    test_log["test/average_negative_loss"] = accelerator.gather(total_test_negative_loss).mean().item() / denom
                if args.regression_objective:
                    test_log["test/average_regression_loss"] = (
                        accelerator.gather(total_test_regression_loss).mean().item() / denom
                    )
                    test_log["test/average_distillation_loss"] = (
                        accelerator.gather(total_test_distillation_loss).mean().item() / denom
                    )
                if args.run_gre_eval:
                    from college.evaluation.gre import gre_eval

                    gre_scores = gre_eval(
                        emb_gen_model=model,
                        tokenizer_mlm=tokenizer_mlm,
                        tokenizer_task=tokenizer_task,
                        device=accelerator.device,
                    )
                    for key, value in gre_scores.items():
                        test_log[f"gre_scores/gre_acc_at_{key}"] = value

                accelerator.log(test_log)

                avg_test = test_log["test/average_loss"]
                avg_new_tok = test_log["test/average_new_token_loss"]
                if (avg_test < best_test_loss or avg_new_tok < best_new_token_loss) and args.saving:
                    best_test_loss = min(best_test_loss, avg_test)
                    best_new_token_loss = min(best_new_token_loss, avg_new_tok)
                    save_dir = os.path.join(checkpoint_path, f"checkpoint_{epoch}_{global_step}")
                    copy_index = 0
                    unique_save_dir = save_dir
                    while os.path.isdir(unique_save_dir):
                        copy_index += 1
                        unique_save_dir = save_dir + f"_v{copy_index}"
                    save_checkpoint(accelerator, tokenizer_mlm, tokenizer_task, unique_save_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
