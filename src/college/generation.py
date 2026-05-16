from typing import Optional

import torch
import torch.nn.functional as F


def decoding_step(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int] = None,
    do_sample: bool = False,
    new_token_idx: int = 32000,
    mask_new_tokens: bool = False,
) -> torch.Tensor:
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    scaled_logits = logits[:, -1, :] / temperature
    if top_k is not None:
        values, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
        scaled_logits[scaled_logits < values[:, [-1]]] = -float("Inf")

    probs = F.softmax(scaled_logits, dim=-1)
    if mask_new_tokens:
        probs[:, new_token_idx:] = 0.0
        probs = probs / probs.sum()

    if do_sample:
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(probs, keepdim=True)


@torch.no_grad()
def generate(
    model,
    context,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    do_sample: bool = False,
    mask_new_tokens: bool = False,
) -> torch.Tensor:
    initial_batch = {
        "contexts": [context],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }
    initial_outputs = model(initial_batch)
    new_tok_id = next(iter(initial_outputs.memories[0]["input_memory"].memory))
    inp_embed = initial_outputs.memories[0]["input_memory"].retrieve(new_tok_id)
    outp_embed = initial_outputs.memories[0]["output_memory"].retrieve(new_tok_id)
    input_weights = model.get_new_weights(task="Task", new_embed=inp_embed)
    output_weights = model.get_new_output_weights(outp_embed)

    first_token = decoding_step(initial_outputs.logits, temperature, top_k, do_sample, mask_new_tokens=mask_new_tokens)
    new_input_ids = torch.cat([input_ids, first_token], dim=1)
    new_attention_mask = torch.cat([attention_mask, attention_mask[:, -1].unsqueeze(1)], dim=1)

    for _ in range(1, max_new_tokens):
        input_embeds = F.embedding(new_input_ids, input_weights)
        outputs = model.secondLM.model(inputs_embeds=input_embeds, attention_mask=new_attention_mask)
        llama_outputs = model.llama_forward(labels=None, outputs=outputs, new_w=output_weights, index=None)
        next_token = decoding_step(llama_outputs.logits, temperature, top_k, do_sample, mask_new_tokens=mask_new_tokens)
        new_input_ids = torch.cat([new_input_ids, next_token], dim=1)
        new_attention_mask = torch.cat([new_attention_mask, new_attention_mask[:, -1].unsqueeze(1)], dim=1)

    return new_input_ids


@torch.no_grad()
def generate_multi(
    model,
    context,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    do_sample: bool = False,
    mask_new_tokens: bool = False,
) -> torch.Tensor:
    initial_batch = {
        "contexts": [context],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }
    initial_outputs = model.multi_token_inference(initial_batch)
    input_memory = initial_outputs.memories[0]["input_memory"]
    output_memory = initial_outputs.memories[0]["output_memory"]
    inp_embed = [input_memory.retrieve(token) for token in sorted(input_memory.memory)]
    outp_embed = [output_memory.retrieve(token) for token in sorted(output_memory.memory)]
    input_weights = model.get_new_input_weights_multi(inp_embed)
    output_weights = model.get_new_output_weights_multi(outp_embed)

    first_token = decoding_step(initial_outputs.logits, temperature, top_k, do_sample=do_sample, mask_new_tokens=mask_new_tokens)
    new_input_ids = torch.cat([input_ids, first_token], dim=1)
    new_attention_mask = torch.cat([attention_mask, attention_mask[:, -1].unsqueeze(1)], dim=1)

    for _ in range(1, max_new_tokens):
        input_embeds = F.embedding(new_input_ids, input_weights)
        outputs = model.secondLM.model(inputs_embeds=input_embeds, attention_mask=new_attention_mask)
        llama_outputs = model.llama_forward(labels=None, outputs=outputs, new_w=output_weights, index=None)
        next_token = decoding_step(llama_outputs.logits, temperature, top_k, do_sample, mask_new_tokens=mask_new_tokens)
        new_input_ids = torch.cat([new_input_ids, next_token], dim=1)
        new_attention_mask = torch.cat([new_attention_mask, new_attention_mask[:, -1].unsqueeze(1)], dim=1)

    return new_input_ids
