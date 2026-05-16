from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from .outputs import (
    CausalLMOutputWithNewToken,
    CausalLMOutputWithNewTokenNegatives,
    CausalLMOutputWithRegressionAndNegativeLoss,
    CausalLMOutputWithRegressionLoss,
)
from .utils import combine_layers, get_matching_indices


class Memory:
    def __init__(self):
        self.memory = {}

    def store(self, nonce: int, emb: torch.Tensor) -> None:
        self.memory[nonce] = emb

    def retrieve(self, nonce: int) -> torch.Tensor:
        return self.memory[nonce]

    def __contains__(self, nonce: int) -> bool:
        return nonce in self.memory

    def detach(self) -> None:
        for key, value in self.memory.items():
            self.memory[key] = value.detach()


class EmbeddingGenerator(nn.Module):
    def __init__(self, first_lm, second_lm_hidden_size: int, num_layers: int):
        super().__init__()
        self.input_hidden_size = first_lm.config.hidden_size
        self.output_hidden_size = second_lm_hidden_size
        self.num_attention_heads = first_lm.config.num_attention_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_hidden_size,
            nhead=self.num_attention_heads,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(self.input_hidden_size)

        self.input_emb_head = nn.Linear(self.input_hidden_size, self.output_hidden_size)
        self.output_emb_head = nn.Linear(self.input_hidden_size, self.output_hidden_size)

    def calc_init_std(self, desired_mean_norm: torch.Tensor) -> float:
        desired_mean_norm = float(desired_mean_norm.detach().cpu())
        return sqrt(((desired_mean_norm / sqrt(self.input_hidden_size)) ** 2) / self.output_hidden_size)

    def init_weights(self, input_embed_mean: torch.Tensor, output_embed_mean: torch.Tensor) -> None:
        input_std = self.calc_init_std(input_embed_mean)
        output_std = self.calc_init_std(output_embed_mean)

        self.input_emb_head.weight.data.normal_(mean=0.0, std=input_std)
        if self.input_emb_head.bias is not None:
            self.input_emb_head.bias.zero_()

        self.output_emb_head.weight.data.normal_(mean=0.0, std=output_std)
        if self.output_emb_head.bias is not None:
            self.output_emb_head.bias.zero_()

    def forward(self, inputs: torch.Tensor, attn_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        concept_embed = self.get_concept_embedding(inputs, attn_mask)
        return self.input_emb_head(concept_embed), self.output_emb_head(concept_embed)

    @torch.no_grad()
    def get_embeds(
        self,
        inputs: torch.Tensor,
        attn_mask: torch.Tensor,
        aux_embed: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        concept_embed = self.get_concept_embedding(inputs, attn_mask)
        out = concept_embed + aux_embed if aux_embed is not None else concept_embed
        return self.input_emb_head(out), self.output_emb_head(out), out

    def get_concept_embedding(self, inputs: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(inputs, src_key_padding_mask=~attn_mask.bool())
        out = self.norm(out)

        out = torch.sum(out * attn_mask.unsqueeze(-1), dim=1) / torch.sum(attn_mask, dim=-1, keepdim=True)
        return torch.mean(out, dim=0, keepdim=True)

    @torch.no_grad()
    def get_input_and_output_embedding(self, concept_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_emb_head(concept_embed), self.output_emb_head(concept_embed)


class CollegeLlama(nn.Module):
    def __init__(
        self,
        first_lm,
        second_lm,
        num_new_tokens: int,
        layers: List[int],
        mask_token_id: Optional[int],
        memory_config,
        num_layers: int,
        distillation_temp: float,
    ):
        super().__init__()
        self.layers = layers
        self.mask_token_id = mask_token_id
        self.firstLM = first_lm
        self.secondLM = second_lm
        self.memory_config = memory_config
        self.num_new_tokens = num_new_tokens
        self.num_layers = num_layers
        self.distillation_temp = distillation_temp

        self.emb_gen = EmbeddingGenerator(
            self.firstLM,
            self.secondLM.config.hidden_size,
            num_layers,
        )
        self.model_name = f"{self.secondLM.config.model_type}_{memory_config.agg_method}"

        with torch.no_grad():
            output_mean_embed = torch.mean(self.secondLM.get_output_embeddings().weight.norm(dim=1))
            input_mean_embed = torch.mean(self.secondLM.get_input_embeddings().weight.norm(dim=1))
            self.emb_gen.init_weights(input_mean_embed, output_mean_embed)

        self.freeze()

    @staticmethod
    def resolve_checkpoint_child(checkpoint_path: str, name: str) -> str:
        checkpoint = Path(checkpoint_path)
        child = checkpoint / name
        if child.exists():
            return str(child)

        legacy_child = Path(str(checkpoint) + name)
        if legacy_child.exists():
            return str(legacy_child)

        return str(child)

    @staticmethod
    def _strip_state_prefix(state_dict, prefix: str):
        if not all(key.startswith(prefix) for key in state_dict):
            return state_dict
        return {key[len(prefix) :]: value for key, value in state_dict.items()}

    @classmethod
    def load_embedding_generator_state(cls, checkpoint_path: str):
        checkpoint = Path(checkpoint_path)
        pytorch_state = checkpoint / "pytorch_model.bin"
        if pytorch_state.exists():
            state = torch.load(str(pytorch_state), map_location="cpu")
        else:
            safetensors_state = checkpoint / "model.safetensors"
            if not safetensors_state.exists():
                raise FileNotFoundError(
                    f"Could not find pytorch_model.bin or model.safetensors in checkpoint {checkpoint_path!r}."
                )
            from safetensors.torch import load_file

            state = load_file(str(safetensors_state), device="cpu")

        if "state_dict" in state:
            state = state["state_dict"]
        state = cls._strip_state_prefix(state, "module.")
        state = cls._strip_state_prefix(state, "emb_gen.")
        return state

    def load_embedding_generator_checkpoint(self, checkpoint_path: str, strict: bool = True):
        state = self.load_embedding_generator_state(checkpoint_path)
        return self.emb_gen.load_state_dict(state, strict=strict)

    def save_embedding_generator_checkpoint(self, output_dir: str, tokenizer_mlm=None, tokenizer_task=None) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.emb_gen.state_dict(), output_path / "pytorch_model.bin")
        if tokenizer_mlm is not None:
            tokenizer_mlm.save_pretrained(output_path / "tokenizerMLM")
        if tokenizer_task is not None:
            tokenizer_task.save_pretrained(output_path / "tokenizerTask")
        return str(output_path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
        first_lm: str = "roberta-large",
        second_lm: Optional[str] = None,
        layer: int = 1,
        num_feature_layers: int = 1,
        num_layers: int = 1,
        distillation_temp: float = 1.0,
        strict: bool = True,
        return_tokenizers: bool = False,
    ):
        if "roberta" not in first_lm.lower():
            raise ValueError(f"Checkpoint loading only supports RoBERTa first_lm checkpoints, got {first_lm!r}.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if second_lm is None:
            from .arguments import DEFAULT_LLAMA_PATH

            second_lm = DEFAULT_LLAMA_PATH

        from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, RobertaForMaskedLM

        from .configs import AggregatorConfig

        tokenizer_mlm = AutoTokenizer.from_pretrained(
            cls.resolve_checkpoint_child(checkpoint_path, "tokenizerMLM"),
            use_fast=False,
        )
        tokenizer_task = LlamaTokenizer.from_pretrained(
            cls.resolve_checkpoint_child(checkpoint_path, "tokenizerTask"),
            use_fast=False,
            legacy=True,
        )
        nonces = list(tokenizer_task.get_added_vocab().keys())
        if not nonces:
            nonces = ["<nonce>"]
            tokenizer_mlm.add_tokens(nonces)
            tokenizer_task.add_tokens(nonces)

        first_model = RobertaForMaskedLM.from_pretrained(first_lm, low_cpu_mem_usage=True)
        second_model = LlamaForCausalLM.from_pretrained(second_lm, low_cpu_mem_usage=True)
        layers = [-index for index in range(layer, layer + num_feature_layers)]
        model = cls(
            first_model,
            second_model,
            len(nonces),
            layers,
            tokenizer_mlm.mask_token_id,
            AggregatorConfig(),
            num_layers,
            distillation_temp=distillation_temp,
        ).to(device)
        model.load_embedding_generator_checkpoint(checkpoint_path, strict=strict)
        model.device = device
        model.firstLM.eval()
        model.secondLM.eval()
        model.eval()

        if return_tokenizers:
            return model, tokenizer_mlm, tokenizer_task
        return model

    @property
    def first_list(self) -> List[int]:
        return list(range(self.firstLM.config.vocab_size, self.firstLM.config.vocab_size + self.num_new_tokens))

    @property
    def second_list(self) -> List[int]:
        return list(range(self.secondLM.config.vocab_size, self.secondLM.config.vocab_size + self.num_new_tokens))

    @property
    def initial_first_ind(self) -> int:
        return self.firstLM.config.vocab_size

    @property
    def initial_second_ind(self) -> int:
        return self.secondLM.config.vocab_size

    def freeze(self) -> None:
        for parameter in self.firstLM.parameters():
            parameter.requires_grad = False
        for parameter in self.secondLM.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.secondLM.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.secondLM.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def swap_with_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        inp = inputs.clone()
        for nonce in self.first_list:
            inp[inp == nonce] = self.mask_token_id
        return inp

    def get_new_output_weights(self, new_embed: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.secondLM.lm_head.weight, new_embed])

    def get_new_weights(self, task: str, new_embed: torch.Tensor) -> torch.Tensor:
        if task == "MLM":
            ref_model = self.firstLM
        elif task == "Task":
            ref_model = self.secondLM
        else:
            raise NotImplementedError(f"Unsupported task: {task}")
        return torch.cat([ref_model.get_input_embeddings().weight, new_embed])

    def get_new_weights_definition_input(
        self,
        new_input_embed: torch.Tensor,
        def_input_embed: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([self.secondLM.get_input_embeddings().weight, new_input_embed, def_input_embed])

    def get_new_weights_definition_output(
        self,
        new_output_embed: torch.Tensor,
        def_output_embed: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([self.secondLM.get_output_embeddings().weight, new_output_embed, def_output_embed])

    def llama_forward(
        self,
        labels: Optional[torch.Tensor],
        outputs,
        new_w: torch.Tensor,
        index: Optional[int],
        new_token_loss: bool = False,
    ):
        hidden_states = outputs[0][index, :, :] if index is not None else outputs[0]
        if self.secondLM.config.pretraining_tp > 1:
            lm_head_slices = new_w.split(self.secondLM.config.vocab_size // self.secondLM.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.secondLM.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = F.linear(hidden_states, new_w, bias=self.secondLM.lm_head.bias)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, new_w.shape[0])
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if new_token_loss:
                nt_loss_fct = CrossEntropyLoss(reduction="none")
                non_reduced_loss = nt_loss_fct(shift_logits, shift_labels)
                new_token_ids = torch.tensor(self.second_list, device=shift_logits.device).unique()
                selected = non_reduced_loss[torch.where(torch.isin(torch.flatten(shift_labels), new_token_ids))]
                selected_loss = selected.mean() if selected.numel() > 0 else None
                return (
                    CausalLMOutputWithPast(
                        loss=loss,
                        logits=logits,
                        past_key_values=outputs.past_key_values,
                        hidden_states=outputs.hidden_states,
                        attentions=outputs.attentions,
                    ),
                    selected_loss,
                )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_new_input_weights_multi(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.secondLM.get_input_embeddings().weight, *embeds])

    def get_new_output_weights_multi(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.secondLM.get_output_embeddings().weight, *embeds])

    def multi_token_inference(self, batch: Dict[str, torch.Tensor]) -> CausalLMOutputWithNewToken:
        if "labels" not in batch:
            raise ValueError("Batch must contain labels.")

        contexts = batch["contexts"]
        task_ids = batch["input_ids"]
        task_attn = batch["attention_mask"]
        task_labels = batch["labels"]

        if len(contexts) != task_ids.shape[0]:
            raise ValueError("Context count must match task batch size.")

        embeds = []
        mem_embeds = []
        for i, ctx in enumerate(contexts):
            input_memory = Memory()
            output_memory = Memory()
            for context_item in ctx:
                first_token_ids = torch.tensor(self.first_list, device=context_item["input_ids"].device)
                new_token = context_item["input_ids"][torch.isin(context_item["input_ids"], first_token_ids)].unique()[0].item()
                mlm_ids = self.swap_with_mask(context_item["input_ids"])
                with torch.no_grad():
                    first_out = self.firstLM(
                        input_ids=mlm_ids,
                        attention_mask=context_item["attention_mask"],
                        output_hidden_states=True,
                    )
                combined = combine_layers(first_out.hidden_states, self.layers)
                if len(combined.shape) == 2:
                    combined = combined.unsqueeze(0)

                inp_embs, out_embs = self.emb_gen(combined, context_item["attention_mask"])
                input_memory.store(new_token, inp_embs)
                output_memory.store(new_token, out_embs)

            input_weights = [input_memory.retrieve(token) for token in sorted(input_memory.memory)]
            new_w = self.get_new_input_weights_multi(input_weights)
            embeds.append(F.embedding(task_ids[i], new_w))
            mem_embeds.append({"input_memory": input_memory, "output_memory": output_memory})

        input_embeds = torch.stack(embeds)
        outputs = self.secondLM.model(inputs_embeds=input_embeds, attention_mask=task_attn)

        outs = []
        for i, mem in enumerate(mem_embeds):
            out_embs = [mem["output_memory"].retrieve(token) for token in sorted(mem["output_memory"].memory)]
            output_weights = self.get_new_output_weights_multi(out_embs)
            llama_outputs, new_tok_loss = self.llama_forward(task_labels[i], outputs, output_weights, i, new_token_loss=True)
            outs.append(
                CausalLMOutputWithNewToken(
                    loss=llama_outputs.loss,
                    logits=llama_outputs.logits,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                    new_token_loss=new_tok_loss,
                    memories=[mem],
                )
            )

        return outs[0]

    def forward(self, batch: Dict[str, torch.Tensor], output_hidden_states: bool = False):
        if "labels" not in batch:
            raise ValueError("Batch must contain labels.")

        contexts = batch["contexts"]
        task_ids = batch["input_ids"]
        task_attn = batch["attention_mask"]
        task_labels = batch["labels"]
        batch_size = task_ids.shape[0]

        negative_ids = batch.get("negative_input_ids")
        negative_attn_mask = batch.get("negative_attention_mask")
        negative_labels = batch.get("negative_labels")
        base_ids = batch.get("base_input_ids")
        base_attn_mask = batch.get("base_attention_mask")
        base_labels = batch.get("base_labels")

        has_negatives = negative_ids is not None and negative_attn_mask is not None and negative_labels is not None
        has_base = base_ids is not None and base_attn_mask is not None and base_labels is not None

        if len(contexts) != batch_size:
            raise ValueError("Context count must match task batch size.")

        mem_embeds = []
        embeds = []
        neg_embeds = []
        first_token_ids = torch.tensor(self.first_list, device=task_ids.device).unique()

        for i in range(batch_size):
            context = contexts[i].to(self.firstLM.device)
            input_memory = Memory()
            output_memory = Memory()

            if self.mask_token_id is not None:
                new_token = context["input_ids"][torch.isin(context["input_ids"], first_token_ids)].unique()[0].item()
                mlm_ids = self.swap_with_mask(context["input_ids"])
            else:
                new_token = first_token_ids[0].item()
                mlm_ids = context["input_ids"]

            with torch.no_grad():
                first_out = self.firstLM(input_ids=mlm_ids, attention_mask=context["attention_mask"], output_hidden_states=True)
            combined = combine_layers(first_out.hidden_states, self.layers)
            if len(combined.shape) == 2:
                combined = combined.unsqueeze(0)

            inp_embs, out_embs = self.emb_gen(combined, context["attention_mask"])
            input_memory.store(new_token, inp_embs)
            output_memory.store(new_token, out_embs)

            new_w = self.get_new_weights(task="Task", new_embed=inp_embs)
            embeds.append(F.embedding(task_ids[i], new_w))
            mem_embeds.append({"input_memory": input_memory, "output_memory": output_memory})

            if has_negatives:
                neg_embeds.append(F.embedding(negative_ids[i], new_w))

        if has_base:
            with torch.no_grad():
                base_outputs = self.secondLM.model(input_ids=base_ids, attention_mask=base_attn_mask)
                base_final_outs = self.llama_forward(
                    base_labels,
                    base_outputs,
                    self.secondLM.get_output_embeddings().weight,
                    index=None,
                    new_token_loss=False,
                )
        else:
            base_outputs = None
            base_final_outs = None

        if has_negatives:
            input_embeds = torch.stack(embeds + neg_embeds)
            attn = torch.cat([task_attn, negative_attn_mask], dim=0)
        else:
            input_embeds = torch.stack(embeds)
            attn = task_attn

        outputs = self.secondLM.model(inputs_embeds=input_embeds, attention_mask=attn, output_hidden_states=output_hidden_states)

        outs = []
        for i, mem in enumerate(mem_embeds):
            new_token_id = next(iter(mem["output_memory"].memory))
            out_embs = mem["output_memory"].retrieve(new_token_id)
            output_weights = self.get_new_output_weights(new_embed=out_embs)
            llama_outputs, new_tok_loss = self.llama_forward(task_labels[i], outputs, output_weights, i, new_token_loss=True)

            if has_negatives:
                negative_llama_outputs = self.llama_forward(negative_labels[i], outputs, output_weights, i + batch_size)
                out_vals = CausalLMOutputWithNewTokenNegatives(
                    loss=llama_outputs.loss + negative_llama_outputs.loss,
                    positive_loss=llama_outputs.loss,
                    negative_loss=negative_llama_outputs.loss,
                    positive_logits=None,
                    negative_logits=None,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                    new_token_loss=new_tok_loss,
                    memories=[mem],
                )
            else:
                out_vals = None

            if has_base:
                indices_in_base, indices_in_replaced = get_matching_indices(
                    base_ids[i][base_attn_mask[i] == 1].tolist(),
                    task_ids[i][task_attn[i] == 1].tolist(),
                )

                cosine_loss = nn.CosineEmbeddingLoss()
                regression_loss = cosine_loss(
                    outputs[0][i, indices_in_replaced],
                    base_outputs[0][i, indices_in_base],
                    target=torch.ones(outputs[0][i, indices_in_replaced].shape[0], device=base_outputs[0].device),
                ).mean()

                mse_loss = MSELoss()
                distillation_loss = mse_loss(
                    llama_outputs.logits[indices_in_replaced, : self.initial_second_ind],
                    base_final_outs.logits[i, indices_in_base, : self.initial_second_ind],
                )

                regression_out_vals = CausalLMOutputWithRegressionLoss(
                    loss=llama_outputs.loss,
                    regression_loss=regression_loss,
                    distillation_loss=distillation_loss,
                    new_token_loss=new_tok_loss,
                    memories=[mem],
                )
                if has_negatives:
                    out_vals = CausalLMOutputWithRegressionAndNegativeLoss(
                        loss=out_vals.loss,
                        hidden_states=None,
                        positive_loss=out_vals.positive_loss,
                        negative_loss=out_vals.negative_loss,
                        positive_logits=None,
                        negative_logits=None,
                        base_logits=None,
                        base_hidden_states=None,
                        past_key_values=None,
                        attentions=None,
                        new_token_loss=new_tok_loss,
                        memories=[mem],
                        regression_loss=regression_loss,
                        distillation_loss=distillation_loss,
                    )
                else:
                    out_vals = regression_out_vals

            if out_vals is None:
                out_vals = CausalLMOutputWithNewToken(
                    loss=llama_outputs.loss,
                    logits=llama_outputs.logits,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                    new_token_loss=new_tok_loss,
                    memories=[mem],
                )
            outs.append(out_vals)

        final_loss = torch.stack([out.loss for out in outs]).mean()
        final_new_token_losses = [out.new_token_loss for out in outs if out.new_token_loss is not None]
        final_new_token_loss = torch.stack(final_new_token_losses).mean() if final_new_token_losses else None
        final_memories = [out.memories[0] for out in outs]

        if has_negatives:
            final_positive_loss = torch.stack([out.positive_loss for out in outs]).mean()
            final_negative_loss = torch.stack([out.negative_loss for out in outs]).mean()

        if has_base:
            final_regression_loss = torch.stack([out.regression_loss for out in outs]).mean()
            final_distillation_loss = torch.stack([out.distillation_loss for out in outs]).mean()
            final_base_hiddens = [out.base_hidden_states for out in outs]

        if has_negatives and has_base:
            return CausalLMOutputWithRegressionAndNegativeLoss(
                loss=final_loss,
                hidden_states=[out.hidden_states for out in outs],
                positive_loss=final_positive_loss,
                negative_loss=final_negative_loss,
                positive_logits=None,
                negative_logits=None,
                base_logits=None,
                base_hidden_states=final_base_hiddens,
                new_token_loss=final_new_token_loss,
                memories=final_memories,
                regression_loss=final_regression_loss,
                distillation_loss=final_distillation_loss,
            )

        if has_negatives:
            return CausalLMOutputWithNewTokenNegatives(
                loss=final_loss,
                positive_loss=final_positive_loss,
                negative_loss=final_negative_loss,
                positive_logits=None,
                negative_logits=None,
                hidden_states=[out.hidden_states for out in outs],
                attentions=[out.attentions for out in outs],
                new_token_loss=final_new_token_loss,
                memories=final_memories,
            )

        if has_base:
            return CausalLMOutputWithRegressionLoss(
                loss=final_loss,
                logits=None,
                base_logits=None,
                hidden_states=[out.hidden_states for out in outs],
                base_hidden_states=final_base_hiddens,
                new_token_loss=final_new_token_loss,
                memories=final_memories,
                regression_loss=final_regression_loss,
                distillation_loss=final_distillation_loss,
            )

        final_logits = torch.cat([out.logits for out in outs if out.logits is not None], dim=0)
        return CausalLMOutputWithNewToken(
            loss=final_loss,
            logits=final_logits,
            hidden_states=outputs.hidden_states,
            attentions=None,
            past_key_values=None,
            new_token_loss=final_new_token_loss,
            memories=final_memories,
        )
