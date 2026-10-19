# """
# llm_lora_decoder.py
# ====================
# Replaces BaseCMN (modules/base_cmn.py) as the report-generation decoder.
#
# Design: PAG (Progressive Anatomical Grounding)
#   - A 7B causal LLM is loaded in 4-bit (QLoRA-style) and kept FROZEN.
#   - Only LoRA adapters (attention projections) + two small connector MLPs
#     are trainable. This mirrors S2D-Align's "frozen visual encoder + frozen
#     LLM, only the adapter/connector is trained per curriculum stage" recipe,
#     except we also allow LoRA on the LLM (paper keeps GLLM fully frozen and
#     only trains SMA; LoRA is our added innovation for better domain
#     adaptation without full fine-tuning cost).
#   - VisualToLLMProjector plays the role of the paper's SMA output
#     projection: it maps ClinicalSlotAttention's plan_tokens (and, in
#     Stage 2, the TextMemoryAdapter's key-phrase tokens) from d_model into
#     the LLM's hidden size, so they can be prepended as soft-prompt /
#     prefix embeddings before the report tokens.
#
# Usage:
#     decoder = LoRAReportDecoder(args, d_model=args.d_model)
#     loss = decoder(plan_tokens, report_texts, aux_tokens=None)     # train
#     reports = decoder.generate(plan_tokens)                        # sample
# """
# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
#
#
# class VisualToLLMProjector(nn.Module):
#     """Connector MLP: d_model -> llm_hidden_size (SMA-equivalent output head)."""
#
#     def __init__(self, d_model, llm_hidden, dropout=0.1):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Linear(d_model, llm_hidden),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(llm_hidden, llm_hidden),
#         )
#         self.norm = nn.LayerNorm(llm_hidden)
#
#     def forward(self, x):
#         return self.norm(self.proj(x))
#
#
# class LoRAReportDecoder(nn.Module):
#     def __init__(self, args, plan_token_dim, text_token_dim):
#         super().__init__()
#         self.llm_name = getattr(args, 'llm_name', 'Qwen/Qwen2.5-7B-Instruct')
#         self.max_new_tokens = getattr(args, 'llm_max_new_tokens', 120)
#         self.max_report_len = getattr(args, 'llm_max_report_len', 200)
#
#         quant_cfg = None
#         if getattr(args, 'llm_4bit', True):
#             quant_cfg = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_quant_type='nf4',
#                 bnb_4bit_compute_dtype=torch.bfloat16,
#             )
#
#         # --- LLM's own tokenizer. NOTE: this is intentionally separate from
#         # modules/tokenizers.py's custom radiology vocabulary -- the LLM
#         # needs its native BPE tokenizer, so training/eval must feed RAW
#         # report strings to this decoder, not the old integer report_ids.
#         self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, use_fast=True)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.tokenizer.padding_side = 'right'
#
#         # base_llm = AutoModelForCausalLM.from_pretrained(
#         #     self.llm_name,
#         #     quantization_config=quant_cfg,
#         #     torch_dtype=torch.bfloat16,
#         # )
#         # llm_lora_decoder.py, __init__ mein
#         self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, use_fast=True)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.tokenizer.padding_side = 'right'
#
#         base_llm = AutoModelForCausalLM.from_pretrained(
#             self.llm_name,
#             quantization_config=quant_cfg,
#             torch_dtype=torch.float32,  # bf16 ki jagah, dtype masle khatam
#         )
#         if quant_cfg is not None:
#             base_llm = prepare_model_for_kbit_training(base_llm)
#         use_lora = getattr(args, 'use_lora', True)
#
#         if use_lora:
#
#             lora_cfg = LoraConfig( r=getattr(args, 'lora_r', 16),lora_alpha=getattr(args, 'lora_alpha', 32), lora_dropout=getattr(args, 'lora_dropout', 0.05),
#                 bias='none', task_type='CAUSAL_LM', target_modules=getattr(
#                     args,'lora_target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj'], ), )
#
#             self.llm = get_peft_model(base_llm, lora_cfg)
#
#             # Freeze original LLM weights
#             for name, p in self.llm.named_parameters():
#                 if 'lora_' not in name:
#                     p.requires_grad_(False)
#
#         else:
#
#             # Pure frozen LLM ablation
#             self.llm = base_llm
#
#             for p in self.llm.parameters():
#                 p.requires_grad_(False)
#
#         # lora_cfg = LoraConfig(
#         #     r=getattr(args, 'lora_r', 16),
#         #     lora_alpha=getattr(args, 'lora_alpha', 32),
#         #     lora_dropout=getattr(args, 'lora_dropout', 0.05),
#         #     bias='none',
#         #     task_type='CAUSAL_LM',
#         #     target_modules=getattr(
#         #         args, 'lora_target_modules',
#         #         ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
#         #     ),
#         # )
#         # self.llm = get_peft_model(base_llm, lora_cfg)
#         #
#         # # Belt-and-braces: keep every non-LoRA LLM weight frozen.
#         # for name, p in self.llm.named_parameters():
#         #     if 'lora_' not in name:
#         #         p.requires_grad_(False)
#
#         self.hidden_size = self.llm.config.hidden_size
#         self.hidden_size = self.llm.config.hidden_size
#         self.visual_projector = VisualToLLMProjector(plan_token_dim, self.hidden_size)  # was d_model
#         self.text_projector = VisualToLLMProjector(text_token_dim, self.hidden_size)  # stays d_model
#
#     # ------------------------------------------------------------------
#     def get_input_embeddings(self):
#         return self.llm.get_input_embeddings()
#
#     def _build_prefix(self, plan_tokens, aux_tokens=None):
#         # projector ke Linear layers float32 hain -> input bhi float32 mein force karo
#         prefix = self.visual_projector(plan_tokens.float())
#         if aux_tokens is not None:
#             aux = self.text_projector(aux_tokens.float())
#             prefix = torch.cat([prefix, aux], dim=1)
#         # LLM (bf16) ke text embeddings ke saath cat hone se pehle target dtype match karo
#         target_dtype = self.get_input_embeddings().weight.dtype
#         return prefix.to(target_dtype)
#
#     # ------------------------------------------------------------------
#     def forward(self, plan_tokens, report_texts, aux_tokens=None):
#         """
#         Teacher-forced training step.
#
#         plan_tokens:  (B, K, d_model) -- output of ClinicalSlotAttention.
#         report_texts: list[str], length B -- ground-truth report strings.
#         aux_tokens:   optional (B, K, d_model) -- Stage-2 key-phrase memory
#                       tokens from TextMemoryAdapter.
#
#         Returns: scalar CE loss (computed internally by the HF LLM).
#         """
#         device = plan_tokens.device
#         texts_with_eos = [str(t).strip() + self.tokenizer.eos_token for t in report_texts]
#
#         enc = self.tokenizer(
#             texts_with_eos, padding=True, truncation=True,
#             max_length=self.max_report_len, return_tensors='pt',
#         ).to(device)
#         report_ids = enc.input_ids
#         report_masks = enc.attention_mask
#
#         prefix_embeds = self._build_prefix(plan_tokens, aux_tokens)  # (B,P,H)
#         B, P, _ = prefix_embeds.shape
#
#         text_embeds = self.get_input_embeddings()(report_ids)  # (B,T,H)
#         inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
#
#         prefix_mask = torch.ones(B, P, device=device, dtype=report_masks.dtype)
#         attention_mask = torch.cat([prefix_mask, report_masks], dim=1)
#
#         # Ignore prefix + padding positions in the loss.
#         ignore = torch.full((B, P), -100, device=device, dtype=torch.long)
#         labels_text = report_ids.clone()
#         labels_text[report_masks == 0] = -100
#         labels = torch.cat([ignore, labels_text], dim=1)
#
#         out = self.llm(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             labels=labels,
#         )
#         return out.loss
#
#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def generate(self, plan_tokens, aux_tokens=None, **gen_kwargs):
#         """Inference: prefix-only (Stage 1) generation. Returns decoded
#         report strings directly -- no separate vocab/decode step needed."""
#         device = plan_tokens.device
#         prefix_embeds = self._build_prefix(plan_tokens, aux_tokens)
#         B, P, _ = prefix_embeds.shape
#         attention_mask = torch.ones(B, P, device=device, dtype=torch.long)
#
#         gen_kwargs.setdefault('max_new_tokens', self.max_new_tokens)
#         gen_kwargs.setdefault('do_sample', False)
#         gen_kwargs.setdefault('num_beams', getattr(self, 'beam_size', 5))
#
#         output_ids = self.llm.generate(
#             inputs_embeds=prefix_embeds,
#             attention_mask=attention_mask,
#             pad_token_id=self.tokenizer.pad_token_id,
#             eos_token_id=self.tokenizer.eos_token_id,  # <-- YE LINE add karo
#             **gen_kwargs,
#         )
#         return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#_________________ Bio_ClinicalBERT-----------------------------
# import os
# # Mirror & timeout set
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "800"
#
# from huggingface_hub import snapshot_download
#
# snapshot_download(
#     repo_id="emilyalsentzer/Bio_ClinicalBERT",
#     local_dir=r"E:\usman\GOU1\spg\Bio_ClinicalBERT",
#     resume_download=True,
#     force_download=False
# )
# from transformers import AutoModel, AutoTokenizer
# model_path = r"E:\usman\GOU1\spg\Bio_ClinicalBERT"
# model = AutoModel.from_pretrained(model_path, local_files_only=True)
# print("Successfully loaded text encoder")
#=================================== Qwen================================
# from modelscope import snapshot_download
# model_dir = snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir=r"E:\usman\GOU1\spg\Qwen2.5-7B-Instruct")
# print("Download Complete")
#=================================End qwen downlod
# from transformers import AutoTokenizer#
# tokenizer = AutoTokenizer.from_pretrained( r"E:\usman\GOU1\spg\Bio_ClinicalBERT",local_files_only=True)
# # print("BERT OK")
#=======================
# GPU check==========================================
import torch
print(torch.__version__)
print(torch.cuda.is_available())
# =============================End GPU Check===========