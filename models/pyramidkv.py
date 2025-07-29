# import torch
# from opencompass.models.base import BaseModel
# from opencompass.utils.logging import get_logger
# from transformers import AutoTokenizer
# from pyramidkv.monkeypatch import replace_llama, replace_mistral
# from transformers import AutoModelForCausalLM
# import numpy as np

# class PyramidKVModel(BaseModel):
#     def __init__(self,
#                  path: str,
#                  model_kwargs: dict = dict(),
#                  tokenizer_kwargs: dict = dict(),
#                  max_seq_len: int = 2048,
#                  generation_kwargs: dict = dict(),
#                  method: str = "pyramidkv",
#                  attn_implementation: str = "flash_attention_2",
#                  max_capacity_prompt: int = 128):
#         super().__init__(path=path, max_seq_len=max_seq_len)
#         self.logger = get_logger()

#         self.model_name_or_path = path
#         self.method = method
#         self.attn_implementation = attn_implementation
#         self.max_capacity_prompt = max_capacity_prompt
#         self.generation_kwargs = generation_kwargs

#         if self.method == 'pyramidkv':
#             if "Llama" in self.model_name_or_path:
#                 replace_llama(self.method.lower())
#             elif "Mistral" in self.model_name_or_path:
#                 replace_mistral(self.method.lower())
#             else:
#                 raise ValueError("Unsupported model type for pyramidkv.")

#         self._load_tokenizer(path, tokenizer_kwargs)
#         self._load_model(path, model_kwargs)
#         self._set_model_configuration()

#     def _load_tokenizer(self, path: str, tokenizer_kwargs: dict):
#         self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

#     def _load_model(self, path: str, model_kwargs: dict):
#         self.model = AutoModelForCausalLM.from_pretrained(
#             path,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             **model_kwargs
#         ).eval()

#     def _set_model_configuration(self):
#         layers = len(self.model.model.layers)
#         window_sizes = 8
#         kernel_sizes = 7
#         # window_sizes = 32
#         # kernel_sizes = 30

#         pooling = "maxpool"

#         if not isinstance(window_sizes, list):
#             window_sizes = [window_sizes] * layers
#         if not isinstance(self.max_capacity_prompt, list):
#             self.max_capacity_prompt = [self.max_capacity_prompt] * layers
#         if not isinstance(kernel_sizes, list):
#             kernel_sizes = [kernel_sizes] * layers

#         for i in range(layers):
#             self.model.model.layers[i].self_attn.config.window_size = window_sizes[i]
#             self.model.model.layers[i].self_attn.config.max_capacity_prompt = self.max_capacity_prompt[i]
#             self.model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
#             self.model.model.layers[i].self_attn.config.pooling = pooling

#     @torch.no_grad()
#     def generate(self, inputs: list, max_out_len: int) -> list:
#         self.model.eval()
#         outputs_text = []

#         batch_size = len(inputs)
#         prompts = [f"{text}" for text in inputs]
#         input_tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
#         input_ids = input_tokens.input_ids.to(self.model.device)
#         attention_mask = input_tokens.attention_mask.to(self.model.device)

#         # streaming prefill，不进行裁剪
#         outputs = self.model(input_ids=input_ids, use_cache=True)
#         past_key_values = outputs.past_key_values
#         next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
#         generated_ids = next_tokens.clone()

#         # streaming decode
#         for _ in range(max_out_len - 1):
#             outputs = self.model(input_ids=next_tokens, past_key_values=past_key_values, use_cache=True)
#             past_key_values = outputs.past_key_values
#             next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
#             generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

#             if (next_tokens == self.tokenizer.eos_token_id).all():
#                 break

#         for i in range(batch_size):
#             text = self.tokenizer.decode(
#                 generated_ids[i],
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=True,
#                 spaces_between_special_tokens=False
#             ).strip()
#             outputs_text.append(text)

#         return outputs_text

import torch
from opencompass.models.base import BaseModel
from opencompass.utils.logging import get_logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyramidkv.monkeypatch import replace_llama, replace_mistral
import numpy as np

class PyramidKVModel(BaseModel):
    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_kwargs: dict = dict(),
                 max_seq_len: int = 2048,
                 generation_kwargs: dict = dict(),
                 method: str = "pyramidkv",  # <- 支持 snapkv
                 attn_implementation: str = "flash_attention_2",
                 max_capacity_prompt: int = 128):
        super().__init__(path=path, max_seq_len=max_seq_len)
        self.logger = get_logger()

        self.model_name_or_path = path
        self.method = method
        self.attn_implementation = attn_implementation
        self.max_capacity_prompt = max_capacity_prompt
        self.generation_kwargs = generation_kwargs

        if self.method in ['pyramidkv', 'snapkv']:
            if "Llama" in self.model_name_or_path:
                replace_llama(self.method.lower())
            elif "Mistral" in self.model_name_or_path:
                replace_mistral(self.method.lower())
            else:
                raise ValueError("Unsupported model type for " + self.method)

        self._load_tokenizer(path, tokenizer_kwargs)
        self._load_model(path, model_kwargs)
        self._set_model_configuration()

    def _load_tokenizer(self, path: str, tokenizer_kwargs: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def _load_model(self, path: str, model_kwargs: dict):
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.float16,
            **model_kwargs
        ).eval()

    def _set_model_configuration(self):
        layers = len(self.model.model.layers)

        if self.method == "snapkv":
            window_sizes = 32
        else:
            window_sizes = 8

        kernel_sizes = 7
        pooling = "maxpool"

        if not isinstance(window_sizes, list):
            window_sizes = [window_sizes] * layers
        if not isinstance(self.max_capacity_prompt, list):
            self.max_capacity_prompt = [self.max_capacity_prompt] * layers
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes] * layers

        for i in range(layers):
            self.model.model.layers[i].self_attn.config.window_size = window_sizes[i]
            self.model.model.layers[i].self_attn.config.max_capacity_prompt = self.max_capacity_prompt[i]
            self.model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
            self.model.model.layers[i].self_attn.config.pooling = pooling

    @torch.no_grad()
    def generate(self, inputs: list, max_out_len: int) -> list:
        self.model.eval()
        outputs_text = []

        batch_size = len(inputs)
        prompts = [f"{text}" for text in inputs]
        input_tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = input_tokens.input_ids.to(self.model.device)
        attention_mask = input_tokens.attention_mask.to(self.model.device)

        outputs = self.model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = next_tokens.clone()

        for _ in range(max_out_len - 1):
            outputs = self.model(input_ids=next_tokens, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

            if (next_tokens == self.tokenizer.eos_token_id).all():
                break

        for i in range(batch_size):
            text = self.tokenizer.decode(
                generated_ids[i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False
            ).strip()
            outputs_text.append(text)

        return outputs_text

