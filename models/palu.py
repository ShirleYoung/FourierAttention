import sys
sys.path.append("/path/to/Palu")

import os
import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from opencompass.models.base import BaseModel
from opencompass.utils.logging import get_logger
from utils import load_model_and_tokenizer   # use utils from Palu 

class PaluCausalLM(BaseModel):
    def __init__(self,
                 path: str,
                 tokenizer_kwargs: dict = dict(),
                 generation_kwargs: dict = dict(),
                 max_seq_len: int = 409600,
                 max_out_len: int = 200,
                 mode: str = 'none',
                 end_str: Optional[str] = None):
        super().__init__(path=path, max_seq_len=max_seq_len)
        self.logger = get_logger()
        self.end_str = end_str
        self.max_out_len = max_out_len

        self.model, self.tokenizer = load_model_and_tokenizer(path)
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.model.eval()

        self.generation_kwargs =generation_kwargs

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: Optional[int] = None) -> List[str]:
        outputs_text = []
        max_out_len = max_out_len or self.max_out_len

        for prompt in inputs:
            input_ids = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            input_len = input_ids['input_ids'].shape[1]

            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=max_out_len
            )

            generated_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            outputs_text.append(generated_text)

            print("=" * 40)
            print(f"Prompt: {prompt}")
            print(f"Output: {generated_text}")
            print("=" * 40)

        return outputs_text
