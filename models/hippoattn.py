import os
import torch
from typing import List, Optional, Dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from opencompass.models.base import BaseModel
from opencompass.utils.logging import get_logger
import json
from pathlib import Path


# 用于定义模型类型
MODEL_TYPES = ['llama', 'qwen', 'mistral', 'internlm2', 'other']  # 根据需求添加更多模型

class HippoattnCausalLM(BaseModel):
    def __init__(self,
                 path: str,
                 model_type: str,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 max_seq_len: int = 2048,
                 mode: str = 'none',
                 end_str: Optional[str] = None,
                 use_custom_attention: bool = False):
        # 初始化父类
        BaseModel.__init__(self, path=path, max_seq_len=max_seq_len)
        self.logger = get_logger()
        self.model_type = model_type
        self.mode = mode
        self.end_str = end_str
        self.use_custom_attention = use_custom_attention

        self._load_tokenizer(path=path, tokenizer_path=tokenizer_path, tokenizer_kwargs=tokenizer_kwargs)
        self._load_model(path=path, model_type=model_type, model_kwargs=model_kwargs)
        self.generation_kwargs = generation_kwargs

    def _load_tokenizer(self, path: Optional[str], tokenizer_path: Optional[str], tokenizer_kwargs: dict):
        """加载指定路径的tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or path, **tokenizer_kwargs)
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def _load_model(self, path: str, model_type: str, model_kwargs: dict):
        """根据模型类型加载不同的预训练模型"""
        self.config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        
        # 针对不同的模型类型加载不同的预训练模型
        if model_type == 'llama':
            self.model = LlamaForCausalLM.from_pretrained(
                path,
                config=self.config,
                device_map="auto"  # 直接传入 device_map 进行自动设备映射
            )
        elif model_type == 'qwen':
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                # **model_kwargs,
                config=self.config,
                device_map="auto"
            )
        elif model_type == 'internlm2':
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                **model_kwargs,
                config=self.config,
                device_map="auto"
            )
        elif model_type == 'mistral':
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                **model_kwargs,
                config=self.config,
                device_map="auto"
            )
        else:  # 默认情况
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                **model_kwargs,
                config=self.config,
                device_map="auto"
            )
        
        self.model.eval()  # 设置模型为评估模式

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """根据输入生成文本"""
        self.model.eval()  # 设置模型为评估模式
        outputs_text = []
        
        for text in inputs:
            
            print("="*50)
            print(f"text:{text}")
            print("="*50)
            input_ids = self.tokenizer(text, return_tensors="pt", truncation=True).input_ids.to(self.model.device)

            outputs = self.model.generate(input_ids, max_new_tokens=max_out_len, **self.generation_kwargs)
            #outputs = self.model.generate(input_ids, max_new_tokens=max_out_len)
            generated_text = self.tokenizer.decode(outputs[0,input_ids.shape[1]:], skip_special_tokens=True)
            outputs_text.append(generated_text)
            print('*'*50)
            print(f"Generated text:{generated_text}")
            print('*'*50)

        return outputs_text