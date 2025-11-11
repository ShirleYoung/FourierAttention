import torch
from opencompass.models.base import BaseModel
from opencompass.utils.logging import get_logger
from transformers import AutoTokenizer, AutoConfig
from mmengine.registry import MODELS


class ExtraConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@MODELS.register_module()  
class FourierModel(BaseModel):
    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_kwargs: dict = dict(),
                 max_seq_len: int = 2048,
                 max_out_len: int = 500,
                 generation_kwargs: dict = dict(),
                 midstates: int = 1024,
                 non_critical_dims_path: str = "/path/to/dimdifferjson_32k/non_critical_dims_hippofourier_kvdiffer6_512mid_splithead.json"):
        super().__init__(path=path, max_seq_len=max_seq_len)
        self.logger = get_logger()
        self.max_out_len = max_out_len
        
        self.model_name_or_path = path
        self.non_critical_dims_path = non_critical_dims_path
        self.midstates = midstates
        self.generation_kwargs = generation_kwargs
        
        # 检测模型类型
        self.model_type = self._detect_model_type(path)
        self.logger.info(f"Detected model type: {self.model_type}")
        
        self._load_model_and_tokenizer(path, model_kwargs, tokenizer_kwargs)

    def _detect_model_type(self, path: str) -> str:
        """检测模型类型（Llama / Qwen2 / Qwen3）"""
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)

        # 1) 直接依据 model_type
        mt = getattr(config, "model_type", None)
        if isinstance(mt, str):
            mtl = mt.lower()
            if "llama" in mtl:
                return "llama"
            # 新版 transformers/Qwen 族常见写法
            if "qwen3" in mtl:
                return "qwen3"
            if "qwen2" in mtl:
                return "qwen2"
            # 旧版可能用到 "qwen"（不含代际），默认按 qwen2 处理
            if mtl == "qwen":
                return "qwen2"

        # 2) 依据 architectures 字段（更稳）
        archs = getattr(config, "architectures", None) or []
        archs_l = [str(a).lower() for a in archs]
        if any("qwen3forcausallm" in a or "qwen3" in a for a in archs_l):
            return "qwen3"
        if any("qwen2forcausallm" in a or "qwen2" in a for a in archs_l):
            return "qwen2"
        if any(a == "qwenforcausallm" or "qwen" in a for a in archs_l):
            # 旧 Qwen（1.x）默认按 qwen2 的分支处理
            return "qwen2"

        # 3) 依据路径关键词（兜底）
        pl = path.lower()
        if "llama" in pl:
            return "llama"
        if "qwen3" in pl:
            return "qwen3"
        if "qwen2" in pl or "qwen-2" in pl or "qwen2.5" in pl:
            # Qwen2.5 在很多仓库里仍使用 model_type=qwen2
            return "qwen2"
        if "qwen" in pl:
            return "qwen2"

        raise ValueError(
            f"Unable to detect model type from path: {path}. "
            f"Supported: Llama, Qwen2, Qwen3."
        )

    def _load_model_and_tokenizer(self, path: str, model_kwargs: dict, tokenizer_kwargs: dict):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # Load model configuration
        self.config = AutoConfig.from_pretrained(path, trust_remote_code=True)

        # 根据模型类型动态导入对应的模型和缓存类
        if self.model_type == 'llama':
            from modeling_llama_fourier import LlamaForCausalLM
            from cache_utils_fourier import DynamicCache
            self.DynamicCacheClass = DynamicCache
            ModelClass = LlamaForCausalLM
            self.logger.info("Loading Llama model with Fourier attention")

        elif self.model_type == 'qwen2':
            # 你需要提供以下文件：lxr_modeling_qwen2_fourier.py / lxr_cache_utils_fourier_qwen2.py
            from modeling_qwen2_fourier import Qwen2ForCausalLM
            from cache_utils_fourier_qwen import DynamicCache
            self.DynamicCacheClass = DynamicCache
            ModelClass = Qwen2ForCausalLM
            self.logger.info("Loading Qwen2 model with Fourier attention")

        elif self.model_type == 'qwen3':
            # 你需要提供以下文件：lxr_modeling_qwen3_fourier.py / lxr_cache_utils_fourier_qwen3.py
            from modeling_qwen3_fourier import Qwen3ForCausalLM
            from cache_utils_fourier_qwen import DynamicCache
            self.DynamicCacheClass = DynamicCache
            ModelClass = Qwen3ForCausalLM
            self.logger.info("Loading Qwen3 model with Fourier attention")

        else:
            raise ValueError(f"Unsupported model_type={self.model_type}")

        # Load model with auto device mapping
        self.model = ModelClass.from_pretrained(
            path,
            config=self.config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **model_kwargs
        ).eval()

        # Initialize DynamicCache for past_key_values
        self.past_key_values = self.DynamicCacheClass(self._get_extra_config())


    def _get_extra_config(self):
        """获取额外配置信息"""
        extra_config = ExtraConfig(
            numinittokens=4,
            maxlocallen=1020,
            maxmidstates=self.midstates,
            non_critical_dims_path=self.non_critical_dims_path,
            max_position_embeddings=self.config.max_position_embeddings,
            num_key_value_heads=self.config.num_key_value_heads,
            max_new_tokens=self.max_out_len,
            model_type=self.model_type,
        )
        return extra_config

    @torch.no_grad()
    def generate(self, inputs: list, max_out_len: int) -> list:
        """生成文本"""
        self.model.eval()
        outputs_text = []

        input_tokens = self.tokenizer(
            inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.max_seq_len
        )
        input_ids = input_tokens.input_ids.to(self.model.device)
        init_len = input_ids.size(1)

        generated_sequence = input_ids
        eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id

        # 生成文本
        for step in range(max_out_len):
            outputs = self.model(
                input_ids=input_ids, 
                past_key_values=self.past_key_values, 
                use_cache=True
            )
            
            # 获取下一个token
            input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_sequence = torch.cat([generated_sequence, input_ids], dim=-1)
            self.past_key_values = outputs.past_key_values

            # 检查是否生成了结束符
            if input_ids.item() == eos_token_id:
                self.logger.info(f"EOS token generated at step {step}")
                break

        # 重置缓存
        self.past_key_values = self.DynamicCacheClass(self._get_extra_config())
        
        # 解码生成的文本
        for i in range(len(inputs)):
            text = self.tokenizer.decode(
                generated_sequence[i, init_len:], 
                skip_special_tokens=True
            )
            outputs_text.append(text)

        return outputs_text
