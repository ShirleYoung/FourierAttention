import torch
from opencompass.models.base import BaseModel
from opencompass.utils.logging import get_logger
from transformers import AutoTokenizer
from streaming_llm.utils import load
from streaming_llm.enable_streaming_llm import enable_streaming_llm


class StreamingModel(BaseModel):
    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_kwargs: dict = dict(),
                 max_seq_len: int = 2048,
                 generation_kwargs: dict = dict(),
                 enable_streaming: bool = True,
                 start_size: int = 4,
                 recent_size: int = 1024):
        # 初始化父类
        BaseModel.__init__(self, path=path, max_seq_len=max_seq_len)
        self.logger = get_logger()

        self.model_name_or_path = path
        self.enable_streaming = enable_streaming
        self.start_size = start_size
        self.recent_size = recent_size
        self.generation_kwargs = generation_kwargs

        self._load_tokenizer(path, tokenizer_kwargs)
        self._load_model(path, model_kwargs)

        if self.enable_streaming:
            self.kv_cache = enable_streaming_llm(
                self.model,
                start_size=self.start_size,
                recent_size=self.recent_size
            )
        else:
            self.kv_cache = None

    def _load_tokenizer(self, path: str, tokenizer_kwargs: dict):
        """加载指定路径的tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def _load_model(self, path: str, model_kwargs: dict):
        """加载模型"""
        self.model, self.tokenizer = load(path)

    def _greedy_generate(self, input_ids, past_key_values, max_gen_len, generation_kwargs):
        """Greedy generation logic"""
        return greedy_generate(self.model, self.tokenizer, input_ids, past_key_values, max_gen_len, generation_kwargs)

    @torch.no_grad()
    def generate(self, inputs: list, max_out_len: int) -> list:
        """根据输入生成文本"""
        self.model.eval()
        outputs_text = []

        for text in inputs:
            # 添加 USER 和 ASSISTANT 前缀
            prompt = f"USER: {text}\n\nASSISTANT: "
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(self.model.device)

            # 如果启用了 streaming 模式，则调用流式推理
            if self.enable_streaming:
                past_key_values = None
                # 直接使用带有角色前缀的 prompt
                generated_text =streaming_inference(self.model, self.tokenizer, [text], self.kv_cache, max_gen_len=max_out_len, generation_kwargs=self.generation_kwargs)
                # generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            else:
                # 否则使用 greedy 生成
                generated_text = self._greedy_generate(input_ids, None, max_out_len, self.generation_kwargs)

            outputs_text.append(generated_text)
            self.logger.info(f"Generated text: {generated_text}")

        return outputs_text


# 原本的
# @torch.no_grad()
# def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, generation_kwargs):
#     # 获取初始输出
#     outputs = model(
#         input_ids=input_ids,
#         past_key_values=past_key_values,
#         use_cache=True,
#     )
    
#     past_key_values = outputs.past_key_values
#     pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)  # 获取最后一个 token 的预测
#     generated_ids = [pred_token_idx.item()]  # 存储生成的 token
#     pos = 0

#     # 输出调试信息
#     # print(f"Initial input: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
#     # print(f"First token generated: {tokenizer.decode([pred_token_idx.item()])}")

#     # 生成过程，最大生成长度
#     for _ in range(max_gen_len - 1):
#         # 每次从生成的 token 中继续推理
#         outputs = model(
#             input_ids=pred_token_idx,
#             past_key_values=past_key_values,
#             use_cache=True,
#         )
#         past_key_values = outputs.past_key_values
#         pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
#         generated_ids.append(pred_token_idx.item())

#         # # 调试生成的 token
#         # print(f"Generated token: {tokenizer.decode([pred_token_idx.item()])}")

#         # 检查生成的文本
#         generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
#         if len(generated_text.split()) > pos:
#             pos = len(generated_text.split())

#         # 如果遇到 eos_token，则停止生成
#         if pred_token_idx == tokenizer.eos_token_id:
#             break

#     return past_key_values, generated_text

# prefill阶段也压缩的
@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, generation_kwargs):
    # === PREFILL阶段: 对input_ids进行截断以节省显存 ===
    if past_key_values is None and generation_kwargs is not None:
        num_init_tokens = generation_kwargs.get('numinittokens', 4)
        max_local_len = generation_kwargs.get('maxlocallen', 1024)

        total_len = input_ids.shape[1]

        # 保留前 num_init_tokens 和后 max_local_len 个 tokens
        keep_front = input_ids[:, :num_init_tokens]
        keep_tail = input_ids[:, -max_local_len:] if total_len > max_local_len else input_ids

        # 如果截断后比原始短，才执行拼接
        if keep_front.shape[1] + keep_tail.shape[1] < total_len:
            input_ids = torch.cat([keep_front, keep_tail], dim=1)

    # === PREFILL 阶段: 第一次生成 ===
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0

    # === DECODE 阶段: 后续逐 token 生成 ===
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        # 如果生成了 <eos>，提前终止
        if pred_token_idx.item() == tokenizer.eos_token_id:
            break

    # 解码最终输出
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
        spaces_between_special_tokens=False,
    ).strip()

    return past_key_values, generated_text


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000, generation_kwargs=None):
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        # 添加 USER 和 ASSISTANT 前缀
        # print("\nUSER: " + prompt + "\nASSISTANT:", end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]

        # 确保缓存正确传递
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        # 这里调用 greedy_generate
        past_key_values, generated_text  = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, generation_kwargs=generation_kwargs
        )

        return generated_text


