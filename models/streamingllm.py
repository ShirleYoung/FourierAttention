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
        # Initialize parent class
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
        """Load tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def _load_model(self, path: str, model_kwargs: dict):
        """Load models"""
        self.model, self.tokenizer = load(path)

    def _greedy_generate(self, input_ids, past_key_values, max_gen_len, generation_kwargs):
        """Greedy generation logic"""
        return greedy_generate(self.model, self.tokenizer, input_ids, past_key_values, max_gen_len, generation_kwargs)

    @torch.no_grad()
    def generate(self, inputs: list, max_out_len: int) -> list:
        self.model.eval()
        outputs_text = []

        for text in inputs:
            # Add USER and ASSISTANT prefix
            prompt = f"USER: {text}\n\nASSISTANT: "
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(self.model.device)

            # If streaming mode is enabled, call streaming inference
            if self.enable_streaming:
                past_key_values = None
                # Use prompt with role prefix directly
                generated_text =streaming_inference(self.model, self.tokenizer, [text], self.kv_cache, max_gen_len=max_out_len, generation_kwargs=self.generation_kwargs)
                # generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            else:
                # Otherwise use greedy generation
                generated_text = self._greedy_generate(input_ids, None, max_out_len, self.generation_kwargs)

            outputs_text.append(generated_text)
            self.logger.info(f"Generated text: {generated_text}")

        return outputs_text


# Original version
# @torch.no_grad()
# def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, generation_kwargs):
#     # Get initial output
#     outputs = model(
#         input_ids=input_ids,
#         past_key_values=past_key_values,
#         use_cache=True,
#     )
    
#     past_key_values = outputs.past_key_values
#     pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)  # Get prediction for last token
#     generated_ids = [pred_token_idx.item()]  # Store generated token
#     pos = 0

#     # Output debug information
#     # print(f"Initial input: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
#     # print(f"First token generated: {tokenizer.decode([pred_token_idx.item()])}")

#     # Generation process, maximum generation length
#     for _ in range(max_gen_len - 1):
#         # Continue reasoning from generated tokens each time
#         outputs = model(
#             input_ids=pred_token_idx,
#             past_key_values=past_key_values,
#             use_cache=True,
#         )
#         past_key_values = outputs.past_key_values
#         pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
#         generated_ids.append(pred_token_idx.item())

#         # # Debug generated tokens
#         # print(f"Generated token: {tokenizer.decode([pred_token_idx.item()])}")

#         # Check generated text
#         generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
#         if len(generated_text.split()) > pos:
#             pos = len(generated_text.split())

#         # If EOS token is encountered, stop generation
#         if pred_token_idx == tokenizer.eos_token_id:
#             break

#     return past_key_values, generated_text

# Also compress in prefill stage
@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, generation_kwargs):
    # === PREFILL stage: Truncate input_ids to save GPU memory ===
    if past_key_values is None and generation_kwargs is not None:
        num_init_tokens = generation_kwargs.get('numinittokens', 4)
        max_local_len = generation_kwargs.get('maxlocallen', 1024)

        total_len = input_ids.shape[1]

        # Keep first num_init_tokens and last max_local_len tokens
        keep_front = input_ids[:, :num_init_tokens]
        keep_tail = input_ids[:, -max_local_len:] if total_len > max_local_len else input_ids

        # Concatenate only if truncated length is less than original
        if keep_front.shape[1] + keep_tail.shape[1] < total_len:
            input_ids = torch.cat([keep_front, keep_tail], dim=1)

    # === PREFILL stage: First generation ===
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0

    # === DECODE stage: Generate subsequent tokens one by one ===
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        # If EOS token generated, terminate early
        if pred_token_idx.item() == tokenizer.eos_token_id:
            break

    # Decode final output
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
        # Add USER and ASSISTANT prefix
        # print("\nUSER: " + prompt + "\nASSISTANT:", end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]

        # Ensure cache is passed correctly
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        # Call greedy_generate here
        past_key_values, generated_text  = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len, generation_kwargs=generation_kwargs
        )

        return generated_text


