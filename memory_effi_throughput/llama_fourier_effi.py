import os
import time
import torch
from transformers import AutoConfig, AutoTokenizer
from dataclasses import dataclass

# --- IMPORTANT ---
# The following imports are from your custom files. Ensure these files
# are in the same directory as this script or in your Python path.
from modeling_llama_fourier import LlamaForCausalLM
from cache_utils_fourier import DynamicCache

@dataclass
class ExtraConfig:
    """
    Configuration class for custom cache parameters.
    """
    max_new_tokens: int
    numinittokens: int
    maxlocallen: int
    maxmidstates: int
    non_critical_dims_path: str
    max_position_embeddings: int
    num_key_value_heads: int
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    dtype: torch.dtype

def model_gen(model, config, extra_config, input_ids):
    """
    Performs a single forward pass to simulate the prefill stage.
    Initializes a new cache for the generation sequence.
    """
    # This configuration is specific to your custom HippoAttention model

    with torch.no_grad():
        past_key_values = DynamicCache(extra_config)
        # 预填充阶段：处理 prompt
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        # 取最后一个 token 的 argmax 作为下一个输入
        input_q = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values

    return outputs, input_q, past_key_values

def model_decode(model, config, extra_config, input_ids, past_key_values, max_new_tokens=10):
    """
    使用 argmax 采样进行 decode，生成 max_new_tokens 个 token，并计时。
    返回最终输出和 decode 总耗时（秒）。
    """
    model.eval()
    generated = input_ids
    total_decode_time = 0

    with torch.no_grad():
        for step in range(max_new_tokens):
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(input_ids=generated, past_key_values=past_key_values, use_cache=True)
            torch.cuda.synchronize()
            end_time = time.time()
            total_decode_time += (end_time - start_time)

            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return generated, total_decode_time

def main():
    # --- Configuration ---
    # MODEL_PATH = "meta-llama/Llama-3.2-3B/" 
    MODEL_PATH = "meta-llama/Llama-3.1-8B/" 
    # NON_CRITICAL_DIMS_PATH = " "# your path to FourierAttention/memory_effi_throughput/jsonl/3b_non_compress_info.json
    NON_CRITICAL_DIMS_PATH = " " # your path to FourierAttention/memory_effi_throughput/jsonl/8b_non_compress_info.json
    REPEATS = 10 # Increase repeats for more stable averaging
    CONTEXT_LENGTH = 2560  # [160, 320, 640, 1280, 1920, 2560]  # 4k, 8k, 16k, 32k, 48k, 64k
    BATCH_SIZE = 1
    CONTEXT_LENGTH //= 2

    # --- Environment Setup ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(NON_CRITICAL_DIMS_PATH):
        raise FileNotFoundError("Model or non-critical dims path not found.")

    # --- Model and Tokenizer Loading ---
    print("Loading model and tokenizer...")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH, config=config, device_map=device,
        torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    extra_config = ExtraConfig(
        max_new_tokens=1, # Only for one token generation
        numinittokens=4,
        maxlocallen=1020,
        maxmidstates=1024,
        non_critical_dims_path=NON_CRITICAL_DIMS_PATH,
        max_position_embeddings=config.max_position_embeddings,
        num_key_value_heads=config.num_key_value_heads,
        num_hidden_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        dtype=torch.float16,
        num_attention_heads=config.num_attention_heads,
    )
    print("Model loaded.")

    # --- Prompt Creation ---
    begin = "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards."
    end = "What is the special magic number for wandering-age mentioned in the provided text? The special magic number for wandering-age mentioned in the provided text is"
    needle = "One of the special magic numbers for wandering-age is: 8090293."
    hazy = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    text = '\n'.join([begin] + [hazy] * CONTEXT_LENGTH + [needle] + [hazy] * CONTEXT_LENGTH + [end])
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids: torch.Tensor = inputs["input_ids"].to(device)
    input_ids = input_ids.repeat(BATCH_SIZE, 1)
    gen_token_num = BATCH_SIZE * 10
    print(f"Prompt length: {input_ids.shape[1]} tokens")

    # --- Prefill Time Measurement ---
    print("\n--- Measuring Prefill Performance ---")

    # Warm-up run to handle any initial CUDA setup overhead
    print("Warming up...")
    model_gen(model, config, extra_config, input_ids)
    torch.cuda.synchronize()

    total_prefill_time = 0
    total_decode_time = 0
    print(f"Running benchmark ({REPEATS} repeats)...")
    for i in range(REPEATS):
        # Synchronize before starting the timer to ensure all previous CUDA operations are complete
        torch.cuda.synchronize()
        start_time = time.time()

        _, input_q, past_key_values = model_gen(model, config, extra_config, input_ids)

        # Synchronize after the operation to ensure it has completed before stopping the timer
        torch.cuda.synchronize()
        end_time = time.time()

        # add decode time
        # 统计 decode 时间
        _, decode_time = model_decode(model, config, extra_config, input_q, past_key_values, max_new_tokens=10)

        total_prefill_time += (end_time - start_time)
        total_decode_time += decode_time
        print(f"Repeat {i+1}/{REPEATS} Prefill: {(end_time - start_time) * 1000:.2f} ms")
        print(f"Decode time (10 tokens, argmax): {decode_time * 1000:.2f} ms")
        print(f"Throughput {gen_token_num/(end_time - start_time + decode_time)} token per second")


    average_prefill_time_ms = (total_prefill_time / REPEATS) * 1000
    average_decode_time_ms = (total_decode_time / REPEATS) * 1000

    print("-" * 50)
    print(f"Average Prefill Time over {REPEATS} runs: {average_prefill_time_ms:.2f} ms")
    print(f"Average Encode+Decode Time over {REPEATS} runs: {average_decode_time_ms:.2f} ms")
    print("-" * 50)


if __name__ == "__main__":
    main()