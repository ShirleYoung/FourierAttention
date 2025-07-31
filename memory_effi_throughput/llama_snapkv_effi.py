import os
import time
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

# ✅ 添加对 snapkv 的支持
from pyramidkv.monkeypatch import replace_llama

def print_gpu_memory_info():
    current_device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(current_device).total_memory
    allocated_memory = torch.cuda.memory_allocated(current_device)
    cached_memory = torch.cuda.max_memory_allocated(current_device)
    total_memory_gb = total_memory / (1024 ** 3)
    allocated_memory_gb = allocated_memory / (1024 ** 3)
    cached_memory_gb = cached_memory / (1024 ** 3)
    usage_percentage = (allocated_memory / total_memory) * 100

    print(f"Total Mem: {total_memory_gb:.2f} GB, Allocated Mem: {allocated_memory_gb:.2f} GB ({usage_percentage:.2f}% used), Reserved Mem: {cached_memory_gb:.2f} GB")


def model_gen(model, input_ids, max_new_tokens, eos_token_id):
    generated_sequence = input_ids
    past_key_values = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_sequence = torch.cat([generated_sequence, input_ids], dim=-1)
            past_key_values = outputs.past_key_values

            # Check for eos_token_id in the batch
            if torch.any(input_ids == eos_token_id):
                break

    return generated_sequence[:, -max_new_tokens:]


def throughput_test(model, input_ids, max_new_tokens, eos_token_id, batch_size, repeats=10):
    start_time = time.time()
    for _ in range(repeats):
        generated_sequence = model_gen(model, input_ids, max_new_tokens, eos_token_id)
        torch.cuda.synchronize()  # Ensure all CUDA operations finish before measuring time
    end_time = time.time()

    # Calculate throughput: total tokens generated / elapsed time
    elapsed_time = end_time - start_time
    total_generated_tokens = max_new_tokens * repeats * batch_size
    throughput = total_generated_tokens / elapsed_time  # tokens per second
    print(f"Throughput: {throughput:.2f} tokens per second")


if __name__ == "__main__":
    # ✅ Replace attention mechanism with SnapKV
    replace_llama("snapkv")

    device = torch.device("cuda:0")
    model_name = "meta-llama/Llama-3.2-3B"

    config = AutoConfig.from_pretrained(model_name)
    config._attn_implementation = "eager"  # ✅ Enable FlashAttention 2

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    # ✅ Set SnapKV-specific parameters
    layers = len(model.model.layers)
    for i in range(layers):
        model.model.layers[i].self_attn.config.window_size = 32
        model.model.layers[i].self_attn.config.kernel_size = 7
        model.model.layers[i].self_attn.config.pooling = "maxpool"
        model.model.layers[i].self_attn.config.max_capacity_prompt = 3076

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    begin = "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards."
    end = "What is the special magic number for wandering-age mentioned in the provided text? The special magic number for wandering-age mentioned in the provided text is"
    needle = "One of the special magic numbers for wandering-age is: 8090293."
    hazy = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

    text_repeat_num = 320 # [160, 320, 640, 1280, 1920, 2560]  # 4k, 8k, 16k, 32k, 
    text = '\n'.join([hazy] * (text_repeat_num // 2))
    text = '\n'.join([begin, text, needle, text, end])

    max_new_tokens = 10  # You can modify this to control how many tokens are generated in one call

    # Prepare inputs and move to GPU
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    # Generate input_ids with batch size support
    batch_size = 1  # Modify this to change the batch size
    input_ids = inputs["input_ids"].repeat(batch_size, 1).to(device)  # Repeat for batch size

    # # Run throughput test
    # throughput_test(model, input_ids, max_new_tokens, eos_token_id, batch_size)

    # Benchmarking prefill time
    repeats = 10
    prefill_time = 0
    for i in range(repeats):
        start_time = time.time()
        model_gen(model, input_ids, max_new_tokens=1, eos_token_id=eos_token_id)
        torch.cuda.synchronize()
        end_time = time.time()
        prefill_time += (end_time - start_time)
    prefill_time /= repeats

    print(f"Prefill: {prefill_time:.3f} ms")
    print_gpu_memory_info()
    print('\n')
