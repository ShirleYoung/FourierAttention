import os
import time
import torch
from transformers import AutoTokenizer

from utils import load_model_and_tokenizer  # use utils from Palu 

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


def model_gen(model, input_ids, max_new_tokens):
    generated_sequence = input_ids
    init_len = input_ids.size(1)
    past_key_values = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_sequence = torch.cat([generated_sequence, input_ids], dim=-1)
            past_key_values = outputs.past_key_values

            # Corrected check for batch inputs
            if torch.any(input_ids == eos_token_id):
                break

    return generated_sequence[:, init_len:]


def throughput_test(model, input_ids, max_new_tokens, eos_token_id, batch_size, repeats=10):
    start_time = time.time()
    for _ in range(repeats):
        model_gen(model, input_ids, max_new_tokens)
        torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = end_time - start_time
    total_generated_tokens = max_new_tokens * repeats * batch_size
    throughput = total_generated_tokens / elapsed_time  # tokens per second
    print(f"Throughput: {throughput:.2f} tokens per second")


if __name__ == "__main__":
    device = torch.device("cuda:0")

    model_path = "" #path to the model compressed by Palu
    model, tokenizer = load_model_and_tokenizer(model_path, use_flash_attn2=False)
    model = model.to(device)

    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    begin = "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards."
    end = "What is the special magic number for wandering-age mentioned in the provided text? The special magic number for wandering-age mentioned in the provided text is"
    needle = "One of the special magic numbers for wandering-age is: 8090293."
    hazy = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

    text_repeat_num = 640  # You can modify this as needed. When you set it 160, it means 4K.
    text = '\n'.join([hazy] * (text_repeat_num // 2))
    text = '\n'.join([begin, text, needle, text, end])

    max_new_tokens = 10
    batch_size = 4 # You can tune this till OOM

    # Tokenize input and duplicate for batch
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(device)
    input_ids = input_ids.repeat(batch_size, 1)

    # Generate a sample output
    generated_sequence = model_gen(model, input_ids, max_new_tokens)
    generated_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
    print(f"N_CTX: {input_ids.size(1)}, Custom Model, Generated text: {generated_text}")

    # Run throughput benchmark
    throughput_test(model, input_ids, max_new_tokens, eos_token_id, batch_size)

    # Run prefill benchmark
    repeats = 10
    prefill_time = 0
    for i in range(repeats):
        start_time = time.time()
        model_gen(model, input_ids, max_new_tokens=1)
        torch.cuda.synchronize(device)
        end_time = time.time()
        prefill_time += (end_time - start_time)
    prefill_time /= repeats

    print(f"Prefill: {prefill_time:.3f} s")
    print_gpu_memory_info()
    print('\n')

