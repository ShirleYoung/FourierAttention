import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import gc
import functools
from collections import defaultdict
from typing import List

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
# from tinychat.models import LlavaLlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from datasets import load_dataset

import json 

from hippo_function_approx_legt_fourier_awq import MultiDimHiPPO

device = "cuda:0"

def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]

def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")

def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x

__all__ = ["run_awq"]

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_blocks(model):
    if model.__class__.__name__ in ("LlamaForCausalLM", "Qwen2ForCausalLM"):
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    elif model.__class__.__name__ == "LlavaLlamaModel":
        layers = model.llm.model.layers
    else:
        raise NotImplementedError(type(model))
    return layers

def move_embed(model, device):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif isinstance(model, LlavaLlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif "llavallamamodel" in str(model.__class__).lower():
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))


def compute_cosine_similarity(qkv_result_tocompress, qkv_result_mid_decompressed):
    qkv_result_mid_decompressed = qkv_result_mid_decompressed.unsqueeze(0)
    bs, seq_len, num_dims = qkv_result_tocompress.shape
    mean_decompressed = torch.mean(qkv_result_mid_decompressed[0, :, :], dim=0, keepdim=True)
    mean_tocompress = torch.mean(qkv_result_tocompress[0, :, :], dim=0, keepdim=True)
    scale = torch.max(torch.abs(qkv_result_mid_decompressed[0, :, :] - mean_decompressed), dim=0).values / torch.max(torch.abs(qkv_result_tocompress[0, :, :] - mean_tocompress), dim=0).values
    
    data = qkv_result_mid_decompressed[0, :, :]  # Shape: [seq_len, input_dim]
    means = data.mean(dim=0)  # Calculate mean value of each input_dim, result shape is [input_dim]
    data_centered = data - means  # Center the data
    data_scaled = data_centered / scale  # Scale proportionally
    data_final = data_scaled + torch.mean(qkv_result_tocompress[0, :, :], dim=0)  # Add back mean
    qkv_result_mid_decompressed[0, :, :] = data_final

    all_sample_similarities = torch.zeros(bs, num_dims)

    for j in range(bs):  # Process each sample
        for i in range(num_dims):  # Calculate similarity for each feature dimension
            compressed_vectors = qkv_result_tocompress[j, :, i]  # shape: [seq_len]
            decompressed_vectors = qkv_result_mid_decompressed[j, :, i]  # shape: [seq_len]

            # similarity = F.cosine_similarity(compressed_vectors.unsqueeze(0), decompressed_vectors.unsqueeze(0), dim=-1)  # Shape [1]
            similarity = F.mse_loss(compressed_vectors.unsqueeze(0), decompressed_vectors.unsqueeze(0))

            all_sample_similarities[j, i] = similarity.item()

    return all_sample_similarities

@torch.no_grad()
def run_awq(
    model,
    enc,
    w_bit,
    q_config,
    input_texts,  # List of input texts
    n_samples=512,
    seqlen=512,
    auto_scale=True,
    mse_range=True,
):

    all_sample_similarities = {}

    if "bigcode" in str(model.__class__).lower():
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)

    # Encode text list into model input, but do not merge them into a single batch
    samples = [torch.tensor(enc.encode(text)).to(device) for text in input_texts]
    for sample in samples:
        sample = sample.unsqueeze(0)
        print(f"samples长度{len(samples)},samples内部大小{samples[0].size()}")

        inps = []
        layer_kwargs = {}

        layers[0] = layers[0].cuda()
        move_embed(model, "cuda")

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                print("we are in forward")
                inps.append(inp)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        layers[0] = Catcher(layers[0])
        try:
            if model.__class__.__name__ == "LlavaLlamaModel":
                model.llm(sample.to(next(model.parameters()).device))  # Pass sample one by one
            else:
                model(sample.to(next(model.parameters()).device))  # Pass sample one by one
        except Exception as e:
            pass

        # del samples
        layers[0] = layers[0].module
        print(f"未进行inps = inps[0]的inps长度是：{len(inps)}")
        # inps = inps[0]

        layers[0] = layers[0].cpu()
        move_embed(model, "cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # print(f"inputs size: {inps[0].size()}")

        # **Process sample by sample, not batch processing**
        # for idx, sample in enumerate(inps):
            # print(f"Processing sample {idx + 1}/{len(inps)}...")
        sample = torch.tensor(inps[0])
        idx = 0
        # Solve layer by layer
        for i in tqdm.tqdm(range(len(layers)), desc=f"Layer-by-layer processing sample {idx + 1}..."):
            layer = layers[i]
            layer = layer.cuda()
            named_linears = get_named_linears(layer)

            def cache_individual_qkv_hooks(m, x, y, name, qkv_dict, similarity_dict,numinittokens, maxlocallen, maxmidstates):
                print(f"name is {name}")
                x_input = x[0]  # Get input X
                print(f"x's shape is {x_input.size()}")
                seq_len_ori = x_input.size(1)

                w_weights = m.weight  # Get current module weights W_q/W_k/W_v
                print(f"Weights' shape is {w_weights.size()}")
                qkv_result = torch.matmul(x_input, w_weights.T)  # Independently compute Q/K/V
                # qkv_result = qkv_result.detach().cpu()
                qkv_dict[name] = qkv_result

                hippo = MultiDimHiPPO(
                    N=qkv_result.size(2) * maxmidstates,
                    input_dim=qkv_result.size(2),
                    method='legt',
                    dt=1 / (seq_len_ori - numinittokens - maxlocallen),
                    T=1
                ).to('cuda')

                qkv_result_init = qkv_result[:, :numinittokens, :]
                qkv_result_local = qkv_result[:, -maxlocallen:, :]  # Local window key matrix
                qkv_result_tocompress = qkv_result[:, numinittokens:-maxlocallen, :]

                qkv_result_mid_compressed = hippo(qkv_result_tocompress.to("cuda"),token_num = 0)
                qkv_result_mid_decompressed = hippo.reconstruct(qkv_result_mid_compressed, seq_len_ori - numinittokens - maxlocallen)
                qkv_result_mid_decompressed = qkv_result_mid_decompressed.squeeze(0)
            
                
                similarities = compute_cosine_similarity(qkv_result_tocompress, qkv_result_mid_decompressed)
                similarity_dict[name].append(similarities)

            qkv_results = defaultdict(list)
            similarity_results= defaultdict(list)
            handles = []
            for name, module in named_linears.items():
                print("******")
                print(name)
                print("******")
                # Register independent hooks for k_proj and v_proj respectively
                if "k_proj" in name or "v_proj" in name:
                    print("we are in if")
                    handles.append(
                        module.register_forward_hook(
                            functools.partial(cache_individual_qkv_hooks, name=name, qkv_dict=qkv_results,similarity_dict=similarity_results, numinittokens=4, maxlocallen=1024, maxmidstates=1024)
                        )
                    )
            
            # sample=sample.unsqueeze(0)
            sample = sample.to(next(layer.parameters()).device)
            sample = sample.to("cuda")
            print(f"sample的维度：{sample.size()}")
            sample = layer(sample, **layer_kwargs)[0]  

            for h in handles:
                h.remove()

            # qkv_results = {k: torch.cat(v, dim=0) for k, v in qkv_results.items()}

            print(f"Sample {idx + 1} Layer {i} QKV results computed individually:")
            for name, tensor in similarity_results.items():
                print(f"{name}: {len(tensor)}{len(tensor[0])}")

                # Store results to dictionary
                layer_name = f"Layer_{i}"
                print(f"当前在第{i}层")
                if layer_name not in all_sample_similarities:
                    all_sample_similarities[layer_name] = {}
                    all_sample_similarities[layer_name][name] = tensor
                elif name not in all_sample_similarities[layer_name]:
                    all_sample_similarities[layer_name][name] =  tensor
                else:
                    print("we are appending")
                    all_sample_similarities[layer_name][name].append(tensor)


            torch.cuda.empty_cache()

            layer = layer.cpu()
            gc.collect()
            torch.cuda.empty_cache()

    # Define a dictionary to save feature dimensions corresponding to 80% smallest mean values
    non_critical_dims_dict = {}

    # Traverse each layername and name corresponding tensor in all_sample_similarities
    for layername, layer_data in all_sample_similarities.items():
        
        # Set proportion of non-critical dimensions (different for each layer)
        percent = 0
        layer_number = int(layername.split('_')[1])  # Extract layer number

        
        print(f"layername is :{layername}")
        print(f"layer_data is :{layer_data}")
        non_critical_dims_dict[layername] = {}
        # print(layer_data)
        for name, tensor in layer_data.items():
            # Print tensor length and first element
            # print(len(tensor))
            # print(tensor[0])
            # print("name")
            # Set compression ratio
            if 20 <= layer_number <= 27:
                if name == "self_attn.k_proj":
                    percent = 0.5
                else:
                    percent = 0.7
            elif 0 <= layer_number <= 3:
                if name == "self_attn.k_proj":
                    percent = 0.9
                else:
                    percent = 0.95
            else:
                percent = 0.8
            
            
            # Initialize a tensor with the same shape as tensor[0][0] to store mean values
            mean_similarities = torch.zeros(tensor[0][0].shape)

            # Traverse all elements in tensor and calculate mean value of each feature dimension
            for t in tensor:
                mean_similarities += t[0].squeeze(0)  # Accumulate first dimension ([0]) of each tensor

            # Calculate mean value by dividing by the length of tensor
            mean_similarities /= len(tensor)

            # Sort mean values and select 80% smallest feature dimensions
            mean_abs_values = mean_similarities.abs()  # Get absolute value of mean
            sorted_indices = torch.argsort(mean_abs_values)  # Sort indices

            ### Save to file
            # Convert to JSON serializable format
            mean_similarities_list = mean_similarities.tolist()  # Convert tensor to list
            sorted_indices_list = sorted_indices.tolist()  # Convert tensor to list

            # Create a dictionary to store this data
            data = {
                "mean_similarities": mean_similarities_list,
                "sorted_indices": sorted_indices_list
            }

            # Save data to JSON file
            with open('/FourierAttention/compressed_dims_file/data_ruler4k.json ', 'a') as json_file:  # your path to save the mean_similaritites and sorted_indices so that you can use compressed_dims_search_readfile.py to easily change the scale
                json.dump({layername: {name: data}}, json_file)
                json_file.write("\n")  


            # Select 80% feature dimensions
            num_non_critical = int(percent * mean_similarities.shape[0])  
            non_critical_indices = sorted_indices[:num_non_critical]  

            # Save to dictionary
            non_critical_dims_dict[layername][name] = non_critical_indices.tolist()  # Convert to list and save


    # Save non-critical dimensions result (can be modified)
    with open("/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_4k_fourier.json", "w", encoding="utf-8") as f:
        json.dump(non_critical_dims_dict, f, ensure_ascii=False, indent=4)

    return


from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Model path, can be changed
    model_name = "meta-llama/Llama-3.2-3B"
    print("loading models and tokenizer...")
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="cuda:0")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    file_paths = [
        # your path to your context
    ]

    # Used to save all extracted origin_prompts
    origin_prompts = []

    # Traverse each file and extract only 2 origin_prompts
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        prompts = [entry['origin_prompt'] for entry in data.values()][:2]
        origin_prompts.extend(prompts)

    # Print total number of extracted prompts
    print(f"Total origin prompts count: {len(origin_prompts)}")  # Should be 8 * 2 = 16

    # Set running parameters
    w_bit = 8  # Weight quantization bits, e.g. 8 or 4
    q_config = {}  # Additional quantization configuration

    # Extract input features
    print("Starting to extract input features...")
    input_feat = run_awq(
        model=model,
        enc=tokenizer,
        w_bit=w_bit,
        q_config=q_config,
        input_texts=origin_prompts,
        n_samples=10,  
        seqlen=512,  
        auto_scale=False,
        mse_range=False,
    )

if __name__ == "__main__":
    main()
