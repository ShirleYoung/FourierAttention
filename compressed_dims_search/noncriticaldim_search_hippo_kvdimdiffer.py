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
    
    data = qkv_result_mid_decompressed[0, :, :]  # 形状为 [seq_len, input_dim]
    means = data.mean(dim=0)  # 计算每个 input_dim 的平均值, 结果形状为 [input_dim]
    data_centered = data - means  # 中心化
    data_scaled = data_centered / scale  # 按比例缩放
    data_final = data_scaled + torch.mean(qkv_result_tocompress[0, :, :], dim=0)  # 加回均值
    qkv_result_mid_decompressed[0, :, :] = data_final

    all_sample_similarities = torch.zeros(bs, num_dims)

    for j in range(bs):  # 对每个样本进行操作
        for i in range(num_dims):  # 对每个特征维度计算相似度
            compressed_vectors = qkv_result_tocompress[j, :, i]  # shape: [seq_len]
            decompressed_vectors = qkv_result_mid_decompressed[j, :, i]  # shape: [seq_len]

            # similarity = F.cosine_similarity(compressed_vectors.unsqueeze(0), decompressed_vectors.unsqueeze(0), dim=-1)  # 形状 [1]
            similarity = F.mse_loss(compressed_vectors.unsqueeze(0), decompressed_vectors.unsqueeze(0))

            all_sample_similarities[j, i] = similarity.item()

    return all_sample_similarities

@torch.no_grad()
def run_awq(
    model,
    enc,
    w_bit,
    q_config,
    input_texts,  # 输入文本的列表
    n_samples=512,
    seqlen=512,
    auto_scale=True,
    mse_range=True,
):

    all_sample_similarities = {}

    if "bigcode" in str(model.__class__).lower():
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)

    # 将文本列表编码成模型输入，但不将它们合并成一个批次
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
                model.llm(sample.to(next(model.parameters()).device))  # 逐个样本传入
            else:
                model(sample.to(next(model.parameters()).device))  # 逐个样本传入
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

        # print(f"inputs的size {inps[0].size()}")

        # **逐样本处理**，而非批量处理
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
                x_input = x[0]  # 获取输入 X
                print(f"x's shape is {x_input.size()}")
                seq_len_ori = x_input.size(1)

                w_weights = m.weight  # 获取当前模块的权重 W_q/W_k/W_v
                print(f"Weights' shape is {w_weights.size()}")
                qkv_result = torch.matmul(x_input, w_weights.T)  # 独立计算 Q/K/V
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
                qkv_result_local = qkv_result[:, -maxlocallen:, :]  # 局部窗口对应key矩阵
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
                # 分别为 k_proj, v_proj 注册独立钩子
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

                # 将结果存储到字典
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

    # 定义一个字典保存80%小的平均值对应的特征维度
    non_critical_dims_dict = {}

    # 遍历 all_sample_similarities 中每个 layername 和 name 对应的 tensor
    for layername, layer_data in all_sample_similarities.items():
        
        # 设置非关键维度的比例（每层设置成不一样的）
        percent = 0
        layer_number = int(layername.split('_')[1])  # 提取层数部分

        
        print(f"layername is :{layername}")
        print(f"layer_data is :{layer_data}")
        non_critical_dims_dict[layername] = {}
        # print(layer_data)
        for name, tensor in layer_data.items():
            # 打印 tensor 长度和第一个元素
            # print(len(tensor))
            # print(tensor[0])
            # print("name")
            # 设置压缩比例
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
            
            
            # 初始化一个与 tensor[0][0] 相同形状的张量，用来存储平均值
            mean_similarities = torch.zeros(tensor[0][0].shape)

            # 遍历 tensor 中所有的元素，计算每个特征维度的平均值
            for t in tensor:
                mean_similarities += t[0].squeeze(0)  # 对每个 tensor 的第一维（[0]）进行累加

            # 计算平均值，除以 tensor 的长度
            mean_similarities /= len(tensor)

            # 对平均值进行排序，选出80%最小的特征维度
            mean_abs_values = mean_similarities.abs()  # 求平均值的绝对值
            sorted_indices = torch.argsort(mean_abs_values)  # 排序索引

            ### 存文件 
            # 转换为可以被JSON序列化的格式
            mean_similarities_list = mean_similarities.tolist()  # 将张量转换为列表
            sorted_indices_list = sorted_indices.tolist()  # 将张量转换为列表

            # 创建一个字典来存储这些数据
            data = {
                "mean_similarities": mean_similarities_list,
                "sorted_indices": sorted_indices_list
            }

            # 将数据保存到JSON文件
            with open('/FourierAttention/compressed_dims_file/data_ruler4k.json ', 'a') as json_file:  # your path to save the mean_similaritites and sorted_indices so that you can use compressed_dims_search_readfile.py to easily change the scale
                json.dump({layername: {name: data}}, json_file)
                json_file.write("\n")  


            # 选出前80%的特征维度
            num_non_critical = int(percent * mean_similarities.shape[0])  
            non_critical_indices = sorted_indices[:num_non_critical]  

            # 保存到字典
            non_critical_dims_dict[layername][name] = non_critical_indices.tolist()  # 转换为 list 保存


    # 保存非关键维度结果，可改
    with open("/FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_4k_fourier.json", "w", encoding="utf-8") as f:
        json.dump(non_critical_dims_dict, f, ensure_ascii=False, indent=4)

    return


from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # 模型路径，可改
    model_name = "meta-llama/Llama-3.2-3B"
    print("loading models and tokenizer...")
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="cuda:0")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    file_paths = [
        '/remote-home1/qqwang/outputs/Llama-3.2-3B-ruler/20250430_110126/predictions/Llama-3.2-3B-ruler/ruler_niah_multikey_1_4k.json',
        '/remote-home1/qqwang/outputs/Llama-3.2-3B-ruler/20250430_110126/predictions/Llama-3.2-3B-ruler/ruler_niah_multikey_2_4k.json',
        '/remote-home1/qqwang/outputs/Llama-3.2-3B-ruler/20250430_110126/predictions/Llama-3.2-3B-ruler/ruler_niah_multikey_3_4k.json',
        '/remote-home1/qqwang/outputs/Llama-3.2-3B-ruler/20250430_110126/predictions/Llama-3.2-3B-ruler/ruler_niah_multivalue_4k.json',
    ]

    # 用于保存所有提取出的origin_prompt
    origin_prompts = []

    # 遍历每个文件，只提取2条origin_prompt
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        prompts = [entry['origin_prompt'] for entry in data.values()][:2]
        origin_prompts.extend(prompts)

    # 打印总共提取的prompt数量
    print(f"Total origin prompts count: {len(origin_prompts)}")  # 应该是 8 * 2 = 16

    # 设置运行参数
    w_bit = 8  # 权重量化比特数，例如 8 或 4
    q_config = {}  # 量化的额外配置

    # 提取输入特征
    print("开始提取输入特征...")
    input_feat = run_awq(
        model=model,
        enc=tokenizer,
        w_bit=w_bit,
        q_config=q_config,
        input_texts=origin_prompts, # 输入文本的列表
        n_samples=10,  # 仅测试用，加载 10 个样本以加速运行
        seqlen=512,  # 输入序列长度
        auto_scale=False,
        mse_range=False,
    )

if __name__ == "__main__":
    main()
