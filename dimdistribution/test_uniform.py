import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取 JSON 文件
with open("/remote-home1/qqwang/hippofourier/dimdifferjson_4k/non_critical_dims_hippofourier_kvdiffer0.json", "r") as f:
    data = json.load(f)

# 提取指定层的索引
indices = data["Layer_26"]["self_attn.k_proj"]

# 排序
indices = sorted(indices)

# 分段和直方图参数
chunk_size = 128
bin_size = 16
num_bins = chunk_size // bin_size
bin_edges = np.arange(0, chunk_size + bin_size, bin_size)

# 计算每个 chunk 的 bin 计数并累加
chunk_counts = []

max_index = max(indices)
num_chunks = (max_index + 1 + chunk_size - 1) // chunk_size

for i in range(num_chunks):
    start = i * chunk_size
    end = start + chunk_size

    # 找到落在该段的索引
    chunk_indices = [idx for idx in indices if start <= idx < end]
    if not chunk_indices:
        continue

    # 相对当前 chunk 的偏移
    relative_indices = [idx - start for idx in chunk_indices]

    # 构造重排序映射：[0,64,1,65,...]
    reorder_map = []
    for j in range(chunk_size // 2):
        reorder_map.append(j)
        reorder_map.append(j + chunk_size // 2)
    reorder_dict = {orig: new for new, orig in enumerate(reorder_map)}

    # 将相对索引映射到新的排序位置
    reordered_positions = [reorder_dict[idx] for idx in relative_indices if idx in reorder_dict]

    # 计算 bin 计数
    hist, _ = np.histogram(reordered_positions, bins=bin_edges)
    chunk_counts.append(hist)


if chunk_counts:
    avg_counts = np.mean(chunk_counts, axis=0)
    # 绘制平均直方图
    plt.bar(bin_edges[:-1], avg_counts, width=bin_size, edgecolor='black', align='edge', zorder=2)

    # 设置字体大小
    ax = plt.gca()
    ax.set_xticks(bin_edges)
    ax.set_xticklabels(bin_edges, rotation=0, fontsize=16)
    ax.set_yticklabels(ax.get_yticks(), rotation=0, fontsize=16)

    # 设置标题和标签
    # plt.title("Average Distribution Across Chunks", fontsize=18)
    plt.xlabel("Head Dimension", fontsize=16)
    plt.ylabel("Average Count", fontsize=16)

    # # 网格设置：置于下方
    # plt.grid(True, zorder=1)

    plt.tight_layout()

    # 保存图像
    save_dir = "/remote-home1/qqwang"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "llama3.2-3b_average_distribution_layer26_reconstruct_all76%.pdf")
    plt.savefig(save_path, format='pdf')
    plt.close()
