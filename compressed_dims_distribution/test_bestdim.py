import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Read JSON file
with open(" ", "r") as f:  #your path to /FourierAttention/compressed_dims_file/llama3.2-3b/compressed_dims_4k_fourier.json"
    data = json.load(f)

# Extract indices of specified layer
indices = data["Layer_0"]["self_attn.k_proj"]

# Sort
indices = sorted(indices)

# Chunking and histogram parameters
chunk_size = 128
bin_size = 16
num_bins = chunk_size // bin_size
bin_edges = np.arange(0, chunk_size + bin_size, bin_size)

# Calculate bin counts for each chunk and accumulate
chunk_counts = []

max_index = max(indices)
num_chunks = (max_index + 1 + chunk_size - 1) // chunk_size

for i in range(num_chunks):
    start = i * chunk_size
    end = start + chunk_size

    # Find indices falling in this chunk
    chunk_indices = [idx for idx in indices if start <= idx < end]
    if not chunk_indices:
        continue

    # Relative offset for current chunk
    relative_indices = [idx - start for idx in chunk_indices]

    # Construct reorder mapping: [0,64,1,65,...]
    reorder_map = []
    for j in range(chunk_size // 2):
        reorder_map.append(j)
        reorder_map.append(j + chunk_size // 2)
    reorder_dict = {orig: new for new, orig in enumerate(reorder_map)}

    # Map relative indices to new sort positions
    reordered_positions = [reorder_dict[idx] for idx in relative_indices if idx in reorder_dict]

    # Calculate bin counts
    hist, _ = np.histogram(reordered_positions, bins=bin_edges)
    chunk_counts.append(hist)


if chunk_counts:
    avg_counts = np.mean(chunk_counts, axis=0)
    # Plot average histogram
    plt.bar(bin_edges[:-1], avg_counts, width=bin_size, edgecolor='black', align='edge', zorder=2)

    # Set font size
    ax = plt.gca()
    ax.set_xticks(bin_edges)
    ax.set_xticklabels(bin_edges, rotation=0, fontsize=16)
    ax.set_yticklabels(ax.get_yticks(), rotation=0, fontsize=16)

    # Set title and labels
    # plt.title("Average Distribution Across Chunks", fontsize=18)
    plt.xlabel("Head Dimension", fontsize=16)
    plt.ylabel("Average Count", fontsize=16)

    # # Grid settings below
    # plt.grid(True, zorder=1)

    plt.tight_layout()

    # Save image
    save_dir = " "
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "llama3.2-3b_average_distribution_layer0_reconstruct.pdf")
    plt.savefig(save_path, format='pdf')
    plt.close()
