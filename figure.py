import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data extracted from the tables
# Structure: {dataset: {model: {method: {'context_all': x, 'context_topk': x, 'memory_all': x, 'memory_topk': x}}}}
data = {
    'World Capital': {
        'GPT2 XL': {
            'Feature Ablation': {'context_all': 8.36, 'context_topk': 3.67, 'memory_all': 10.52, 'memory_topk': 6.05},
            'Integrated Gradient': {'context_all': 10.98, 'context_topk': 5.21, 'memory_all': 9.61,
                                    'memory_topk': 5.24},
            'Attention (Generation Head)': {'context_all': 10.63, 'context_topk': 5.40, 'memory_all': 10.40,
                                            'memory_topk': 5.87},
            'Attention (MI)': {'context_all': 8.62, 'context_topk': 3.43, 'memory_all': 9.35, 'memory_topk': 4.70}
        },
        'Pythia 2.8b': {
            'Feature Ablation': {'context_all': 7.66, 'context_topk': 3.40, 'memory_all': 8.36, 'memory_topk': 4.08},
            'Integrated Gradient': {'context_all': 13.12, 'context_topk': 6.58, 'memory_all': 13.32,
                                    'memory_topk': 8.89},
            'Attention (Generation Head)': {'context_all': 12.25, 'context_topk': 5.08, 'memory_all': 10.89,
                                            'memory_topk': 4.98},
            'Attention (MI)': {'context_all': 9.18, 'context_topk': 3.27, 'memory_all': 9.89, 'memory_topk': 4.27}
        },
        'Qwen/Qwen2.5-3B': {
            'Feature Ablation': {'context_all': 11.40, 'context_topk': 4.98, 'memory_all': 13.20, 'memory_topk': 9.83},
            'Integrated Gradient': {'context_all': 11.14, 'context_topk': 5.14, 'memory_all': 9.71,
                                    'memory_topk': 5.46},
            'Attention (Generation Head)': {'context_all': 12.00, 'context_topk': 5.23, 'memory_all': 10.78,
                                            'memory_topk': 5.65},
            'Attention (MI)': {'context_all': 9.05, 'context_topk': 3.35, 'memory_all': 9.52, 'memory_topk': 4.79}
        },
        'EleutherAI/pythia-6.9b': {
            'Feature Ablation': {'context_all': 16.51, 'context_topk': 9.97, 'memory_all': 13.76, 'memory_topk': 10.71},
            'Integrated Gradient': {'context_all': 13.34, 'context_topk': 7.17, 'memory_all': 11.65,
                                    'memory_topk': 8.13},
            'Attention (Generation Head)': {'context_all': 11.46, 'context_topk': 4.75, 'memory_all': 9.18,
                                            'memory_topk': 4.55},
            'Attention (MI)': {'context_all': 10.35, 'context_topk': 3.90, 'memory_all': 9.13, 'memory_topk': 4.30}
        },
        'Qwen/Qwen2.5-7B': {
            'Feature Ablation': {'context_all': 9.33, 'context_topk': 4.38, 'memory_all': 14.31, 'memory_topk': 10.31},
            'Integrated Gradient': {'context_all': 10.32, 'context_topk': 4.90, 'memory_all': 10.22,
                                    'memory_topk': 4.87},
            'Attention (Generation Head)': {'context_all': 9.75, 'context_topk': 4.66, 'memory_all': 9.62,
                                            'memory_topk': 4.81},
            'Attention (MI)': {'context_all': 9.41, 'context_topk': 3.87, 'memory_all': 9.75, 'memory_topk': 4.86}
        }
    },
    'Counterfact': {
        'GPT2 XL': {
            'Feature Ablation': {'context_all': 13.01, 'context_topk': 6.96, 'memory_all': 12.84, 'memory_topk': 7.45},
            'Integrated Gradient': {'context_all': 11.68, 'context_topk': 5.87, 'memory_all': 11.29,
                                    'memory_topk': 6.03},
            'Attention (Generation Head)': {'context_all': 11.56, 'context_topk': 4.67, 'memory_all': 11.08,
                                            'memory_topk': 4.78},
            'Attention (MI)': {'context_all': 10.32, 'context_topk': 3.64, 'memory_all': 10.41, 'memory_topk': 4.05}
        },
        'Pythia 2.8b': {
            'Feature Ablation': {'context_all': 17.51, 'context_topk': 12.60, 'memory_all': 15.78,
                                 'memory_topk': 11.73},
            'Integrated Gradient': {'context_all': 13.12, 'context_topk': 6.58, 'memory_all': 13.32,
                                    'memory_topk': 8.89},
            'Attention (Generation Head)': {'context_all': 11.73, 'context_topk': 4.58, 'memory_all': 10.97,
                                            'memory_topk': 4.45},
            'Attention (MI)': {'context_all': 11.36, 'context_topk': 3.76, 'memory_all': 10.96, 'memory_topk': 4.46}
        },
        'Qwen/Qwen2.5-3B': {
            'Feature Ablation': {'context_all': 11.57, 'context_topk': 5.24, 'memory_all': 11.90, 'memory_topk': 5.85},
            'Integrated Gradient': {'context_all': 12.01, 'context_topk': 5.26, 'memory_all': 11.90,
                                    'memory_topk': 5.24},
            'Attention (Generation Head)': {'context_all': 12.46, 'context_topk': 5.55, 'memory_all': 12.41,
                                            'memory_topk': 5.56},
            'Attention (MI)': {'context_all': 9.67, 'context_topk': 3.74, 'memory_all': 10.21, 'memory_topk': 3.76}
        },
        'EleutherAI/pythia-6.9b': {
            'Feature Ablation': {'context_all': 17.66, 'context_topk': 11.85, 'memory_all': 16.57,
                                 'memory_topk': 12.26},
            'Integrated Gradient': {'context_all': 13.75, 'context_topk': 6.24, 'memory_all': 12.80,
                                    'memory_topk': 6.44},
            'Attention (Generation Head)': {'context_all': 11.15, 'context_topk': 4.04, 'memory_all': 9.82,
                                            'memory_topk': 3.95},
            'Attention (MI)': {'context_all': 11.13, 'context_topk': 3.62, 'memory_all': 10.94, 'memory_topk': 4.22}
        },
        'Qwen/Qwen2.5-7B': {
            'Feature Ablation': {'context_all': 12.95, 'context_topk': 6.57, 'memory_all': 13.39, 'memory_topk': 7.48},
            'Integrated Gradient': {'context_all': 10.48, 'context_topk': 3.92, 'memory_all': 10.59,
                                    'memory_topk': 3.96},
            'Attention (Generation Head)': {'context_all': 10.55, 'context_topk': 4.05, 'memory_all': 10.64,
                                            'memory_topk': 4.02},
            'Attention (MI)': {'context_all': 9.83, 'context_topk': 3.49, 'memory_all': 10.86, 'memory_topk': 4.31}
        }
    },
    'Fakepedia': {
        'GPT2 XL': {
            'Feature Ablation': {'context_all': 73.99, 'context_topk': 9.27, 'memory_all': 73.45, 'memory_topk': 9.65},
            'Integrated Gradient': {'context_all': 71.75, 'context_topk': 5.52, 'memory_all': 70.38,
                                    'memory_topk': 4.96},
            'Attention (Generation Head)': {'context_all': 72.90, 'context_topk': 4.24, 'memory_all': 71.96,
                                            'memory_topk': 4.17},
            'Attention (MI)': {'context_all': 75.33, 'context_topk': 3.47, 'memory_all': 70.79, 'memory_topk': 3.96}
        },
        'Pythia 2.8b': {
            'Feature Ablation': {'context_all': 81.62, 'context_topk': 10.09, 'memory_all': 76.33,
                                 'memory_topk': 12.56},
            'Integrated Gradient': {'context_all': 77.07, 'context_topk': 4.79, 'memory_all': 72.61,
                                    'memory_topk': 5.83},
            'Attention (Generation Head)': {'context_all': 77.78, 'context_topk': 3.69, 'memory_all': 73.30,
                                            'memory_topk': 4.37},
            'Attention (MI)': {'context_all': 77.11, 'context_topk': 3.34, 'memory_all': 74.17, 'memory_topk': 3.59}
        },
        'Qwen/Qwen2.5-3B': {
            'Feature Ablation': {'context_all': 73.20, 'context_topk': 7.18, 'memory_all': 73.11, 'memory_topk': 7.84},
            'Integrated Gradient': {'context_all': 71.62, 'context_topk': 4.24, 'memory_all': 71.65,
                                    'memory_topk': 4.03},
            'Attention (Generation Head)': {'context_all': 75.46, 'context_topk': 6.61, 'memory_all': 75.49,
                                            'memory_topk': 6.90},
            'Attention (MI)': {'context_all': 72.15, 'context_topk': 3.57, 'memory_all': 74.10, 'memory_topk': 4.13}
        },
        'EleutherAI/pythia-6.9b': {
            'Feature Ablation': {'context_all': 76.48, 'context_topk': 11.17, 'memory_all': 76.53,
                                 'memory_topk': 12.68},
            'Integrated Gradient': {'context_all': 70.95, 'context_topk': 4.38, 'memory_all': 71.45,
                                    'memory_topk': 4.38},
            'Attention (Generation Head)': {'context_all': 71.50, 'context_topk': 3.84, 'memory_all': 71.94,
                                            'memory_topk': 3.91},
            'Attention (MI)': {'context_all': 73.13, 'context_topk': 3.47, 'memory_all': 73.37, 'memory_topk': 3.70}
        },
        'Qwen/Qwen2.5-7B': {
            'Feature Ablation': {'context_all': 72.65, 'context_topk': 8.04, 'memory_all': 72.83, 'memory_topk': 7.82},
            'Integrated Gradient': {'context_all': 70.49, 'context_topk': 3.71, 'memory_all': 71.15,
                                    'memory_topk': 3.76},
            'Attention (Generation Head)': {'context_all': 70.94, 'context_topk': 3.22, 'memory_all': 72.10,
                                            'memory_topk': 3.47},
            'Attention (MI)': {'context_all': 70.46, 'context_topk': 3.36, 'memory_all': 73.64, 'memory_topk': 4.11}
        }
    },
    'ConflictQA': {
        'GPT2 XL': {
            'Feature Ablation': {'context_all': 72.32, 'context_topk': 7.74, 'memory_all': 71.86, 'memory_topk': 8.34},
            'Integrated Gradient': {'context_all': 70.90, 'context_topk': 5.50, 'memory_all': 69.69,
                                    'memory_topk': 5.60},
            'Attention (Generation Head)': {'context_all': 70.59, 'context_topk': 3.46, 'memory_all': 69.67,
                                            'memory_topk': 3.55},
            'Attention (MI)': {'context_all': 69.76, 'context_topk': 3.38, 'memory_all': 70.39, 'memory_topk': 3.94}
        },
        'Pythia 2.8b': {
            'Feature Ablation': {'context_all': 76.01, 'context_topk': 10.80, 'memory_all': 73.33,
                                 'memory_topk': 11.26},
            'Integrated Gradient': {'context_all': 73.00, 'context_topk': 5.41, 'memory_all': 70.52,
                                    'memory_topk': 5.89},
            'Attention (Generation Head)': {'context_all': 71.26, 'context_topk': 3.72, 'memory_all': 68.95,
                                            'memory_topk': 3.74},
            'Attention (MI)': {'context_all': 70.94, 'context_topk': 3.33, 'memory_all': 70.94, 'memory_topk': 4.25}
        },
        'Qwen/Qwen2.5-3B': {
            'Feature Ablation': {'context_all': 72.44, 'context_topk': 6.61, 'memory_all': 71.00, 'memory_topk': 6.94},
            'Integrated Gradient': {'context_all': 70.25, 'context_topk': 3.93, 'memory_all': 68.67,
                                    'memory_topk': 3.65},
            'Attention (Generation Head)': {'context_all': 73.75, 'context_topk': 4.32, 'memory_all': 72.05,
                                            'memory_topk': 4.73},
            'Attention (MI)': {'context_all': 70.84, 'context_topk': 3.16, 'memory_all': 70.72, 'memory_topk': 3.66}
        },
        'EleutherAI/pythia-6.9b': {
            'Feature Ablation': {'context_all': 76.21, 'context_topk': 11.37, 'memory_all': 74.24,
                                 'memory_topk': 11.52},
            'Integrated Gradient': {'context_all': 73.03, 'context_topk': 5.97, 'memory_all': 71.04,
                                    'memory_topk': 6.13},
            'Attention (Generation Head)': {'context_all': 70.36, 'context_topk': 3.62, 'memory_all': 68.20,
                                            'memory_topk': 3.71},
            'Attention (MI)': {'context_all': 69.26, 'context_topk': 3.36, 'memory_all': 71.09, 'memory_topk': 3.75}
        },
        'Qwen/Qwen2.5-7B': {
            'Feature Ablation': {'context_all': 71.47, 'context_topk': 6.53, 'memory_all': 72.41, 'memory_topk': 6.97},
            'Integrated Gradient': {'context_all': 70.30, 'context_topk': 3.80, 'memory_all': 70.90,
                                    'memory_topk': 3.71},
            'Attention (Generation Head)': {'context_all': 72.82, 'context_topk': 4.44, 'memory_all': 73.13,
                                            'memory_topk': 4.50},
            'Attention (MI)': {'context_all': 72.10, 'context_topk': 3.40, 'memory_all': 72.98, 'memory_topk': 4.04}
        }
    }
}

# Define colors for methods
method_colors = {
    'Feature Ablation': '#1f77b4',
    'Integrated Gradient': '#ff7f0e',
    'Attention (Generation Head)': '#2ca02c',
    'Attention (MI)': '#d62728'
}

# Get all models and datasets
models = ['GPT2 XL', 'Pythia 2.8b', 'Qwen/Qwen2.5-3B', 'EleutherAI/pythia-6.9b', 'Qwen/Qwen2.5-7B']
datasets = ['World Capital', 'Counterfact', 'Fakepedia', 'ConflictQA']
methods = ['Feature Ablation', 'Integrated Gradient', 'Attention (Generation Head)', 'Attention (MI)']

# Create plots for Top-K ranks
for model in models:
    fig, ax = plt.subplots(figsize=(16, 8))

    # Prepare data for plotting
    x = np.arange(len(datasets))
    width = 0.09  # Width of each bar
    multiplier = 0

    # For each method, create pairs of bars (context and memory)
    for i, method in enumerate(methods):
        context_ranks = []
        memory_ranks = []

        for dataset in datasets:
            context_ranks.append(data[dataset][model][method]['context_topk'])
            memory_ranks.append(data[dataset][model][method]['memory_topk'])

        # Plot context bars
        offset = width * multiplier
        bars1 = ax.bar(x + offset, context_ranks, width,
                       label=f'{method} (Context)',
                       color=method_colors[method], alpha=0.8)

        # Plot memory bars right next to context bars
        bars2 = ax.bar(x + offset + width, memory_ranks, width,
                       label=f'{method} (Memory)',
                       color=method_colors[method], alpha=0.5)

        multiplier += 2  # Move to next pair position

    # Set labels and title
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Rank (Top-K) - Lower is Better', fontsize=12)
    ax.set_title(f'Context vs Memory Rank (Top-K) - {model}', fontsize=14, fontweight='bold')

    # Set x-axis ticks and labels
    ax.set_xticks(x + width * 3.5)  # Center the labels
    ax.set_xticklabels(datasets)

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

# Create plots for All ranks
for model in models:
    fig, ax = plt.subplots(figsize=(16, 8))

    # Prepare data for plotting
    x = np.arange(len(datasets))
    width = 0.09  # Width of each bar
    multiplier = 0

    # For each method, create pairs of bars (context and memory)
    for i, method in enumerate(methods):
        context_ranks = []
        memory_ranks = []

        for dataset in datasets:
            context_ranks.append(data[dataset][model][method]['context_all'])
            memory_ranks.append(data[dataset][model][method]['memory_all'])

        # Plot context bars
        offset = width * multiplier
        bars1 = ax.bar(x + offset, context_ranks, width,
                       label=f'{method} (Context)',
                       color=method_colors[method], alpha=0.8)

        # Plot memory bars right next to context bars
        bars2 = ax.bar(x + offset + width, memory_ranks, width,
                       label=f'{method} (Memory)',
                       color=method_colors[method], alpha=0.5)

        multiplier += 2  # Move to next pair position

    # Set labels and title
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Rank (All) - Lower is Better', fontsize=12)
    ax.set_title(f'Context vs Memory Rank (All) - {model}', fontsize=14, fontweight='bold')

    # Set x-axis ticks and labels
    ax.set_xticks(x + width * 3.5)  # Center the labels
    ax.set_xticklabels(datasets)

    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

# Create violin plots showing distribution across all models
print("\n=== Violin Plots: Distribution Across All Models ===\n")


# Prepare data for violin plots
def prepare_violin_data(rank_type='topk'):
    violin_data = {
        'Context': {method: [] for method in methods},
        'Memory': {method: [] for method in methods}
    }

    for dataset in datasets:
        for model in models:
            for method in methods:
                if rank_type == 'topk':
                    context_val = data[dataset][model][method]['context_topk']
                    memory_val = data[dataset][model][method]['memory_topk']
                else:  # 'all'
                    context_val = data[dataset][model][method]['context_all']
                    memory_val = data[dataset][model][method]['memory_all']

                violin_data['Context'][method].append(context_val)
                violin_data['Memory'][method].append(memory_val)

    return violin_data


# Create violin plot for Top-K ranks
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

violin_data_topk = prepare_violin_data('topk')

# Plot Context Answer Instances
positions = np.arange(len(methods))
violins_context = []
for i, method in enumerate(methods):
    parts = ax1.violinplot([violin_data_topk['Context'][method]], positions=[i],
                           widths=0.7, showmeans=True, showextrema=True)
    violins_context.append(parts)

    # Color the violin plots
    for pc in parts['bodies']:
        pc.set_facecolor(method_colors[method])
        pc.set_alpha(0.8)
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)

ax1.set_xticks(positions)
ax1.set_xticklabels(methods, rotation=30, ha='right')
ax1.set_ylabel('Rank (Top-K)', fontsize=12)
ax1.set_title('Context Answer Instances - Top-K Ranks\nDistribution Across All Models', fontsize=14)
ax1.grid(True, alpha=0.3, axis='y')

# Plot Memory Answer Instances
violins_memory = []
for i, method in enumerate(methods):
    parts = ax2.violinplot([violin_data_topk['Memory'][method]], positions=[i],
                           widths=0.7, showmeans=True, showextrema=True)
    violins_memory.append(parts)

    # Color the violin plots
    for pc in parts['bodies']:
        pc.set_facecolor(method_colors[method])
        pc.set_alpha(0.5)
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)

ax2.set_xticks(positions)
ax2.set_xticklabels(methods, rotation=30, ha='right')
ax2.set_ylabel('Rank (Top-K)', fontsize=12)
ax2.set_title('Memory Answer Instances - Top-K Ranks\nDistribution Across All Models', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Rank Distribution (Top-K) Across All Models and Datasets', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Create violin plot for All ranks
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

violin_data_all = prepare_violin_data('all')

# Plot Context Answer Instances
positions = np.arange(len(methods))
violins_context = []
for i, method in enumerate(methods):
    parts = ax1.violinplot([violin_data_all['Context'][method]], positions=[i],
                           widths=0.7, showmeans=True, showextrema=True)
    violins_context.append(parts)

    # Color the violin plots
    for pc in parts['bodies']:
        pc.set_facecolor(method_colors[method])
        pc.set_alpha(0.8)
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)

ax1.set_xticks(positions)
ax1.set_xticklabels(methods, rotation=30, ha='right')
ax1.set_ylabel('Rank (All)', fontsize=12)
ax1.set_title('Context Answer Instances - All Ranks\nDistribution Across All Models', fontsize=14)
ax1.grid(True, alpha=0.3, axis='y')

# Plot Memory Answer Instances
violins_memory = []
for i, method in enumerate(methods):
    parts = ax2.violinplot([violin_data_all['Memory'][method]], positions=[i],
                           widths=0.7, showmeans=True, showextrema=True)
    violins_memory.append(parts)

    # Color the violin plots
    for pc in parts['bodies']:
        pc.set_facecolor(method_colors[method])
        pc.set_alpha(0.5)
    parts['cmeans'].set_color('red')
    parts['cmeans'].set_linewidth(2)

ax2.set_xticks(positions)
ax2.set_xticklabels(methods, rotation=30, ha='right')
ax2.set_ylabel('Rank (All)', fontsize=12)
ax2.set_title('Memory Answer Instances - All Ranks\nDistribution Across All Models', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Rank Distribution (All) Across All Models and Datasets', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Create combined violin plot with both Context and Memory on same plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Top-K ranks combined
positions = np.arange(len(methods))
width = 0.35

for i, method in enumerate(methods):
    # Context violins
    parts_c = ax1.violinplot([violin_data_topk['Context'][method]],
                             positions=[i - width / 2],
                             widths=width, showmeans=True, showextrema=True)
    for pc in parts_c['bodies']:
        pc.set_facecolor(method_colors[method])
        pc.set_alpha(0.8)
    parts_c['cmeans'].set_color('darkred')

    # Memory violins
    parts_m = ax1.violinplot([violin_data_topk['Memory'][method]],
                             positions=[i + width / 2],
                             widths=width, showmeans=True, showextrema=True)
    for pc in parts_m['bodies']:
        pc.set_facecolor(method_colors[method])
        pc.set_alpha(0.5)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    parts_m['cmeans'].set_color('darkblue')

ax1.set_xticks(positions)
ax1.set_xticklabels(methods, rotation=0)
ax1.set_ylabel('Rank (Top-K)', fontsize=12)
ax1.set_title('Top-K Rank Distribution - Context (solid) vs Memory (transparent)', fontsize=14)
ax1.grid(True, alpha=0.3, axis='y')

# All ranks combined
for i, method in enumerate(methods):
    # Context violins
    parts_c = ax2.violinplot([violin_data_all['Context'][method]],
                             positions=[i - width / 2],
                             widths=width, showmeans=True, showextrema=True)
    for pc in parts_c['bodies']:
        pc.set_facecolor(method_colors[method])
        pc.set_alpha(0.8)
    parts_c['cmeans'].set_color('darkred')

    # Memory violins
    parts_m = ax2.violinplot([violin_data_all['Memory'][method]],
                             positions=[i + width / 2],
                             widths=width, showmeans=True, showextrema=True)
    for pc in parts_m['bodies']:
        pc.set_facecolor(method_colors[method])
        pc.set_alpha(0.5)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    parts_m['cmeans'].set_color('darkblue')

ax2.set_xticks(positions)
ax2.set_xticklabels(methods, rotation=0)
ax2.set_ylabel('Rank (All)', fontsize=12)
ax2.set_title('All Rank Distribution - Context (solid) vs Memory (transparent)', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')

# Add legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='gray', alpha=0.8, label='Context Answer'),
    Patch(facecolor='gray', alpha=0.5, label='Memory Answer'),
    Patch(facecolor='darkred', label='Mean (Context)'),
    Patch(facecolor='darkblue', label='Mean (Memory)')
]
ax1.legend(handles=legend_elements, loc='upper right')

# Create box plots as an alternative visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))


# Prepare data for box plots
def prepare_box_data(rank_type='topk'):
    box_data = []
    labels = []
    colors = []

    for method in methods:
        context_vals = []
        memory_vals = []

        for dataset in datasets:
            for model in models:
                if rank_type == 'topk':
                    context_vals.append(data[dataset][model][method]['context_topk'])
                    memory_vals.append(data[dataset][model][method]['memory_topk'])
                else:
                    context_vals.append(data[dataset][model][method]['context_all'])
                    memory_vals.append(data[dataset][model][method]['memory_all'])

        box_data.extend([context_vals, memory_vals])
        labels.extend([f'{method}\n(Context)', f'{method}\n(Memory)'])
        colors.extend([method_colors[method], method_colors[method]])

    return box_data, labels, colors


# Top-K box plot
box_data_topk, labels_topk, colors_topk = prepare_box_data('topk')
bp1 = axes[0, 0].boxplot(box_data_topk, labels=labels_topk, patch_artist=True)
for patch, color, i in zip(bp1['boxes'], colors_topk, range(len(colors_topk))):
    patch.set_facecolor(color)
    patch.set_alpha(0.8 if i % 2 == 0 else 0.5)
axes[0, 0].set_ylabel('Rank (Top-K)', fontsize=12)
axes[0, 0].set_title('Top-K Rank Distribution (Box Plot)', fontsize=14)
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].tick_params(axis='x', rotation=45, labelsize=9)

# All ranks box plot
box_data_all, labels_all, colors_all = prepare_box_data('all')
bp2 = axes[0, 1].boxplot(box_data_all, labels=labels_all, patch_artist=True)
for patch, color, i in zip(bp2['boxes'], colors_all, range(len(colors_all))):
    patch.set_facecolor(color)
    patch.set_alpha(0.8 if i % 2 == 0 else 0.5)
axes[0, 1].set_ylabel('Rank (All)', fontsize=12)
axes[0, 1].set_title('All Rank Distribution (Box Plot)', fontsize=14)
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=9)

# Summary statistics for Top-K
axes[1, 0].axis('off')
summary_topk = []
for method in methods:
    context_mean = np.mean(violin_data_topk['Context'][method])
    context_std = np.std(violin_data_topk['Context'][method])
    memory_mean = np.mean(violin_data_topk['Memory'][method])
    memory_std = np.std(violin_data_topk['Memory'][method])
    summary_topk.append([method, f'{context_mean:.2f}±{context_std:.2f}', f'{memory_mean:.2f}±{memory_std:.2f}'])

table1 = axes[1, 0].table(cellText=summary_topk,
                          colLabels=['Method', 'Context (mean±std)', 'Memory (mean±std)'],
                          cellLoc='center',
                          loc='center')
table1.auto_set_font_size(False)
table1.set_fontsize(10)
table1.scale(1, 2)
axes[1, 0].set_title('Summary Statistics - Top-K Ranks', fontsize=14, pad=20)

# Summary statistics for All ranks
axes[1, 1].axis('off')
summary_all = []
for method in methods:
    context_mean = np.mean(violin_data_all['Context'][method])
    context_std = np.std(violin_data_all['Context'][method])
    memory_mean = np.mean(violin_data_all['Memory'][method])
    memory_std = np.std(violin_data_all['Memory'][method])
    summary_all.append([method, f'{context_mean:.2f}±{context_std:.2f}', f'{memory_mean:.2f}±{memory_std:.2f}'])

table2 = axes[1, 1].table(cellText=summary_all,
                          colLabels=['Method', 'Context (mean±std)', 'Memory (mean±std)'],
                          cellLoc='center',
                          loc='center')
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1, 2)
axes[1, 1].set_title('Summary Statistics - All Ranks', fontsize=14, pad=20)

plt.suptitle('Rank Distribution Analysis Across All Models and Datasets', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Print detailed statistics
print("\n=== Detailed Statistics ===\n")
print("Top-K Ranks:")
print("-" * 80)
print(f"{'Method':<30} {'Context (mean±std)':<25} {'Memory (mean±std)':<25}")
print("-" * 80)
for method in methods:
    context_vals = violin_data_topk['Context'][method]
    memory_vals = violin_data_topk['Memory'][method]
    print(f"{method:<30} {np.mean(context_vals):.2f}±{np.std(context_vals):.2f} "
          f"(min:{np.min(context_vals):.1f}, max:{np.max(context_vals):.1f})"
          f"   {np.mean(memory_vals):.2f}±{np.std(memory_vals):.2f} "
          f"(min:{np.min(memory_vals):.1f}, max:{np.max(memory_vals):.1f})")

print("\n\nAll Ranks:")
print("-" * 80)
print(f"{'Method':<30} {'Context (mean±std)':<25} {'Memory (mean±std)':<25}")
print("-" * 80)
for method in methods:
    context_vals = violin_data_all['Context'][method]
    memory_vals = violin_data_all['Memory'][method]
    print(f"{method:<30} {np.mean(context_vals):.2f}±{np.std(context_vals):.2f} "
          f"(min:{np.min(context_vals):.1f}, max:{np.max(context_vals):.1f})"
          f"   {np.mean(memory_vals):.2f}±{np.std(memory_vals):.2f} "
          f"(min:{np.min(memory_vals):.1f}, max:{np.max(memory_vals):.1f})")

# Create plots for Top-K ranks - Alternative layout
for model in models:
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(datasets))
    width = 0.18

    # Plot all context bars together
    for i, method in enumerate(methods):
        context_ranks = []
        for dataset in datasets:
            context_ranks.append(data[dataset][model][method]['context_topk'])

        offset = -width * 1.5 + i * width / 2
        bars = ax.bar(x + offset, context_ranks, width / 2,
                      label=f'{method}',
                      color=method_colors[method], alpha=0.8)

    # Plot all memory bars together
    for i, method in enumerate(methods):
        memory_ranks = []
        for dataset in datasets:
            memory_ranks.append(data[dataset][model][method]['memory_topk'])

        offset = width / 2 + i * width / 2
        bars = ax.bar(x + offset, memory_ranks, width / 2,
                      color=method_colors[method], alpha=0.5, hatch='//')

    # Create custom legend
    from matplotlib.patches import Patch

    legend_elements = []
    for method in methods:
        legend_elements.append(Patch(facecolor=method_colors[method], alpha=0.8, label=method))
    legend_elements.append(Patch(facecolor='gray', alpha=0.8, label='Context Answer'))
    legend_elements.append(Patch(facecolor='gray', alpha=0.5, hatch='//', label='Memory Answer'))

    ax.legend(handles=legend_elements, loc='upper right')

    # Add vertical line to separate context and memory groups
    for i in range(len(datasets)):
        ax.axvline(x=i, color='black', linestyle=':', alpha=0.3)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Rank (Top-K) - Lower is Better', fontsize=12)
    ax.set_title(f'Context vs Memory Rank (Top-K) - {model}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

# Create plots for All ranks - Alternative layout
for model in models:
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(datasets))
    width = 0.18

    # Plot all context bars together
    for i, method in enumerate(methods):
        context_ranks = []
        for dataset in datasets:
            context_ranks.append(data[dataset][model][method]['context_all'])

        offset = -width * 1.5 + i * width / 2
        bars = ax.bar(x + offset, context_ranks, width / 2,
                      label=f'{method}',
                      color=method_colors[method], alpha=0.8)

    # Plot all memory bars together
    for i, method in enumerate(methods):
        memory_ranks = []
        for dataset in datasets:
            memory_ranks.append(data[dataset][model][method]['memory_all'])

        offset = width / 2 + i * width / 2
        bars = ax.bar(x + offset, memory_ranks, width / 2,
                      color=method_colors[method], alpha=0.5, hatch='//')

    # Create custom legend
    from matplotlib.patches import Patch

    legend_elements = []
    for method in methods:
        legend_elements.append(Patch(facecolor=method_colors[method], alpha=0.8, label=method))
    legend_elements.append(Patch(facecolor='gray', alpha=0.8, label='Context Answer'))
    legend_elements.append(Patch(facecolor='gray', alpha=0.5, hatch='//', label='Memory Answer'))

    ax.legend(handles=legend_elements, loc='upper right')

    # Add vertical line to separate context and memory groups
    for i in range(len(datasets)):
        ax.axvline(x=i, color='black', linestyle=':', alpha=0.3)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Rank (All) - Lower is Better', fontsize=12)
    ax.set_title(f'Context vs Memory Rank (All) - {model}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()