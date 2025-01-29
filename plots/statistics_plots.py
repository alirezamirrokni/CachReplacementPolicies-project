import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Create a directory to save plots
output_dir = "cache_policy_plots"
os.makedirs(output_dir, exist_ok=True)

# Set the style for seaborn
sns.set(style="whitegrid")

# Define cache sizes
cache_sizes = [5000, 10000, 50000]

# Define the data
data = {
    'A42': {
        'LRU': [17.7, 18.54, 30.69],
        'N-Hit': [18.31, 20.4, 41.53],
        'Belady': [27.28, 33.78, 68.81],
        'ARC': [22.7, 25.44, 39.18],
        'LARC': [21.36, 23.31, 24.45],
    },
    'A108': {
        'LRU': [15.37, 17.17, 21.98],
        'N-Hit': [11.46, 11.53, 22.2],
        'Belady': [22.36, 25.53, 38.05],
        'ARC': [16.57, 18.09, 25.29],
        'LARC': [13.28, 14.65, 21.97],
    },
    'A129': {
        'LRU': [23.97, 26.75, 46.68],
        'N-Hit': [13.8, 15.33, 48.16],
        'Belady': [34.55, 42.2, 59.4],
        'ARC': [26.14, 29.02, 52.49],
        'LARC': [21.11, 23.27, 48.02],
    },
    'A669': {
        'LRU': [78.69, 91.71, 96.69],
        'N-Hit': [78.3, 89.86, 96.73],
        'Belady': [88.72, 94.61, 97.86],
        'ARC': [80.13, 91.94, 96.97],
        'LARC': [83.34, 92.62, 95.91],
    }
}

# Convert the data into a pandas DataFrame
records = []
for dataset, policies in data.items():
    for policy, hits in policies.items():
        for size, hit_ratio in zip(cache_sizes, hits):
            records.append({
                'Dataset': dataset,
                'Policy': policy,
                'Cache Size': size,
                'Hit Ratio (%)': hit_ratio
            })

df = pd.DataFrame.from_records(records)

# Define a color palette
palette = sns.color_palette("bright", len(df['Policy'].unique()))

# =========================================
# 1. Individual Line Plots for Each Dataset
# =========================================
datasets = df['Dataset'].unique()
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    subset = df[df['Dataset'] == dataset]
    sns.lineplot(data=subset, x='Cache Size', y='Hit Ratio (%)',
                 hue='Policy', marker='o', palette=palette)
    plt.title(f'Hit Ratio vs Cache Size for Dataset {dataset}', fontsize=14)
    plt.xlabel('Cache Size', fontsize=12)
    plt.ylabel('Hit Ratio (%)', fontsize=12)
    plt.xticks(cache_sizes)
    plt.legend(title='Policy', fontsize=10, title_fontsize=12)
    plt.ylim(0, 100)

    # Save plot
    plt.savefig(os.path.join(output_dir, f"hit_ratio_{dataset}.png"))
    plt.close()

# =========================================
# 2. Summary Line Plot Across All Datasets
# =========================================
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='Cache Size', y='Hit Ratio (%)',
             hue='Policy', style='Dataset', markers=True, dashes=False, palette=palette)
plt.title('Hit Ratio vs Cache Size Across All Datasets', fontsize=16)
plt.xlabel('Cache Size', fontsize=14)
plt.ylabel('Hit Ratio (%)', fontsize=14)
plt.xticks(cache_sizes)
plt.legend(title='Policy', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.ylim(0, 100)

# Save plot
plt.savefig(os.path.join(output_dir, "summary_hit_ratio.png"))
plt.close()

# =========================================
# 3. Bar Plots for Each Dataset
# =========================================
for dataset in datasets:
    subset = df[df['Dataset'] == dataset]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=subset, x='Cache Size', y='Hit Ratio (%)', hue='Policy', palette=palette)
    plt.title(f'Hit Ratio by Policy and Cache Size for Dataset {dataset}', fontsize=14)
    plt.xlabel('Cache Size', fontsize=12)
    plt.ylabel('Hit Ratio (%)', fontsize=12)
    plt.legend(title='Policy', fontsize=10, title_fontsize=12)
    plt.ylim(0, 100)

    # Save plot
    plt.savefig(os.path.join(output_dir, f"barplot_{dataset}.png"))
    plt.close()

# =========================================
# 4. Per-Policy Line Plots Across Datasets
# =========================================
policies = df['Policy'].unique()
for policy in policies:
    plt.figure(figsize=(10, 6))
    subset = df[df['Policy'] == policy]
    sns.lineplot(data=subset, x='Cache Size', y='Hit Ratio (%)',
                 hue='Dataset', marker='o', palette="tab10")
    plt.title(f'Performance of {policy} Across Datasets', fontsize=14)
    plt.xlabel('Cache Size', fontsize=12)
    plt.ylabel('Hit Ratio (%)', fontsize=12)
    plt.xticks(cache_sizes)
    plt.legend(title='Dataset', fontsize=10, title_fontsize=12)
    plt.ylim(0, 100)

    # Save plot
    plt.savefig(os.path.join(output_dir, f"lineplot_{policy}.png"))
    plt.close()

# =========================================
# 5. Per-Policy Bar Plots Across Datasets
# =========================================
for policy in policies:
    plt.figure(figsize=(10, 6))
    subset = df[df['Policy'] == policy]
    sns.barplot(data=subset, x='Cache Size', y='Hit Ratio (%)',
                hue='Dataset', palette="tab10")
    plt.title(f'Hit Ratio of {policy} by Dataset and Cache Size', fontsize=14)
    plt.xlabel('Cache Size', fontsize=12)
    plt.ylabel('Hit Ratio (%)', fontsize=12)
    plt.legend(title='Dataset', fontsize=10, title_fontsize=12)
    plt.ylim(0, 100)

    # Save plot
    plt.savefig(os.path.join(output_dir, f"barplot_{policy}.png"))
    plt.close()

print(f"All plots have been saved in the '{output_dir}' directory.")
