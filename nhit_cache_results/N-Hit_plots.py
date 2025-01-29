import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from sortedcontainers import SortedList
import matplotlib.pyplot as plt

def get_file_name():
    return input("Enter the filename (without extension): ")

class NHitCache:
    def __init__(self, capacity, trigger_threshold=80.0, insertion_threshold=2):
        self.capacity = capacity
        self.trigger_threshold = trigger_threshold
        self.insertion_threshold = insertion_threshold
        self.cache = {}
        self.tracking = defaultdict(int)
        self.sorted_items = SortedList()
        self.insertion_counter = 0

    def _evict(self):
        if len(self.sorted_items) == 0:
            return
        victim_nhit, victim_insertion_counter, victim_item = self.sorted_items.pop(0)
        del self.cache[victim_item]

    def access(self, item):
        self.tracking[item] += 1

    def promote(self, item):
        if len(self.cache) >= self.capacity:
            self._evict()
        nhit = self.tracking[item]
        self.cache[item] = nhit
        self.insertion_counter += 1
        self.sorted_items.add((nhit, self.insertion_counter, item))

    def should_promote(self, item):
        occupancy_percent = 100.0 * len(self.cache) / self.capacity
        if occupancy_percent <= self.trigger_threshold:
            return True
        return self.tracking[item] >= self.insertion_threshold

def collect_statistics(reads, read_misses, writes, write_misses, cold_misses):
    total_requests = reads + writes
    total_misses = read_misses + write_misses
    total_hits = total_requests - total_misses
    read_hits = reads - read_misses
    write_hits = writes - write_misses
    hit_percentage = (total_hits / total_requests * 100) if total_requests > 0 else 0
    read_hit_ratio = (read_hits / reads * 100) if reads > 0 else 0
    write_hit_ratio = (write_hits / writes * 100) if writes > 0 else 0

    return {
        'Read Requests': reads,
        'Read Hits': read_hits,
        'Read Misses': read_misses,
        'Write Requests': writes,
        'Write Hits': write_hits,
        'Write Misses': write_misses,
        'Total Requests': total_requests,
        'Total Hits': total_hits,
        'Total Misses': total_misses,
        'Cold Misses': cold_misses,
        'Hit Percentage': hit_percentage,
        'Read Hit Ratio': read_hit_ratio,
        'Write Hit Ratio': write_hit_ratio,
    }

def simulate_nhit(file_path, cache_size=10000, trigger_threshold=80.0, insertion_threshold=2):
    nhit_cache = NHitCache(cache_size, trigger_threshold, insertion_threshold)

    try:
        data_frame = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    if data_frame.shape[1] < 5:
        print(f"Error: {file_path} does not have at least 5 columns.")
        return None

    read_hits, read_misses = 0, 0
    write_hits, write_misses = 0, 0
    cold_misses = 0
    tracked_access = set()

    offsets = data_frame.iloc[:, 2].to_numpy()
    operations = data_frame.iloc[:, 4].to_numpy()

    for offset, operation in zip(offsets, operations):
        if offset in nhit_cache.cache:
            if operation == "Read":
                read_hits += 1
            else:
                write_hits += 1
        else:
            if operation == "Read":
                read_misses += 1
            else:
                write_misses += 1
            
            if offset not in tracked_access:
                cold_misses += 1
                tracked_access.add(offset)
            
            nhit_cache.access(offset)
            if nhit_cache.should_promote(offset):
                nhit_cache.promote(offset)

    stats = collect_statistics(
        read_hits + read_misses,
        read_misses,
        write_hits + write_misses,
        write_misses,
        cold_misses
    )
    return stats

def main():
    file_name = get_file_name()
    cache_size = 10000
    trigger_thresholds = [50, 60, 70, 80, 90]
    insertion_thresholds = [1, 2, 3, 4]
    
    results = {ins_thresh: [] for ins_thresh in insertion_thresholds}
    base_path = Path(__file__).parent
    file_path = base_path / f"{file_name}.csv"

    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        return

    for ins_thresh in tqdm(insertion_thresholds, desc=f"Processing {file_name}", leave=False):
        for trig_thresh in trigger_thresholds:
            stats = simulate_nhit(file_path, cache_size, trig_thresh, ins_thresh)
            results[ins_thresh].append(stats['Hit Percentage'] if stats else 0)

    fig, axes = plt.subplots(figsize=(6, 4))
    for ins_thresh, hit_ratios in results.items():
        plt.plot(trigger_thresholds, hit_ratios, marker='o', label=f"ins_thresh={ins_thresh}")

    plt.title(file_name)
    plt.xlabel("Trigger Threshold (%)")
    plt.ylabel("Total Hit Ratio (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    output_path = base_path / "nhit_cache_results.png"
    plt.savefig(output_path, format='png', dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()