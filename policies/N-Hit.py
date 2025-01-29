import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from sortedcontainers import SortedList

class NHitCache:
    """
    A simulator for the NHit cache policy with eviction based on NHit counts and tracking.
    """

    def __init__(self, capacity, trigger_threshold=80.0, insertion_threshold=2):
        """
        Initializes the NHitCache.

        :param capacity: Maximum number of items the cache can hold.
        :param trigger_threshold: Cache occupancy percentage to trigger tracking.
        :param insertion_threshold: Number of accesses required before promotion.
        """
        self.capacity = capacity
        self.trigger_threshold = trigger_threshold
        self.insertion_threshold = insertion_threshold
        self.cache = {}
        self.tracking = defaultdict(int)
        self.sorted_items = SortedList()
        self.insertion_counter = 0

    def _evict(self):
        """
        Evicts the block with the lowest NHit from the cache (oldest if there's a tie).
        """
        if len(self.sorted_items) == 0:
            return
        victim_nhit, victim_insertion_counter, victim_item = self.sorted_items.pop(0)
        del self.cache[victim_item]

    def access(self, item):
        """
        Accesses an item, updating its NHit count in tracking.

        :param item: The item being accessed.
        """
        self.tracking[item] += 1

    def promote(self, item):
        """
        Promotes an item to the cache if it meets the promotion criteria.

        :param item: The item to promote.
        """
        if len(self.cache) >= self.capacity:
            self._evict()
        nhit = self.tracking[item]
        self.cache[item] = nhit
        self.insertion_counter += 1
        self.sorted_items.add((nhit, self.insertion_counter, item))

    def should_promote(self, item):
        """
        Determines whether an item should be promoted to the cache.

        :param item: The item to evaluate.
        :return: True if the item should be promoted, False otherwise.
        """
        occupancy_percent = 100.0 * len(self.cache) / self.capacity
        if occupancy_percent <= self.trigger_threshold:
            return True
        return self.tracking[item] >= self.insertion_threshold


def simulate_nhit(file_path, cache_size=10000, trigger_threshold=80.0, insertion_threshold=2):
    """
    Simulates the NHit promotion policy.

    :param file_path: Path to the input CSV file.
    :param cache_size: Maximum number of items the cache can hold.
    :param trigger_threshold: Cache occupancy percentage to trigger tracking.
    :param insertion_threshold: Number of accesses required before promotion.
    """
    nhit_cache = NHitCache(cache_size, trigger_threshold, insertion_threshold)

    try:
        data_frame = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if data_frame.shape[1] < 5:
        print(f"Error: {file_path} does not have at least 5 columns.")
        return

    total_requests = 0
    read_hits, read_misses = 0, 0
    write_hits, write_misses = 0, 0
    cold_misses = 0
    tracked_access = set()
    offsets = data_frame.iloc[:, 2].to_numpy()
    operations = data_frame.iloc[:, 4].to_numpy()

    for offset, operation in tqdm(zip(offsets, operations), total=len(offsets), desc=f"Processing {file_path.stem}"):
        total_requests += 1
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
        read_hits + read_misses, read_misses,
        write_hits + write_misses, write_misses,
        cold_misses
    )
    display_results(stats, file_path.stem)


def collect_statistics(reads, read_misses, writes, write_misses, cold_misses):
    """
    Collects and calculates cache statistics.

    :param reads: Total read requests.
    :param read_misses: Total read misses.
    :param writes: Total write requests.
    :param write_misses: Total write misses.
    :param cold_misses: Total cold misses.
    :return: A dictionary containing all calculated statistics.
    """
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


def display_results(stats, filename):
    """
    Displays the simulation results in a tabular format.

    :param stats: A dictionary containing simulation statistics.
    :param filename: The name of the file being processed.
    """
    table = [
        ["Read Requests", stats['Read Requests'], ""],
        ["Read Hits", stats['Read Hits'], f"{stats['Read Hit Ratio']:.2f}%"],
        ["Read Misses", stats['Read Misses'], f"{(stats['Read Misses'] / stats['Read Requests'] * 100) if stats['Read Requests'] else 0:.2f}%"],
        ["Write Requests", stats['Write Requests'], ""],
        ["Write Hits", stats['Write Hits'], f"{stats['Write Hit Ratio']:.2f}%"],
        ["Write Misses", stats['Write Misses'], f"{(stats['Write Misses'] / stats['Write Requests'] * 100) if stats['Write Requests'] else 0:.2f}%"],
        ["Total Requests", stats['Total Requests'], ""],
        ["Total Hits", stats['Total Hits'], f"{stats['Hit Percentage']:.2f}%"],
        ["Total Misses", stats['Total Misses'], f"{(stats['Total Misses'] / stats['Total Requests'] * 100) if stats['Total Requests'] else 0:.2f}%"],
        ["Cold Misses", stats['Cold Misses'], f"{(stats['Cold Misses'] / stats['Total Misses'] * 100) if stats['Total Misses'] else 0:.2f}%"],
    ]

    headers = ["Metric", "Count", "Ratio"]

    print(f"\nSimulation Results for {filename}:")
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print("----------------------------")


def main():
    """
    Main function to execute the simulation for multiple CSV files.
    """
    filenames = ["A42", "A108", "A129", "A669"]
    cache_size = 10000

    for file_name in filenames:
        file_path = Path(__file__).parent / f"{file_name}.csv"
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist.")
            continue
        simulate_nhit(file_path, cache_size)

main()
