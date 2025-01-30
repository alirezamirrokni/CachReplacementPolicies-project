import pandas as pd
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate

class LRUCache:
    """
    A simulator for an LRU cache with data filtering, statistics collection, and detailed result reporting.
    """

    def __init__(self, max_capacity):
        """
        Initializes the LRUCache.

        :param max_capacity: Maximum number of items the cache can hold.
        """
        self.max_capacity = max_capacity
        self.cache_store = OrderedDict()

    def load_and_filter_data(self, file_path, start_time=0, end_time=float('inf')) -> pd.DataFrame:
        """
        Loads data from a CSV file and filters based on the timestamp range.

        :param file_path: Path to the CSV file.
        :param start_time: Start timestamp for filtering.
        :param end_time: End timestamp for filtering.
        :return: pandas DataFrame containing filtered rows of the file.
        """
        raw_data = pd.read_csv(file_path)
        filtered_data = raw_data[(raw_data.iloc[:, 0] >= start_time) & (raw_data.iloc[:, 0] <= end_time)]
        return filtered_data

    def access_or_update_cache(self, item):
        """
        Accesses or updates the cache. If the item exists, it refreshes its position.
        If the item doesn't exist, it adds it to the cache and evicts the least recently used item if necessary.

        :param item: The item to access or update in the cache.
        :return: True if the item was already in the cache, False otherwise.
        """
        if item in self.cache_store:
            self.cache_store.move_to_end(item)
            return True
        if len(self.cache_store) >= self.max_capacity:
            self.cache_store.popitem(last=False)
        self.cache_store[item] = True
        return False

    def simulate_lru_policy(self, dataset, filename):
        """
        Simulates the LRU cache policy with the provided dataset.

        :param dataset: pandas DataFrame containing the data to process.
        :param filename: The name of the file being processed (used for result indication).
        """
        read_requests, read_misses = 0, 0
        write_requests, write_misses = 0, 0
        offsets = dataset.iloc[:, 2].to_numpy()
        operations = dataset.iloc[:, 4].to_numpy()

        for idx, offset in enumerate(tqdm(offsets, desc=f"Processing {filename}")):
            operation_type = operations[idx]
            if operation_type == 'Read':
                read_requests += 1
                if not self.access_or_update_cache(offset):
                    read_misses += 1
            else:
                write_requests += 1
                if not self.access_or_update_cache(offset):
                    write_misses += 1

        stats = self.collect_statistics(read_requests, read_misses, write_requests, write_misses)
        self.display_results(stats, filename)

    @staticmethod
    def collect_statistics(reads, read_misses, writes, write_misses):
        """
        Collects and calculates cache statistics.

        :param reads: Total read requests.
        :param read_misses: Total read misses.
        :param writes: Total write requests.
        :param write_misses: Total write misses.
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
            'Hit Percentage': hit_percentage,
            'Read Hit Ratio': read_hit_ratio,
            'Write Hit Ratio': write_hit_ratio,
        }

    @staticmethod
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
        ]

        headers = ["Metric", "Count", "Ratio"]

        print(f"\nSimulation Results:")
        print(tabulate(table, headers=headers, tablefmt="grid"))
        print("----------------------------")


def cache_simulator(filenames, cache_size=10000, start_time=0, end_time=float('inf')):
    """
    Simulates the LRU cache for a list of CSV files.

    :param filenames: List of CSV file names (without extensions).
    :param cache_size: Maximum number of items the cache can hold (default 10000).
    :param start_time: Start timestamp for filtering (default 0).
    :param end_time: End timestamp for filtering (default inf).
    """
    script_dir = Path(__file__).parent

    for file_name in filenames:
        file_path = script_dir / f"{file_name}.csv"

        if not file_path.exists():
            print(f"Error: File {file_path} does not exist.")
            continue

        simulator = LRUCache(cache_size)
        dataset = simulator.load_and_filter_data(file_path, start_time, end_time)
        simulator.simulate_lru_policy(dataset, file_path.stem)


def main():
    """
    Main function to execute the simulation for multiple CSV files.
    """
    filenames = ["A42", "A108", "A129", "A669"]
    cache_size = int(input("Enter cache size (default 10000): ") or 10000)
    start_time = float(input("Enter start time (default 0): ") or 0)
    end_time = float(input("Enter end time (default inf): ") or float('inf'))

    cache_simulator(filenames, cache_size, start_time, end_time)


main()