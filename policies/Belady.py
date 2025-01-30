import csv
import heapq
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate

INFINITY = float('inf')

class OptimalCache:
    """
    Optimal Cache Replacement Policy (Belady's Algorithm).
    """
    def __init__(self, capacity):
        """
        Initializes the Optimal Cache with a given capacity.

        :param capacity: Maximum number of items the cache can hold.
        """
        self.capacity = capacity
        self.cache = set()
        self.heap = []
        self.page_next_use = {}
        self.seen_pages = set()

    def is_hit(self, page):
        """
        Checks if a page is in the cache.

        :param page: The page offset to check.
        :return: True if hit, False otherwise.
        """
        return page in self.cache

    def access_page(self, page, next_use):
        """
        Accesses a page, updating the cache and its usage information.

        :param page: The page offset being accessed.
        :param next_use: The index of the next use of this page.
        :return: Tuple (hit: bool, evicted_page: Optional[int])
        """
        if page in self.cache:
            self.page_next_use[page] = next_use
            heapq.heappush(self.heap, (-next_use, page))
            return True, None
        else:
            evicted_page = None
            if len(self.cache) >= self.capacity:
                while self.heap:
                    farthest_neg_nu, farthest_page = heapq.heappop(self.heap)
                    farthest_nu = -farthest_neg_nu
                    if farthest_page in self.cache and self.page_next_use.get(farthest_page, -1) == farthest_nu:
                        self.cache.remove(farthest_page)
                        del self.page_next_use[farthest_page]
                        evicted_page = farthest_page
                        break
            self.cache.add(page)
            self.page_next_use[page] = next_use
            heapq.heappush(self.heap, (-next_use, page))
            return False, evicted_page

    def current_occupancy(self):
        """
        Returns the current number of items in the cache.

        :return: Integer count of items in cache.
        """
        return len(self.cache)

class CacheSimulator:
    """
    Simulator for Belady's Optimal Cache Replacement Algorithm.
    """
    def __init__(self, cache_size=10000):
        """
        Initializes the Cache Simulator with a given cache size.

        :param cache_size: Maximum number of items the cache can hold.
        """
        self.cache_size = cache_size

    def read_csv(self, filename, start_time=0, end_time=INFINITY):
        """
        Reads the relevant columns from a CSV file and filters requests based on start and end time.

        :param filename: Path to the CSV file.
        :param start_time: Start time for filtering requests.
        :param end_time: End time for filtering requests.
        :return: List of tuples containing (page_offset, request_type).
        """
        sequence = []
        with filename.open('r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) < 5:
                    continue
                try:
                    timestamp = float(row[0])
                    if timestamp < start_time or timestamp > end_time:
                        continue
                    page_offset = int(row[2])
                    request_type = row[4].strip().lower()
                    if request_type in {'read', 'write'}:
                        sequence.append((page_offset, request_type))
                except (ValueError, IndexError):
                    continue
        return sequence

    def precompute_next_use(self, sequence):
        """
        Precomputes the next use index for each page in the sequence.

        :param sequence: List of page requests as (page_offset, request_type).
        :return: List where next_use[i] is the index of the next request for the same page after i, or INFINITY.
        """
        N = len(sequence)
        next_use = [INFINITY] * N
        last_occurrence = {}
        for i in range(N - 1, -1, -1):
            page, _ = sequence[i]
            if page in last_occurrence:
                next_use[i] = last_occurrence[page]
            last_occurrence[page] = i
        return next_use

    def simulate(self, sequence, next_use, filename):
        """
        Simulates Belady's Optimal Cache Replacement Algorithm.

        :param sequence: List of page requests as (page_offset, request_type).
        :param next_use: Precomputed next use indices for each request.
        :param filename: The name of the file being processed (for progress bar).
        :return: Dictionary containing statistics on hits, misses, and other metrics.
        """
        cache = OptimalCache(self.cache_size)
        total_requests = len(sequence)
        total_hits = 0
        total_misses = 0
        read_hits = 0
        read_misses = 0
        write_hits = 0
        write_misses = 0

        for i in tqdm(range(total_requests), desc=f"Processing {filename}"):
            page, req_type = sequence[i]
            nu = next_use[i]

            if cache.is_hit(page):
                total_hits += 1
                if req_type == 'read':
                    read_hits += 1
                else:
                    write_hits += 1
                cache.access_page(page, nu)
            else:
                total_misses += 1
                if req_type == 'read':
                    read_misses += 1
                else:
                    write_misses += 1
                cache.access_page(page, nu)

        hit_percentage = (total_hits / total_requests) * 100 if total_requests > 0 else 0
        read_total = read_hits + read_misses
        write_total = write_hits + write_misses
        read_hit_ratio = (read_hits / read_total) * 100 if read_total > 0 else 0
        write_hit_ratio = (write_hits / write_total) * 100 if write_total > 0 else 0

        stats = {
            'Read Requests': read_total,
            'Read Hits': read_hits,
            'Read Misses': read_misses,
            'Write Requests': write_total,
            'Write Hits': write_hits,
            'Write Misses': write_misses,
            'Total Requests': total_requests,
            'Total Hits': total_hits,
            'Total Misses': total_misses,
            'Hit Percentage': hit_percentage,
            'Read Hit Ratio': read_hit_ratio,
            'Write Hit Ratio': write_hit_ratio
        }

        return stats

    def display_results(self, stats):
        """
        Displays the simulation results in a tabular format.

        :param stats: Dictionary containing statistics to display.
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
            ["Total Misses", stats['Total Misses'], f"{(stats['Total Misses'] / stats['Total Requests'] * 100) if stats['Total Requests'] else 0:.2f}%"]
        ]

        headers = ["Metric", "Count", "Ratio"]

        print("\nSimulation Results:")
        print(tabulate(table, headers=headers, tablefmt="grid"))
        print("----------------------------")

def cache_simulator(filename, cache_size=10000, start_time=0, end_time=INFINITY):
    """
    Simulates the cache for a given CSV file.

    :param filename: Name of the CSV file (without extension).
    :param cache_size: Maximum number of items the cache can hold.
    :param start_time: Start time for filtering requests.
    :param end_time: End time for filtering requests.
    """
    script_dir = Path(__file__).parent
    file_path = script_dir / f"{filename}.csv"

    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        return

    simulator = CacheSimulator(cache_size=cache_size)
    sequence = simulator.read_csv(file_path, start_time, end_time)

    if not sequence:
        print("Error: No valid data found in the input file.")
        return

    next_use = simulator.precompute_next_use(sequence)
    stats = simulator.simulate(sequence, next_use, filename)
    simulator.display_results(stats)

def main():
    """
    Main function to execute the simulation for multiple CSV files.
    """
    filenames = ["A42", "A108", "A129", "A669"]
    cache_size = int(input("Enter cache size (default 10000): ") or 10000)
    start_time = float(input("Enter start time (default 0): ") or 0)
    end_time = float(input("Enter end time (default inf): ") or float('inf'))

    for fname in filenames:
        cache_simulator(fname, cache_size, start_time, end_time)

main()