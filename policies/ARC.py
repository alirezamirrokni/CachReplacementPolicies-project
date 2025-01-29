import csv
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate


class ARCCache:
    """
    A simulator for the ARC (Adaptive Replacement Cache) policy with detailed metrics reporting.
    """

    def __init__(self, cache_size):
        """
        Initializes the ARCCache simulator.

        :param cache_size: Maximum number of pages the cache can hold.
        """
        self.cache_size = cache_size
        self.T1 = OrderedDict()
        self.T2 = OrderedDict()
        self.B1 = OrderedDict()
        self.B2 = OrderedDict()
        self.p = 0

    def process_request(self, page):
        """
        Processes a single page request according to the ARC policy.

        :param page: Page number to be processed.
        :return: True if the page is a cache hit, False otherwise.
        """
        if page in self.T1 or page in self.T2:
            if page in self.T1:
                self.T1.pop(page)
                self.T2[page] = True
            else:
                self.T2.move_to_end(page)
            return True
        elif page in self.B1:
            self.p = min(self.cache_size, self.p + max(1, len(self.B2) // max(1, len(self.B1))))
            self.B1.pop(page)
            self.T2[page] = True
        elif page in self.B2:
            self.p = max(0, self.p - max(1, len(self.B1) // max(1, len(self.B2))))
            self.B2.pop(page)
            self.T2[page] = True
        else:
            self.T1[page] = True

        while len(self.T1) + len(self.T2) > self.cache_size:
            if len(self.T1) > self.p:
                if self.T1:
                    old, _ = self.T1.popitem(last=False)
                    self.B1[old] = True
            else:
                if self.T2:
                    old, _ = self.T2.popitem(last=False)
                    self.B2[old] = True

        while len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2) > 2 * self.cache_size:
            if len(self.B1) > self.p:
                if self.B1:
                    self.B1.popitem(last=False)
            else:
                if self.B2:
                    self.B2.popitem(last=False)

        return False

    def simulate(self, dataset, filename):
        """
        Simulates the ARC policy on the given dataset.

        :param dataset: List of tuples (page, operation type).
        :param filename: Name of the file being processed.
        :return: A dictionary containing statistics (hits and misses).
        """
        read_requests, write_requests = 0, 0
        read_hits, write_hits = 0, 0
        read_misses, write_misses = 0, 0

        for page, operation in tqdm(dataset, desc=f"Processing {filename}", leave=True):
            if operation == "Read":
                read_requests += 1
                if self.process_request(page):
                    read_hits += 1
                else:
                    read_misses += 1
            elif operation == "Write":
                write_requests += 1
                if self.process_request(page):
                    write_hits += 1
                else:
                    write_misses += 1

        return {
            "Read Requests": read_requests,
            "Read Hits": read_hits,
            "Read Misses": read_misses,
            "Write Requests": write_requests,
            "Write Hits": write_hits,
            "Write Misses": write_misses,
            "Total Requests": read_requests + write_requests,
            "Total Hits": read_hits + write_hits,
            "Total Misses": read_misses + write_misses,
            "Read Hit Ratio": read_hits / read_requests * 100 if read_requests > 0 else 0,
            "Write Hit Ratio": write_hits / write_requests * 100 if write_requests > 0 else 0,
            "Overall Hit Ratio": (read_hits + write_hits) / (read_requests + write_requests) * 100 if (read_requests + write_requests) > 0 else 0
        }


def load_sequence(file_path):
    """
    Loads the sequence of page requests from a CSV file.

    :param file_path: Path to the CSV file.
    :return: List of tuples (page number, operation type).
    """
    sequence = []
    try:
        with file_path.open('r') as file:
            reader = csv.reader(file)
            for row in reader:
                try:
                    page = int(float(row[2])) // 4096
                    operation = row[4].strip()
                    if operation in {"Read", "Write"}:
                        sequence.append((page, operation))
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    return sequence


def display_results(stats, filename):
    """
    Displays simulation results in a tabular format.

    :param stats: A dictionary containing simulation statistics.
    :param filename: The name of the file being processed.
    """
    table = [
        ["Read Requests", stats["Read Requests"], ""],
        ["Read Hits", stats["Read Hits"], f"{stats['Read Hit Ratio']:.2f}%"],
        ["Read Misses", stats["Read Misses"], f"{100 - stats['Read Hit Ratio']:.2f}%"],
        ["Write Requests", stats["Write Requests"], ""],
        ["Write Hits", stats["Write Hits"], f"{stats['Write Hit Ratio']:.2f}%"],
        ["Write Misses", stats["Write Misses"], f"{100 - stats['Write Hit Ratio']:.2f}%"],
        ["Total Requests", stats["Total Requests"], ""],
        ["Total Hits", stats["Total Hits"], f"{stats['Overall Hit Ratio']:.2f}%"],
        ["Total Misses", stats["Total Misses"], f"{100 - stats['Overall Hit Ratio']:.2f}%"],
    ]
    headers = ["Metric", "Count", "Ratio"]
    print(f"\nSimulation Results for {filename}:")
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print("----------------------------")


def run_simulation(file_name, cache_size):
    """
    Runs the ARC simulation for a specific file.

    :param file_name: Name of the CSV file.
    :param cache_size: Maximum number of pages the cache can hold.
    """
    script_dir = Path(__file__).parent
    file_path = script_dir / file_name

    sequence = load_sequence(file_path)
    if not sequence:
        print(f"Error: No valid page references found in {file_name}.")
        return

    simulator = ARCCache(cache_size)
    stats = simulator.simulate(sequence, file_path.stem)
    display_results(stats, file_path.stem)


def main():
    """
    Main function to execute the ARC simulation for multiple files.
    """
    filenames = ["A42.csv", "A108.csv", "A129.csv", "A669.csv"]
    cache_size = 10000

    for file_name in filenames:
        run_simulation(file_name, cache_size)


main()
