import pandas as pd
from collections import OrderedDict, deque
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate


class LRUCache:
    """
    Least Recently Used (LRU) Cache Replacement Policy.
    """
    def __init__(self, capacity):
        """
        Initializes the LRU cache with a given capacity.

        :param capacity: Maximum number of items the cache can hold.
        """
        self.capacity = capacity
        self.cache = OrderedDict()

    def is_present(self, key):
        """
        Checks if a key is in the cache without updating its LRU order.

        :param key: Key to check in the cache.
        :return: True if the key is present, False otherwise.
        """
        return key in self.cache

    def access(self, key):
        """
        Accesses a key in the cache, updating its LRU order.

        :param key: Key to access in the cache.
        :return: True if the key is present, False otherwise.
        """
        if key not in self.cache:
            return False
        self.cache.move_to_end(key)
        return True

    def insert(self, key):
        """
        Inserts a key into the cache, evicting the least recently used entry if over capacity.

        :param key: Key to insert into the cache.
        :return: The evicted key if any, otherwise None.
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = True
        if len(self.cache) > self.capacity:
            evicted_key, _ = self.cache.popitem(last=False)
            return evicted_key
        return None

    def occupancy(self):
        """
        Returns the current number of items in the cache.

        :return: Integer count of items in the cache.
        """
        return len(self.cache)


class NHitPolicy:
    """
    N-HIT Cache Promotion Policy.
    """
    def __init__(self, cache_capacity, trigger_threshold=80.0, N=2, tracking_ratio=2):
        """
        Initializes the N-HIT policy with given parameters.

        :param cache_capacity: Maximum number of items the cache can hold.
        :param trigger_threshold: Cache occupancy percentage to trigger selective promotion.
        :param N: Number of accesses required for promotion.
        :param tracking_ratio: Ratio of tracking entries to cache capacity.
        """
        self.trigger_threshold = trigger_threshold
        self.N = N
        self.cache_capacity = cache_capacity
        self.max_tracked_items = int(tracking_ratio * cache_capacity)
        self.access_counts = {}
        self.tracking_queue = deque()

    def should_promote(self, item, is_cache_hit, current_cache_occupancy):
        """
        Determines whether an item should be promoted to the cache.

        :param item: Item to evaluate for promotion.
        :param is_cache_hit: Whether the item is already in the cache.
        :param current_cache_occupancy: Current number of items in the cache.
        :return: True if the item should be promoted, False otherwise.
        """
        occupancy_percent = 100.0 * current_cache_occupancy / self.cache_capacity
        if occupancy_percent <= self.trigger_threshold or is_cache_hit:
            return True
        return self.access_counts.get(item, 0) >= self.N

    def record_access(self, item):
        """
        Records an access to an item, incrementing its access count.

        :param item: Item being accessed.
        """
        if item in self.access_counts:
            self.access_counts[item] += 1
        else:
            if len(self.access_counts) >= self.max_tracked_items:
                oldest_item = self.tracking_queue.popleft()
                del self.access_counts[oldest_item]
            self.access_counts[item] = 1
            self.tracking_queue.append(item)

    def remove_from_tracking(self, item):
        """
        Removes an item from tracking after it's promoted to the cache.

        :param item: Item to remove from tracking.
        """
        if item in self.access_counts:
            del self.access_counts[item]
            try:
                self.tracking_queue.remove(item)
            except ValueError:
                pass


def simulate_nhit_lru(file_path, cache_size=10000, trigger_threshold=80.0, N=2, tracking_ratio=2):
    """
    Simulates a cache using N-HIT for promotion and LRU for eviction.

    :param file_path: Path to the input CSV file.
    :param cache_size: Maximum number of items the cache can hold.
    :param trigger_threshold: Cache occupancy percentage to trigger selective promotion.
    :param N: Number of accesses required for promotion.
    :param tracking_ratio: Ratio of tracking entries to cache capacity.
    """
    lru_cache = LRUCache(cache_size)
    nhit_policy = NHitPolicy(
        cache_capacity=cache_size,
        trigger_threshold=trigger_threshold,
        N=N,
        tracking_ratio=tracking_ratio
    )

    try:
        data_frame = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if data_frame.shape[1] < 5:
        print(f"Error: {file_path} does not have at least 5 columns.")
        return

    total_requests = 0
    total_misses = 0
    total_hits = 0
    read_requests = 0
    write_requests = 0
    read_hits = 0
    write_hits = 0

    for _, row in tqdm(data_frame.iterrows(), total=data_frame.shape[0], desc=f"Processing {file_path.stem}"):
        try:
            item = row.iloc[2]
            operation_type = row.iloc[4].strip().lower()
        except (IndexError, AttributeError):
            continue

        nhit_policy.record_access(item)
        is_hit = lru_cache.is_present(item)

        if operation_type == 'read':
            read_requests += 1
            if is_hit:
                read_hits += 1
                lru_cache.access(item)
            else:
                if nhit_policy.should_promote(item, is_hit, lru_cache.occupancy()):
                    lru_cache.insert(item)
                    nhit_policy.remove_from_tracking(item)
        elif operation_type == 'write':
            write_requests += 1
            if is_hit:
                write_hits += 1
                lru_cache.access(item)
            else:
                if nhit_policy.should_promote(item, is_hit, lru_cache.occupancy()):
                    lru_cache.insert(item)
                    nhit_policy.remove_from_tracking(item)

    total_requests = read_requests + write_requests
    total_hits = read_hits + write_hits
    total_misses = total_requests - total_hits

    table = [
        ["Read Requests", read_requests, ""],
        ["Read Hits", read_hits, f"{(read_hits / read_requests * 100) if read_requests else 0:.2f}%"],
        ["Write Requests", write_requests, ""],
        ["Write Hits", write_hits, f"{(write_hits / write_requests * 100) if write_requests else 0:.2f}%"],
        ["Total Requests", total_requests, ""],
        ["Total Hits", total_hits, f"{(total_hits / total_requests * 100) if total_requests else 0:.2f}%"],
        ["Total Misses", total_misses, f"{(total_misses / total_requests * 100) if total_requests else 0:.2f}%"]
    ]

    headers = ["Metric", "Count", "Ratio"]

    print("\nSimulation Results:")
    print(tabulate(table, headers=headers, tablefmt="grid"))


def main():
    """
    Main function to execute the simulation for multiple CSV files.
    """
    filenames = ["A42", "A108", "A129", "A669"]
    cache_size = 10000
    trigger_threshold = 80.0
    N = 2
    tracking_ratio = 2

    script_dir = Path(__file__).parent

    for fname in filenames:
        file_path = script_dir / f"{fname}.csv"
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist.")
            continue
        simulate_nhit_lru(
            file_path=file_path,
            cache_size=cache_size,
            trigger_threshold=trigger_threshold,
            N=N,
            tracking_ratio=tracking_ratio
        )

main()
