import random
import threading
import multiprocessing
from multiprocessing import Pool

def worker(n, index, results):
    hits = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x * x + y * y <= 1.0:
            hits += 1
    results[index] = hits

def get_primes(limit: int) -> list[int]:
    if limit < 2:
        return []
    
    # Initialize array of True values
    primes = [True] * (limit + 1)
    primes[0] = primes[1] = False
    
    # Sieve of Eratosthenes
    for i in range(2, int(limit ** 0.5) + 1):
        if primes[i]:
            for j in range(i * i, limit + 1, i):
                primes[j] = False
    
    # Return list of prime numbers
    return [i for i in range(2, limit + 1) if primes[i]]

def get_last_prime(limit: int) -> int:
    return get_primes(limit)[-1]

def get_len_of_array(arr: list[int]) -> int: 
    return len(arr)

def sum_primes(limit: int) -> int:
    return sum(get_primes(limit))

def estimate_pi_raw_multi_thread(limit, num_threads=4):
    """
    Estimate π using the Monte Carlo method with raw Python threads.
    
    Because this is CPU-bound, threads are limited by the GIL.
    """
    samples_per_thread = limit // num_threads
    results = [0] * num_threads
    threads = []
    
    def worker(n, index):
        hits = 0
        for _ in range(n):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x*x + y*y <= 1.0:
                hits += 1
        results[index] = hits

    for i in range(num_threads):
        # The last thread takes any extra samples.
        count = samples_per_thread if i < num_threads - 1 else limit - samples_per_thread * (num_threads - 1)
        t = threading.Thread(target=worker, args=(count, i))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    total_hits = sum(results)
    return 4.0 * total_hits / limit

def estimate_pi_raw_single_thread(limit):
    return estimate_pi_raw_multi_thread(limit, 1)

def _monte_carlo_worker(args):
    """Helper function for parallel Monte Carlo estimation"""
    n, seed = args
    random.seed(seed)
    hits = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x * x + y * y <= 1.0:
            hits += 1
    return hits

def estimate_pi_raw_multi_processing(limit, num_processes=4):
    """
    Estimate π using the Monte Carlo method with multiple processes.
    
    This method utilizes the multiprocessing module to perform CPU-bound operations in parallel.
    """
    samples_per_process = limit // num_processes
    
    # Create work items with different seeds
    work_items = []
    for i in range(num_processes):
        # The last process takes any extra samples
        count = samples_per_process if i < num_processes - 1 else limit - samples_per_process * (num_processes - 1)
        work_items.append((count, random.randint(0, 10000) + i))
    
    # Use Pool for better process management
    with Pool(processes=num_processes) as pool:
        results = pool.map(_monte_carlo_worker, work_items)
    
    total_hits = sum(results)
    return 4.0 * total_hits / limit

