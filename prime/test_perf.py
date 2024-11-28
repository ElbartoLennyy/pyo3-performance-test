import time
from prime import get_primes as get_primes_rs
from w_numpy import get_primes as get_primes_numpy
from wo_rust import get_primes as get_primes_wo_rust

def benchmark_function(func, *args, iterations=5):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    avg_time = sum(times) / len(times)
    print(f"{func.__name__} took {avg_time:.6f} seconds on average over {iterations} runs.")
    return avg_time

# Example usage
n = 10**7  # Define the range for prime calculations
benchmark_function(get_primes_rs, n,)
benchmark_function(get_primes_numpy, n)
benchmark_function(get_primes_wo_rust, n)


