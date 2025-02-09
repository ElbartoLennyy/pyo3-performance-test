import time
import csv
import psutil
import os
import gc  # Add garbage collector import
from prime import get_primes as get_primes_rs
from prime import get_last_prime as get_last_prime_rs
from prime import sum_primes as sum_primes_rust
from prime import sum_array as sum_array_rust
from prime import fft_convolve as fft_convolve_rust
from prime import estimate_pi_rust 
from w_numpy import get_primes as get_primes_numpy, get_last_prime as get_last_prime_numpy, sum_primes as sum_primes_numpy, fft_convolve as fft_convolve_numpy
from wo_rust import get_primes as get_primes_wo_rust, get_last_prime as get_last_prime_wo_rust, sum_primes as sum_primes_wo_rust, estimate_pi_raw_multi_thread as estimate_pi_raw_multi_thread, estimate_pi_raw_single_thread as estimate_pi_raw_single_thread, estimate_pi_raw_multi_processing as estimate_pi_raw_multi_processing
from multiprocessing import freeze_support

import numpy as np
def sum_primes_rs_calling_in_python(limit: int) -> int:
    return sum_array_rust(get_primes_rs(limit))
    


def benchmark_function(func, n, name, iterations=5):
    times = []
    memory_usages = []
    output_length = None
    process = psutil.Process(os.getpid())
    
    for _ in range(iterations):
        # Force garbage collection before measurement
        gc.collect()
        
        # Get memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
        peak_memory = mem_before
        
        start = time.perf_counter()
        result = func(**n)
        end = time.perf_counter()
        
        # Get memory after and track peak
        mem_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
        peak_memory = max(peak_memory, mem_after)
        memory_usages.append(peak_memory - mem_before)
        
        times.append(end - start)
        # Determine output length
        if output_length is None:
            if hasattr(result, '__len__') and not isinstance(result, int):
                output_length = len(result)
            else:
                output_length = 1
    
    avg_time = sum(times) / len(times)
    avg_memory = sum(memory_usages) / len(memory_usages)
    print(f"{func.__name__} ({name})")
    print(f"  Time: {avg_time:.6f} seconds on average")
    print(f"  Peak Memory Usage: {avg_memory:.2f} MB on average")
    print(f"  Output length: {output_length}")
    return avg_time, avg_memory, output_length

# Define the range of n values you want to test.
# Running a full sequence from 1 to 10**8 would be extremely time-consuming.
# Here we pick a selection of sizes.
n_values = [2, 10, 100, 1000, 10**4,10**5, 10**6,10**7,10**8 ]
convolve_values = [[16,4],[16**2,4**2],[16**3,4**3],[16**4,4**4],[16**5,4**5],[16**6,4**6]]

# Prepare a CSV file to store the results
with open('benchmark_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Updated CSV Header
    writer.writerow(['Function_Name', 'Type', 'N', 'Average_Time', 'Average_Memory_MB', 'Output_Length'])

    if __name__ == '__main__':
        freeze_support()
        for n in n_values:

            # get_last_prime variants
            avg_time, avg_memory, out_len = benchmark_function(get_last_prime_rs, {'limit':n}, name="Rust_lastprime")
            writer.writerow([get_last_prime_rs.__name__, "Rust_lastprime", n, avg_time, avg_memory, out_len])

            avg_time, avg_memory, out_len = benchmark_function(get_last_prime_numpy, {'limit':n}, name="Numpy_lastprime")
            writer.writerow([get_last_prime_numpy.__name__, "Numpy_lastprime", n, avg_time, avg_memory, out_len])

            avg_time, avg_memory, out_len = benchmark_function(get_last_prime_wo_rust, {'limit':n}, name="Python_lastprime")
            writer.writerow([get_last_prime_wo_rust.__name__, "Python_lastprime", n, avg_time, avg_memory, out_len])

            # sum_primes variants
            avg_time, avg_memory, out_len = benchmark_function(sum_primes_rust, {'limit':n}, name="Rust_sumprimes")
            writer.writerow([sum_primes_rust.__name__, "Rust_sumprimes", n, avg_time, avg_memory, out_len])

            avg_time, avg_memory, out_len = benchmark_function(sum_primes_rs_calling_in_python, {'limit':n}, name="Rust_sumprimes_calling_in_python")
            writer.writerow([sum_primes_rs_calling_in_python.__name__, "Rust_sumprimes_calling_in_python", n, avg_time, avg_memory, out_len])

            avg_time, avg_memory, out_len = benchmark_function(sum_primes_numpy, {'limit':n}, name="Numpy_sumprimes")
            writer.writerow([sum_primes_numpy.__name__, "Numpy_sumprimes", n, avg_time, avg_memory, out_len])

            avg_time, avg_memory, out_len = benchmark_function(sum_primes_wo_rust, {'limit':n}, name="Python_sumprimes")
            writer.writerow([sum_primes_wo_rust.__name__, "Python_sumprimes", n, avg_time, avg_memory, out_len])


            # Rust return array vs only number 
            avg_time, avg_memory, out_len = benchmark_function(get_primes_rs, {'limit':n}, name="Rust_return_array")
            writer.writerow([get_primes_rs.__name__, "Rust_return_array", n, avg_time, avg_memory, out_len])

            avg_time, avg_memory, out_len = benchmark_function(get_last_prime_rs, {'limit':n}, name="Rust_return_number")
            writer.writerow([get_last_prime_rs.__name__, "Rust_return_number", n, avg_time, avg_memory, out_len])


            # Multithread calc pi

            avg_time, avg_memory, out_len = benchmark_function(estimate_pi_rust, {'limit':n, 'threads':4}, name="Rust_estimate_pi")
            writer.writerow([estimate_pi_rust.__name__, "Rust_estimate_pi", n, avg_time, avg_memory, out_len])

            avg_time, avg_memory, out_len = benchmark_function(estimate_pi_rust, {'limit':n, 'threads':1}, name="Rust_estimate_pi_single_thread")
            writer.writerow([estimate_pi_rust.__name__, "Rust_estimate_pi_single_thread", n, avg_time, avg_memory, out_len])

            avg_time, avg_memory, out_len = benchmark_function(estimate_pi_raw_multi_thread, {'limit':n}, name="Python_estimate_pi")
            writer.writerow([estimate_pi_raw_multi_thread.__name__, "Python_estimate_pi", n, avg_time, avg_memory, out_len])

            avg_time, avg_memory, out_len = benchmark_function(estimate_pi_raw_single_thread, {'limit':n}, name="Python_estimate_pi_single_thread")
            writer.writerow([estimate_pi_raw_single_thread.__name__, "Python_estimate_pi_single_thread", n, avg_time, avg_memory, out_len])

            avg_time, avg_memory, out_len = benchmark_function(estimate_pi_raw_multi_processing, {'limit':n}, name="Python_estimate_pi_multi_processing")
            writer.writerow([estimate_pi_raw_multi_processing.__name__, "Python_estimate_pi_multi_processing", n, avg_time, avg_memory, out_len])


        for conv_size in convolve_values:
            sig = np.random.rand(conv_size[0]).astype(np.float32)
            filt = np.random.rand(conv_size[1]).astype(np.float32)
            print(sig.shape, filt.shape)

            avg_time, avg_memory, out_len = benchmark_function(fft_convolve_rust,{'signal':sig,'kernel':filt} , name="Rust_fft_convolve")
            writer.writerow([fft_convolve_rust.__name__, "Rust_fft_convolve", conv_size[0], avg_time, avg_memory, out_len])

            avg_time, avg_memory, out_len = benchmark_function(fft_convolve_numpy,{'signal':sig,'kernel':filt} , name="Numpy_fft_convolve")
            writer.writerow([fft_convolve_numpy.__name__, "Numpy_fft_convolve", conv_size[0], avg_time, avg_memory, out_len])









    






