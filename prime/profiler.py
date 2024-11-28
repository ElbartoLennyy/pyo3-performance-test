
from prime import get_primes as get_primes_rs
from w_numpy import get_primes as get_primes_numpy
from wo_rust import get_primes as get_primes_wo_rust


from line_profiler import LineProfiler

def profile_line(func, *args):
    profiler = LineProfiler()
    profiler.add_function(func)  # Profile the specific function
    profiler(func)(*args)  # Run the function with arguments
    profiler.print_stats()

# Example usage
n = 10**6
# profile_line(get_primes_rs, n)
profile_line(get_primes_numpy, n)
profile_line(get_primes_wo_rust, n)
