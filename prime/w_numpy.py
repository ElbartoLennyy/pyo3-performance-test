import numpy as np

def get_primes(limit: int) -> list[int]:
    if limit < 2:
        return []
    
    # Initialize boolean array with True values
    primes = np.ones(limit + 1, dtype=bool)
    primes[0] = primes[1] = False
    
    # Sieve of Eratosthenes using NumPy's vectorized operations
    for i in range(2, int(np.sqrt(limit)) + 1):
        if primes[i]:
            primes[i * i::i] = False
    
    # Return list of prime numbers using NumPy's nonzero
    return list(np.nonzero(primes)[0])