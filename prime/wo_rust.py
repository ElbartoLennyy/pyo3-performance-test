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