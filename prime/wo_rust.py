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

