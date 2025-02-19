import numpy as np

def get_primes(limit: int) -> np.ndarray:
    if limit < 2:
        return np.array([], dtype=int)
    
    # Initialize boolean array with True values
    primes = np.ones(limit + 1, dtype=bool)
    primes[0] = primes[1] = False
    
    for i in range(2, int(np.sqrt(limit)) + 1):
        if primes[i]:
            # highly optimized numpy operation
            primes[i * i::i] = False
    
    # Return numpy array directly instead of converting to list
    return primes

def sum_primes(limit: int) -> int:
    # No need for conversion since get_primes now returns np.ndarray
    return np.sum(get_primes(limit))

def get_last_prime(limit: int) -> int:
    primes = get_primes(limit)
    return primes[-1] if len(primes) > 0 else 0


def fft_convolve(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Computes the linear convolution of 'signal' with 'kernel' using FFT-based convolution.
    
    Parameters
    ----------
    signal : np.ndarray
        The input signal (e.g., audio samples).
    kernel : np.ndarray
        The filter or impulse response to be applied to the signal.
    
    Returns
    -------
    np.ndarray
        The convolved result, with length = len(signal) + len(kernel) - 1
    """
    # Determine the size needed for zero-padding so we can do linear (non-circular) convolution
    conv_length = len(signal) + len(kernel) - 1
    # Next power of 2 for efficiency (optional, but often helps)
    fft_size = 1
    while fft_size < conv_length:
        fft_size <<= 1
    
    # Zero-pad the input arrays
    padded_signal = np.zeros(fft_size, dtype=np.complex64)
    padded_kernel = np.zeros(fft_size, dtype=np.complex64)
    padded_signal[:len(signal)] = signal
    padded_kernel[:len(kernel)] = kernel

    # Forward FFT
    fft_signal = np.fft.fft(padded_signal)
    fft_kernel = np.fft.fft(padded_kernel)

    # Multiply in frequency domain
    fft_product = fft_signal * fft_kernel

    # Inverse FFT
    convolved = np.fft.ifft(fft_product)

    convolved_real = np.real(convolved[:conv_length]).astype(np.float32)
    
    return convolved_real
