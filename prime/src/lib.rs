use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use rustfft::{num_complex::Complex32, FftPlanner};

fn generate_primes(limit: usize) -> Vec<usize> {
    if limit < 2 {
        return Vec::new();
    }

    let mut primes = vec![true; limit + 1];
    primes[0] = false;
    primes[1] = false;

    for i in 2..=(limit as f64).sqrt() as usize {
        if primes[i] {
            for j in (i * i..=limit).step_by(i) {
                primes[j] = false;
            }
        }
    }

    (2..=limit).filter(|&i| primes[i]).collect()
}
#[pyfunction]
fn get_primes(limit: usize) -> PyResult<Vec<usize>> {
    Ok(generate_primes(limit))
}

#[pyfunction]
fn get_last_prime(limit: usize) -> PyResult<usize> {
    Ok(generate_primes(limit).last().copied().unwrap_or(0))
}
#[pyfunction]
fn sum_primes(limit: usize) -> PyResult<usize> {
    Ok(generate_primes(limit).iter().sum())
}

#[pyfunction]
fn sum_array(arr: Vec<usize>) -> PyResult<usize> {
    Ok(arr.iter().sum())
}
#[pyfunction]
fn fft_convolve(signal: Vec<f32>, kernel: Vec<f32>) -> Vec<f32> {
    // Calculate needed size for linear convolution
    let conv_length = signal.len() + kernel.len() - 1;

    // Calculate next power of 2 for efficient FFT (optional, but typically beneficial)
    let mut fft_size = 1;
    while fft_size < conv_length {
        fft_size <<= 1;
    }

    // Prepare complex buffers with zero-padding
    let mut signal_buffer = vec![Complex32::new(0.0, 0.0); fft_size];
    let mut kernel_buffer = vec![Complex32::new(0.0, 0.0); fft_size];

    // Copy real data into complex buffer
    for (i, &val) in signal.iter().enumerate() {
        signal_buffer[i].re = val;
    }
    for (i, &val) in kernel.iter().enumerate() {
        kernel_buffer[i].re = val;
    }

    // Create forward & inverse FFT plans
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    // Perform forward FFT on both buffers
    fft.process(&mut signal_buffer);
    fft.process(&mut kernel_buffer);

    // Multiply the frequency-domain signals
    for i in 0..fft_size {
        let s = signal_buffer[i];
        let k = kernel_buffer[i];
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        signal_buffer[i] = Complex32::new(s.re * k.re - s.im * k.im, s.re * k.im + s.im * k.re);
    }

    // Inverse FFT to get back time-domain data
    ifft.process(&mut signal_buffer);

    // Convert the complex result to real and normalize by fft_size
    let mut output = vec![0.0_f32; conv_length];
    for i in 0..conv_length {
        output[i] = signal_buffer[i].re / (fft_size as f32);
    }

    output
}

#[pyfunction]
fn estimate_pi_rust(limit: usize, threads: usize) -> PyResult<f64> {
    // Create a temporary thread pool with 4 threads
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    // Run the computation within this pool
    let pi_estimate = pool.install(|| {
        let hits: usize = (0..limit)
            .into_par_iter()
            .map_init(
                || rand::rng(),
                |rng, _| {
                    let x: f64 = rng.random_range(-1.0..1.0);
                    let y: f64 = rng.random_range(-1.0..1.0);
                    if x * x + y * y <= 1.0 {
                        1
                    } else {
                        0
                    }
                },
            )
            .sum();
        4.0 * (hits as f64) / (limit as f64)
    });

    Ok(pi_estimate)
}

/// A Python module implemented in Rust.
#[pymodule]
fn prime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_primes, m)?)?;
    m.add_function(wrap_pyfunction!(get_last_prime, m)?)?;
    m.add_function(wrap_pyfunction!(sum_primes, m)?)?;
    m.add_function(wrap_pyfunction!(sum_array, m)?)?;
    m.add_function(wrap_pyfunction!(fft_convolve, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_pi_rust, m)?)?;
    Ok(())
}
