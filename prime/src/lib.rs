use pyo3::prelude::*;

#[pyfunction]
fn get_primes(limit: usize) -> PyResult<Vec<usize>> {
    if limit < 2 {
        return Ok(Vec::new());
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
    
    Ok((2..=limit)
        .filter(|&i| primes[i])
        .collect())
}

/// A Python module implemented in Rust.
#[pymodule]
fn prime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_primes, m)?)?;
    Ok(())
}
