use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

#[pyfunction]
fn simulate_heston_terminal_returns(
    mu: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    v0: f64,
    paths: usize,
    horizon_days: usize,
    seed: u64,
) -> PyResult<Vec<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dt = 1.0 / 252.0;
    let rho_clamped = rho.clamp(-0.98, 0.98);
    let orth_scale = (1.0 - rho_clamped * rho_clamped).sqrt();
    let mut out = Vec::with_capacity(paths);

    for _ in 0..paths {
        let mut s_rel = 1.0;
        let mut variance = v0.max(1e-10);
        for _ in 0..horizon_days {
            let z_price: f64 = StandardNormal.sample(&mut rng);
            let z_extra: f64 = StandardNormal.sample(&mut rng);
            let z_var = rho_clamped * z_price + orth_scale * z_extra;
            let v_pos = variance.max(0.0);
            variance = variance
                + kappa.max(1e-4) * (theta.max(1e-8) - v_pos) * dt
                + xi.max(1e-4) * (v_pos * dt).sqrt() * z_var;
            variance = variance.max(1e-10);
            let log_return = (mu - 0.5 * v_pos) * dt + (v_pos * dt).sqrt() * z_price;
            s_rel *= log_return.exp();
        }
        out.push(s_rel - 1.0);
    }
    Ok(out)
}

#[pyfunction]
fn backend_info() -> PyResult<String> {
    Ok(format!(
        "heston_var_rust 0.1.0; backend=rayon-parallel; rayon_threads={}",
        rayon::current_num_threads()
    ))
}

#[pyfunction]
fn simulate_heston_portfolio_pnl(
    mu: Vec<f64>,
    kappa: Vec<f64>,
    theta: Vec<f64>,
    xi: Vec<f64>,
    rho: Vec<f64>,
    v0: Vec<f64>,
    weights: Vec<f64>,
    correlation_flat: Vec<f64>,
    capital: f64,
    paths: usize,
    horizon_days: usize,
    seed: u64,
    antithetic: bool,
) -> PyResult<Vec<f64>> {
    let assets = weights.len();
    validate_portfolio_inputs(
        assets,
        &mu,
        &kappa,
        &theta,
        &xi,
        &rho,
        &v0,
        &correlation_flat,
        paths,
        horizon_days,
    )?;

    let lower = cholesky(&correlation_flat, assets)?;
    let dt = 1.0 / 252.0;
    let mut pnl = vec![0.0; paths];
    let threads = rayon::current_num_threads().max(1);
    let mut chunk_size = ((paths + threads - 1) / threads).max(1);
    if antithetic && chunk_size % 2 == 1 {
        chunk_size += 1;
    }

    pnl.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(
                seed.wrapping_add((chunk_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let mut independent = vec![0.0; assets];
            let mut z_price = vec![0.0; assets];
            let mut s_rel = vec![1.0; assets];
            let mut variance = vec![0.0; assets];
            let mut s_rel_anti = vec![1.0; assets];
            let mut variance_anti = vec![0.0; assets];

            if !antithetic {
                for output in chunk.iter_mut() {
                    s_rel.fill(1.0);
                    for i in 0..assets {
                        variance[i] = v0[i].max(1e-10);
                    }

                    for _ in 0..horizon_days {
                        for item in independent.iter_mut() {
                            *item = StandardNormal.sample(&mut rng);
                        }

                        for i in 0..assets {
                            let mut sum = 0.0;
                            for j in 0..=i {
                                sum += lower[i * assets + j] * independent[j];
                            }
                            z_price[i] = sum;
                        }

                        for i in 0..assets {
                            let z_extra: f64 = StandardNormal.sample(&mut rng);
                            let rho_i = rho[i].clamp(-0.98, 0.98);
                            let z_var = rho_i * z_price[i] + (1.0 - rho_i * rho_i).sqrt() * z_extra;
                            let v_pos = variance[i].max(0.0);
                            variance[i] = variance[i]
                                + kappa[i].max(1e-4) * (theta[i].max(1e-8) - v_pos) * dt
                                + xi[i].max(1e-4) * (v_pos * dt).sqrt() * z_var;
                            variance[i] = variance[i].max(1e-10);

                            let log_return =
                                (mu[i] - 0.5 * v_pos) * dt + (v_pos * dt).sqrt() * z_price[i];
                            s_rel[i] *= log_return.exp();
                        }
                    }

                    let mut portfolio_return = 0.0;
                    for i in 0..assets {
                        portfolio_return += weights[i] * (s_rel[i] - 1.0);
                    }
                    *output = portfolio_return * capital;
                }
            } else {
                let mut output_index = 0;
                while output_index < chunk.len() {
                    s_rel.fill(1.0);
                    s_rel_anti.fill(1.0);
                    for i in 0..assets {
                        variance[i] = v0[i].max(1e-10);
                        variance_anti[i] = v0[i].max(1e-10);
                    }

                    for _ in 0..horizon_days {
                        for item in independent.iter_mut() {
                            *item = StandardNormal.sample(&mut rng);
                        }

                        for i in 0..assets {
                            let mut sum = 0.0;
                            for j in 0..=i {
                                sum += lower[i * assets + j] * independent[j];
                            }
                            z_price[i] = sum;
                        }

                        for i in 0..assets {
                            let z_extra: f64 = StandardNormal.sample(&mut rng);
                            let rho_i = rho[i].clamp(-0.98, 0.98);
                            let z_var =
                                rho_i * z_price[i] + (1.0 - rho_i * rho_i).sqrt() * z_extra;

                            let v_pos = variance[i].max(0.0);
                            variance[i] = variance[i]
                                + kappa[i].max(1e-4) * (theta[i].max(1e-8) - v_pos) * dt
                                + xi[i].max(1e-4) * (v_pos * dt).sqrt() * z_var;
                            variance[i] = variance[i].max(1e-10);

                            let log_return =
                                (mu[i] - 0.5 * v_pos) * dt + (v_pos * dt).sqrt() * z_price[i];
                            s_rel[i] *= log_return.exp();

                            let v_anti = variance_anti[i].max(0.0);
                            variance_anti[i] = variance_anti[i]
                                + kappa[i].max(1e-4) * (theta[i].max(1e-8) - v_anti) * dt
                                - xi[i].max(1e-4) * (v_anti * dt).sqrt() * z_var;
                            variance_anti[i] = variance_anti[i].max(1e-10);

                            let log_return_anti =
                                (mu[i] - 0.5 * v_anti) * dt - (v_anti * dt).sqrt() * z_price[i];
                            s_rel_anti[i] *= log_return_anti.exp();
                        }
                    }

                    let mut portfolio_return = 0.0;
                    let mut portfolio_return_anti = 0.0;
                    for i in 0..assets {
                        portfolio_return += weights[i] * (s_rel[i] - 1.0);
                        portfolio_return_anti += weights[i] * (s_rel_anti[i] - 1.0);
                    }
                    chunk[output_index] = portfolio_return * capital;
                    if output_index + 1 < chunk.len() {
                        chunk[output_index + 1] = portfolio_return_anti * capital;
                    }
                    output_index += 2;
                }
            }
        });

    Ok(pnl)
}

fn validate_portfolio_inputs(
    assets: usize,
    mu: &[f64],
    kappa: &[f64],
    theta: &[f64],
    xi: &[f64],
    rho: &[f64],
    v0: &[f64],
    correlation_flat: &[f64],
    paths: usize,
    horizon_days: usize,
) -> PyResult<()> {
    if assets == 0 {
        return Err(PyValueError::new_err("At least one asset is required."));
    }
    if paths == 0 {
        return Err(PyValueError::new_err("paths must be positive."));
    }
    if horizon_days == 0 {
        return Err(PyValueError::new_err("horizon_days must be positive."));
    }
    let lengths = [mu.len(), kappa.len(), theta.len(), xi.len(), rho.len(), v0.len()];
    if lengths.iter().any(|length| *length != assets) {
        return Err(PyValueError::new_err(
            "Parameter vectors must have the same length as weights.",
        ));
    }
    if correlation_flat.len() != assets * assets {
        return Err(PyValueError::new_err(
            "correlation_flat must contain assets * assets values.",
        ));
    }
    Ok(())
}

fn cholesky(matrix: &[f64], n: usize) -> PyResult<Vec<f64>> {
    let mut lower = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= lower[i * n + k] * lower[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(PyValueError::new_err(
                        "Correlation matrix must be positive definite.",
                    ));
                }
                lower[i * n + j] = sum.sqrt();
            } else {
                lower[i * n + j] = sum / lower[j * n + j];
            }
        }
    }
    Ok(lower)
}

#[pymodule]
fn heston_var_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(backend_info, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_heston_terminal_returns, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_heston_portfolio_pnl, m)?)?;
    Ok(())
}
