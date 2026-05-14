use core::fmt;

/// Invalid probability input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProbabilityError {
    NotFinite { index: usize, value: f64 },
    OutOfRange { index: usize, value: f64 },
}

impl fmt::Display for ProbabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFinite { index, value } => {
                write!(f, "probability {index} is not finite: {value}")
            }
            Self::OutOfRange { index, value } => {
                write!(f, "probability {index} is outside [0, 1]: {value}")
            }
        }
    }
}

impl std::error::Error for ProbabilityError {}

/// Computes `product(1 - p_i)` in the log domain.
///
/// `ln_1p` retains small probabilities that would disappear in the direct
/// subtraction/product formulation. The log sum uses Neumaier compensation.
pub fn stable_survival_product(miss_probabilities: &[f64]) -> Result<f64, ProbabilityError> {
    validate(miss_probabilities)?;
    let (log_survival, certain_misses) = survival_log(miss_probabilities);
    if certain_misses > 0 {
        Ok(0.0)
    } else {
        Ok(log_survival.exp().clamp(0.0, 1.0))
    }
}

/// Probability that at least one member of an independent batch misses.
///
/// This is `1 - product(1 - p_i)`, evaluated as `-expm1(sum(log1p(-p_i)))`
/// so rare-event probabilities remain representable.
pub fn batch_miss_probability(miss_probabilities: &[f64]) -> Result<f64, ProbabilityError> {
    validate(miss_probabilities)?;
    let (log_survival, certain_misses) = survival_log(miss_probabilities);
    if certain_misses > 0 {
        Ok(1.0)
    } else {
        Ok((-log_survival.exp_m1()).clamp(0.0, 1.0))
    }
}

/// Shared-barrier head-of-line amplification versus independent execution.
///
/// Under a strict batch barrier, any miss delays all `n` members. The returned
/// ratio is `n * P(any miss) / sum(p_i)`. It approaches `n` for homogeneous
/// rare events. With no possible misses, the neutral convention is `1.0`.
pub fn hol_amplification_factor(miss_probabilities: &[f64]) -> Result<f64, ProbabilityError> {
    validate(miss_probabilities)?;
    if miss_probabilities.is_empty() {
        return Ok(1.0);
    }
    let sum = compensated_sum(miss_probabilities.iter().copied());
    if sum == 0.0 {
        return Ok(1.0);
    }
    let batch_miss = batch_miss_probability(miss_probabilities)?;
    Ok((miss_probabilities.len() as f64 * batch_miss / sum).max(1.0))
}

/// Expected count of individually healthy requests blocked by a late peer.
///
/// For member `i`, the contribution is
/// `(1-p_i) * (1-product_{j != i}(1-p_j))`. The implementation is O(n),
/// including the corner case of one or more certain misses.
pub fn expected_hol_victims(miss_probabilities: &[f64]) -> Result<f64, ProbabilityError> {
    validate(miss_probabilities)?;
    let (log_nonzero_survival, zero_count) = survival_log(miss_probabilities);
    let terms = miss_probabilities.iter().map(|&probability| {
        let survival = 1.0 - probability;
        let excluded_miss = match zero_count {
            0 => {
                // Do not form `1 - p` before taking the log: for sub-ULP
                // probabilities that subtraction rounds to one and fails to
                // exclude the current member from the peer-miss probability.
                let excluded_log = log_nonzero_survival - (-probability).ln_1p();
                (-excluded_log.exp_m1()).clamp(0.0, 1.0)
            }
            1 if survival == 0.0 => (-log_nonzero_survival.exp_m1()).clamp(0.0, 1.0),
            _ => 1.0,
        };
        survival * excluded_miss
    });
    Ok(compensated_sum(terms))
}

fn validate(probabilities: &[f64]) -> Result<(), ProbabilityError> {
    for (index, &value) in probabilities.iter().enumerate() {
        if !value.is_finite() {
            return Err(ProbabilityError::NotFinite { index, value });
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(ProbabilityError::OutOfRange { index, value });
        }
    }
    Ok(())
}

fn survival_log(probabilities: &[f64]) -> (f64, usize) {
    let mut certain_misses = 0;
    let logs = probabilities.iter().filter_map(|&probability| {
        if probability == 1.0 {
            certain_misses += 1;
            None
        } else {
            Some((-probability).ln_1p())
        }
    });
    (compensated_sum(logs), certain_misses)
}

fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let provisional = sum + value;
        if sum.abs() >= value.abs() {
            correction += (sum - provisional) + value;
        } else {
            correction += (value - provisional) + sum;
        }
        sum = provisional;
    }
    sum + correction
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64, tolerance: f64) {
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }

    #[test]
    fn batch_miss_matches_closed_form() {
        close(batch_miss_probability(&[0.1, 0.2]).unwrap(), 0.28, 1e-15);
        close(stable_survival_product(&[0.1, 0.2]).unwrap(), 0.72, 1e-15);
    }

    #[test]
    fn rare_events_do_not_cancel() {
        let miss = batch_miss_probability(&[1e-18, 1e-18, 1e-18]).unwrap();
        close(miss, 3e-18, 1e-32);
        assert!(miss > 0.0);
    }

    #[test]
    fn handles_certain_misses_and_empty_batches() {
        assert_eq!(batch_miss_probability(&[]).unwrap(), 0.0);
        assert_eq!(stable_survival_product(&[]).unwrap(), 1.0);
        assert_eq!(batch_miss_probability(&[0.2, 1.0]).unwrap(), 1.0);
        assert_eq!(stable_survival_product(&[0.2, 1.0]).unwrap(), 0.0);
        assert_eq!(hol_amplification_factor(&[]).unwrap(), 1.0);
    }

    #[test]
    fn computes_hol_metrics() {
        // Any miss probability is .19, expected independent misses is .2.
        close(hol_amplification_factor(&[0.1, 0.1]).unwrap(), 1.9, 1e-14);
        // Either healthy member is a victim exactly when its peer misses.
        close(expected_hol_victims(&[0.1, 0.1]).unwrap(), 0.18, 1e-14);
        close(expected_hol_victims(&[1.0, 0.2]).unwrap(), 0.8, 1e-14);
        close(expected_hol_victims(&[1.0]).unwrap(), 0.0, 1e-14);
    }

    #[test]
    fn expected_victims_excludes_sub_ulp_probabilities() {
        close(expected_hol_victims(&[1e-18]).unwrap(), 0.0, 0.0);
        close(expected_hol_victims(&[1e-18, 1e-18]).unwrap(), 2e-18, 1e-32);
    }

    #[test]
    fn rejects_invalid_probabilities() {
        assert!(matches!(
            batch_miss_probability(&[f64::NAN]),
            Err(ProbabilityError::NotFinite { index: 0, .. })
        ));
        assert!(matches!(
            batch_miss_probability(&[-0.1]),
            Err(ProbabilityError::OutOfRange { index: 0, .. })
        ));
        assert!(matches!(
            batch_miss_probability(&[1.1]),
            Err(ProbabilityError::OutOfRange { index: 0, .. })
        ));
    }
}
