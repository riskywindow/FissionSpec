use core::fmt;

/// One calibrated point on a batch-work/latency curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatencyPoint {
    /// Aggregate work units (tokens, pages, or normalized sequence slots).
    pub units: u32,
    /// End-to-end service time at `units`.
    pub latency_ns: u64,
}

impl LatencyPoint {
    /// Creates a calibration point.
    #[must_use]
    pub const fn new(units: u32, latency_ns: u64) -> Self {
        Self { units, latency_ns }
    }
}

/// Why a latency profile could not be constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileError {
    Empty,
    ZeroUnits { index: usize },
    UnitsNotStrictlyIncreasing { index: usize },
    LatencyDecreased { index: usize },
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("latency profile has no points"),
            Self::ZeroUnits { index } => write!(f, "point {index} has zero work units"),
            Self::UnitsNotStrictlyIncreasing { index } => {
                write!(f, "work units stop increasing at point {index}")
            }
            Self::LatencyDecreased { index } => {
                write!(f, "latency decreases at point {index}")
            }
        }
    }
}

impl std::error::Error for ProfileError {}

/// A validated monotone piecewise-linear latency curve.
///
/// Interpolation rounds upward, making deadline admission conservative. Values
/// below the first knot are clamped to it; values above the final knot use the
/// final segment's slope (or remain constant for a one-point profile).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatencyProfile {
    points: Box<[LatencyPoint]>,
}

impl LatencyProfile {
    /// Validates and owns a set of calibration points.
    pub fn new(points: Vec<LatencyPoint>) -> Result<Self, ProfileError> {
        if points.is_empty() {
            return Err(ProfileError::Empty);
        }
        for (index, point) in points.iter().enumerate() {
            if point.units == 0 {
                return Err(ProfileError::ZeroUnits { index });
            }
            if index > 0 {
                let previous = points[index - 1];
                if point.units <= previous.units {
                    return Err(ProfileError::UnitsNotStrictlyIncreasing { index });
                }
                if point.latency_ns < previous.latency_ns {
                    return Err(ProfileError::LatencyDecreased { index });
                }
            }
        }
        Ok(Self {
            points: points.into_boxed_slice(),
        })
    }

    /// Returns the immutable calibration knots.
    #[must_use]
    pub fn points(&self) -> &[LatencyPoint] {
        &self.points
    }

    /// Predicts service latency for aggregate batch work.
    ///
    /// Empty work has zero latency. Arithmetic uses a wide intermediate and
    /// saturates at `u64::MAX` instead of wrapping.
    #[must_use]
    pub fn predict_ns(&self, units: u32) -> u64 {
        if units == 0 {
            return 0;
        }
        let first = self.points[0];
        if units <= first.units {
            return first.latency_ns;
        }

        let upper = self.points.partition_point(|point| point.units < units);
        if upper < self.points.len() {
            return interpolate_ceil(self.points[upper - 1], self.points[upper], units);
        }

        if self.points.len() == 1 {
            return first.latency_ns;
        }
        let last = self.points[self.points.len() - 1];
        let previous = self.points[self.points.len() - 2];
        interpolate_ceil(previous, last, units)
    }
}

fn interpolate_ceil(left: LatencyPoint, right: LatencyPoint, units: u32) -> u64 {
    let run = u128::from(right.units - left.units);
    let rise = u128::from(right.latency_ns - left.latency_ns);
    let offset = u128::from(units - left.units);
    let increment = (offset * rise).div_ceil(run);
    u128::from(left.latency_ns)
        .saturating_add(increment)
        .min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_monotonicity() {
        assert_eq!(LatencyProfile::new(vec![]), Err(ProfileError::Empty));
        assert_eq!(
            LatencyProfile::new(vec![LatencyPoint::new(0, 10)]),
            Err(ProfileError::ZeroUnits { index: 0 })
        );
        assert_eq!(
            LatencyProfile::new(vec![LatencyPoint::new(1, 10), LatencyPoint::new(1, 11)]),
            Err(ProfileError::UnitsNotStrictlyIncreasing { index: 1 })
        );
        assert_eq!(
            LatencyProfile::new(vec![LatencyPoint::new(1, 10), LatencyPoint::new(2, 9)]),
            Err(ProfileError::LatencyDecreased { index: 1 })
        );
    }

    #[test]
    fn interpolates_upward_and_extrapolates_terminal_slope() {
        let profile = LatencyProfile::new(vec![
            LatencyPoint::new(2, 10),
            LatencyPoint::new(5, 20),
            LatencyPoint::new(7, 22),
        ])
        .unwrap();

        assert_eq!(profile.predict_ns(0), 0);
        assert_eq!(profile.predict_ns(1), 10);
        assert_eq!(profile.predict_ns(2), 10);
        assert_eq!(profile.predict_ns(3), 14); // ceil(10 + 10/3)
        assert_eq!(profile.predict_ns(5), 20);
        assert_eq!(profile.predict_ns(6), 21);
        assert_eq!(profile.predict_ns(9), 24);
    }

    #[test]
    fn one_point_profile_is_constant() {
        let profile = LatencyProfile::new(vec![LatencyPoint::new(1, 42)]).unwrap();
        assert_eq!(profile.predict_ns(1), 42);
        assert_eq!(profile.predict_ns(u32::MAX), 42);
    }

    #[test]
    fn extrapolation_saturates() {
        let profile = LatencyProfile::new(vec![
            LatencyPoint::new(1, 0),
            LatencyPoint::new(2, u64::MAX),
        ])
        .unwrap();
        assert_eq!(profile.predict_ns(3), u64::MAX);
    }
}
