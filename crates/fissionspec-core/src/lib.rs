//! Dependency-free reference primitives for the FissionSpec serving hot path.
//!
//! The crate deliberately separates cold-path calibration from hot-path decisions:
//! a [`LatencyProfile`] validates its knots once, while [`Horizon2Controller`]
//! evaluates a current microbatch and one forecast microbatch without allocation.
//!
//! ```
//! use fissionspec_core::{
//!     BatchView, ControllerConfig, ForecastBatch, Horizon2Controller, LatencyPoint,
//!     LatencyProfile, Priority, ReadyItem, ControllerAction,
//! };
//!
//! let profile = LatencyProfile::new(vec![
//!     LatencyPoint::new(1, 100),
//!     LatencyPoint::new(2, 115),
//! ]).unwrap();
//! let now = ReadyItem::new(1, 0, 1_000, 1, Priority::NORMAL).unwrap();
//! let next = ReadyItem::new(2, 10, 1_000, 1, Priority::NORMAL).unwrap();
//! let controller = Horizon2Controller::new(ControllerConfig::new(50, 8, 0).unwrap());
//!
//! let action = controller.decide(
//!     0,
//!     BatchView::ready_only(&[now]),
//!     Some(ForecastBatch::new(10, BatchView::ready_only(&[next]))),
//!     &profile,
//! );
//! assert_eq!(action, ControllerAction::WaitUntil { at_ns: 10 });
//! ```

#![forbid(unsafe_code)]

mod allocator;
mod analysis;
mod controller;
mod latency;
mod transaction;

pub use allocator::{AllocationError, PageAllocator, PageHandle};
pub use analysis::{
    batch_miss_probability, expected_hol_victims, hol_amplification_factor,
    stable_survival_product, ProbabilityError,
};
pub use controller::{
    BatchView, BypassReason, ControllerAction, ControllerConfig, ControllerConfigError, Decision,
    DecisionReason, ForecastBatch, Horizon2Controller, HorizonCosts, ItemClass, ItemError,
    Priority, ReadyItem, RecoveryItem,
};
pub use latency::{LatencyPoint, LatencyProfile, ProfileError};
pub use transaction::{
    BranchId, Epoch, TransactionError, TransactionMeta, TransactionStamp, TransactionState,
};
