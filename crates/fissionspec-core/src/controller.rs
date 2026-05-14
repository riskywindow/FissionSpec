use crate::{LatencyProfile, TransactionStamp};
use core::cmp;
use core::fmt;

/// Static priority. Larger values carry more weighted-flow cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(u16);

impl Priority {
    pub const BACKGROUND: Self = Self(0);
    pub const NORMAL: Self = Self(1);
    pub const URGENT: Self = Self(u16::MAX);

    #[must_use]
    pub const fn new(level: u16) -> Self {
        Self(level)
    }

    #[must_use]
    pub const fn level(self) -> u16 {
        self.0
    }

    /// Positive cost weight; priority zero still participates in flow cost.
    #[must_use]
    pub const fn weight(self) -> u64 {
        self.0 as u64 + 1
    }
}

/// Malformed ready/recovery work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemError {
    ZeroServiceUnits,
    DeadlineBeforeArrival,
    ReadyBeforeArrival,
    RecoveryReadyBeforeArrival,
}

impl fmt::Display for ItemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroServiceUnits => f.write_str("an item must carry at least one service unit"),
            Self::DeadlineBeforeArrival => f.write_str("deadline precedes original arrival"),
            Self::ReadyBeforeArrival => f.write_str("ready time precedes original arrival"),
            Self::RecoveryReadyBeforeArrival => {
                f.write_str("recovery ready time precedes original arrival")
            }
        }
    }
}

impl std::error::Error for ItemError {}

/// Ordinary decode/prefill work eligible for batching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadyItem {
    request_id: u64,
    original_arrival_ns: u64,
    ready_at_ns: u64,
    deadline_ns: u64,
    service_units: u32,
    priority: Priority,
}

impl ReadyItem {
    pub fn new(
        request_id: u64,
        arrival_ns: u64,
        deadline_ns: u64,
        service_units: u32,
        priority: Priority,
    ) -> Result<Self, ItemError> {
        Self::new_ready_at(
            request_id,
            arrival_ns,
            arrival_ns,
            deadline_ns,
            service_units,
            priority,
        )
    }

    /// Creates work whose current round became ready after the request arrived.
    pub fn new_ready_at(
        request_id: u64,
        original_arrival_ns: u64,
        ready_at_ns: u64,
        deadline_ns: u64,
        service_units: u32,
        priority: Priority,
    ) -> Result<Self, ItemError> {
        if service_units == 0 {
            return Err(ItemError::ZeroServiceUnits);
        }
        if deadline_ns < original_arrival_ns {
            return Err(ItemError::DeadlineBeforeArrival);
        }
        if ready_at_ns < original_arrival_ns {
            return Err(ItemError::ReadyBeforeArrival);
        }
        Ok(Self {
            request_id,
            original_arrival_ns,
            ready_at_ns,
            deadline_ns,
            service_units,
            priority,
        })
    }

    #[must_use]
    pub const fn request_id(self) -> u64 {
        self.request_id
    }

    #[must_use]
    /// Returns the request's original arrival time.
    ///
    /// This is a compatibility alias for [`Self::original_arrival_ns`].
    pub const fn arrival_ns(self) -> u64 {
        self.original_arrival_ns
    }

    #[must_use]
    pub const fn original_arrival_ns(self) -> u64 {
        self.original_arrival_ns
    }

    #[must_use]
    pub const fn ready_at_ns(self) -> u64 {
        self.ready_at_ns
    }

    #[must_use]
    pub const fn deadline_ns(self) -> u64 {
        self.deadline_ns
    }

    #[must_use]
    pub const fn service_units(self) -> u32 {
        self.service_units
    }

    #[must_use]
    pub const fn priority(self) -> Priority {
        self.priority
    }
}

/// Replay/verification work from a fissioned speculative branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecoveryItem {
    request_id: u64,
    original_arrival_ns: u64,
    ready_at_ns: u64,
    deadline_ns: u64,
    replay_units: u32,
    priority: Priority,
    transaction: TransactionStamp,
}

impl RecoveryItem {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: u64,
        original_arrival_ns: u64,
        ready_at_ns: u64,
        deadline_ns: u64,
        replay_units: u32,
        priority: Priority,
        transaction: TransactionStamp,
    ) -> Result<Self, ItemError> {
        if replay_units == 0 {
            return Err(ItemError::ZeroServiceUnits);
        }
        if deadline_ns < original_arrival_ns {
            return Err(ItemError::DeadlineBeforeArrival);
        }
        if ready_at_ns < original_arrival_ns {
            return Err(ItemError::RecoveryReadyBeforeArrival);
        }
        Ok(Self {
            request_id,
            original_arrival_ns,
            ready_at_ns,
            deadline_ns,
            replay_units,
            priority,
            transaction,
        })
    }

    #[must_use]
    pub const fn request_id(self) -> u64 {
        self.request_id
    }

    #[must_use]
    pub const fn original_arrival_ns(self) -> u64 {
        self.original_arrival_ns
    }

    #[must_use]
    pub const fn ready_at_ns(self) -> u64 {
        self.ready_at_ns
    }

    #[must_use]
    pub const fn deadline_ns(self) -> u64 {
        self.deadline_ns
    }

    #[must_use]
    pub const fn replay_units(self) -> u32 {
        self.replay_units
    }

    #[must_use]
    pub const fn priority(self) -> Priority {
        self.priority
    }

    #[must_use]
    pub const fn transaction(self) -> TransactionStamp {
        self.transaction
    }
}

/// A zero-copy heterogeneous microbatch view.
#[derive(Clone, Copy, Debug, Default)]
pub struct BatchView<'a> {
    ready: &'a [ReadyItem],
    recovery: &'a [RecoveryItem],
}

impl<'a> BatchView<'a> {
    #[must_use]
    pub const fn new(ready: &'a [ReadyItem], recovery: &'a [RecoveryItem]) -> Self {
        Self { ready, recovery }
    }

    #[must_use]
    pub const fn ready_only(ready: &'a [ReadyItem]) -> Self {
        Self {
            ready,
            recovery: &[],
        }
    }

    #[must_use]
    pub const fn recovery_only(recovery: &'a [RecoveryItem]) -> Self {
        Self {
            ready: &[],
            recovery,
        }
    }

    #[must_use]
    pub const fn ready(&self) -> &'a [ReadyItem] {
        self.ready
    }

    #[must_use]
    pub const fn recovery(&self) -> &'a [RecoveryItem] {
        self.recovery
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.ready.len() + self.recovery.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.ready.is_empty() && self.recovery.is_empty()
    }

    /// Saturating aggregate work, kept wide so capacity overflow is detectable.
    #[must_use]
    pub fn service_units(&self) -> u64 {
        let ready = self.ready.iter().fold(0_u64, |sum, item| {
            sum.saturating_add(u64::from(item.service_units))
        });
        self.recovery.iter().fold(ready, |sum, item| {
            sum.saturating_add(u64::from(item.replay_units))
        })
    }

    fn visit(self, mut visitor: impl FnMut(ItemSnapshot)) {
        for item in self.ready {
            visitor(ItemSnapshot {
                request_id: item.request_id,
                class: ItemClass::Ready,
                arrival_ns: item.original_arrival_ns,
                ready_at_ns: item.ready_at_ns,
                deadline_ns: item.deadline_ns,
                priority: item.priority,
            });
        }
        for item in self.recovery {
            visitor(ItemSnapshot {
                request_id: item.request_id,
                class: ItemClass::Recovery,
                arrival_ns: item.original_arrival_ns,
                ready_at_ns: item.ready_at_ns,
                deadline_ns: item.deadline_ns,
                priority: item.priority,
            });
        }
    }

    fn weighted_flow_cost(self, completion_ns: u64) -> u128 {
        let mut cost = 0_u128;
        self.visit(|item| {
            let flow = completion_ns.saturating_sub(item.arrival_ns);
            let contribution = u128::from(flow) * u128::from(item.priority.weight());
            cost = cost.saturating_add(contribution);
        });
        cost
    }

    fn latest_ready_at(self) -> u64 {
        let mut latest = 0;
        self.visit(|item| latest = cmp::max(latest, item.ready_at_ns));
        latest
    }

    fn earliest_coalescing_deadline(self, delay_ns: u64, fallback_ns: u64) -> u64 {
        let mut deadline = None;
        self.visit(|item| {
            // An unrepresentable upper bound is later than every representable
            // timestamp, so clamping it to MAX is conservative.
            let item_bound = item.ready_at_ns.saturating_add(delay_ns);
            deadline = Some(deadline.map_or(item_bound, |bound| cmp::min(bound, item_bound)));
        });
        deadline.unwrap_or_else(|| fallback_ns.saturating_add(delay_ns))
    }

    fn latest_slo_start(self, service_and_guard_ns: u64) -> Option<u64> {
        let mut latest = u64::MAX;
        let mut feasible = true;
        self.visit(|item| {
            if let Some(item_latest) = item.deadline_ns.checked_sub(service_and_guard_ns) {
                latest = cmp::min(latest, item_latest);
            } else {
                feasible = false;
            }
        });
        feasible.then_some(latest)
    }

    fn most_urgent(self) -> Option<ItemSnapshot> {
        let mut selected: Option<ItemSnapshot> = None;
        self.visit(|candidate| {
            select_more_urgent(&mut selected, candidate);
        });
        selected
    }

    fn first_not_ready(self, now_ns: u64) -> Option<ItemSnapshot> {
        let mut selected: Option<ItemSnapshot> = None;
        self.visit(|candidate| {
            if candidate.ready_at_ns > now_ns {
                select_more_urgent(&mut selected, candidate);
            }
        });
        selected
    }

    fn first_deadline_violation(
        self,
        guaranteed_completion_ns: Option<u64>,
    ) -> Option<ItemSnapshot> {
        let mut selected: Option<ItemSnapshot> = None;
        self.visit(|candidate| {
            let violates = match guaranteed_completion_ns {
                Some(completion) => candidate.deadline_ns < completion,
                None => true,
            };
            if violates {
                select_more_urgent(&mut selected, candidate);
            }
        });
        selected
    }
}

fn select_more_urgent(selected: &mut Option<ItemSnapshot>, candidate: ItemSnapshot) {
    let replace = match *selected {
        None => true,
        Some(current) => {
            candidate.deadline_ns < current.deadline_ns
                || (candidate.deadline_ns == current.deadline_ns
                    && candidate.priority > current.priority)
        }
    };
    if replace {
        *selected = Some(candidate);
    }
}

#[derive(Clone, Copy, Debug)]
struct ItemSnapshot {
    request_id: u64,
    class: ItemClass,
    arrival_ns: u64,
    ready_at_ns: u64,
    deadline_ns: u64,
    priority: Priority,
}

/// One predicted next microbatch, available atomically at a forecast time.
#[derive(Clone, Copy, Debug)]
pub struct ForecastBatch<'a> {
    pub available_at_ns: u64,
    pub batch: BatchView<'a>,
}

impl<'a> ForecastBatch<'a> {
    #[must_use]
    pub const fn new(available_at_ns: u64, batch: BatchView<'a>) -> Self {
        Self {
            available_at_ns,
            batch,
        }
    }

    fn effective_available_at(self) -> u64 {
        cmp::max(self.available_at_ns, self.batch.latest_ready_at())
    }
}

/// Immutable admission limits for a controller instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControllerConfig {
    max_coalescing_delay_ns: u64,
    max_batch_units: u32,
    deadline_guard_ns: u64,
}

impl ControllerConfig {
    pub fn new(
        max_coalescing_delay_ns: u64,
        max_batch_units: u32,
        deadline_guard_ns: u64,
    ) -> Result<Self, ControllerConfigError> {
        if max_batch_units == 0 {
            return Err(ControllerConfigError::ZeroBatchCapacity);
        }
        Ok(Self {
            max_coalescing_delay_ns,
            max_batch_units,
            deadline_guard_ns,
        })
    }

    #[must_use]
    pub const fn max_coalescing_delay_ns(self) -> u64 {
        self.max_coalescing_delay_ns
    }

    #[must_use]
    pub const fn max_batch_units(self) -> u32 {
        self.max_batch_units
    }

    #[must_use]
    pub const fn deadline_guard_ns(self) -> u64 {
        self.deadline_guard_ns
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerConfigError {
    ZeroBatchCapacity,
}

impl fmt::Display for ControllerConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("maximum batch capacity must be nonzero")
    }
}

impl std::error::Error for ControllerConfigError {}

/// Source queue for a bypassed item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemClass {
    Ready,
    Recovery,
}

/// Why an item must leave the fission/refusion batching lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BypassReason {
    /// Return the item to its readiness/recovery queue; it is not dispatchable.
    WorkNotReady,
    DeadlineUnreachable,
    BatchCapacityExceeded,
}

/// Hot-path result consumed by the serving loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerAction {
    DispatchNow,
    WaitUntil {
        at_ns: u64,
    },
    Bypass {
        request_id: u64,
        class: ItemClass,
        reason: BypassReason,
    },
}

/// Explanation useful for tracing and offline policy evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionReason {
    WaitingForFirstItem,
    CurrentItemNotReady,
    DeadlineUnreachable,
    CurrentBatchOverCapacity,
    NoForecast,
    ForecastOverCapacity,
    FusionOverCapacity,
    FusionWindowClosed,
    FusionDeadlineUnreachable,
    FusionIsOnlyFeasiblePlan,
    ForecastDeadlineUnreachable,
    FusionHasLowerCost,
    LaunchHasLowerCost,
    CostTiePrefersDispatch,
}

/// Objectives of the admissible horizon-2 plans, in weighted nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HorizonCosts {
    pub launch_now_then_next: Option<u128>,
    pub wait_and_fuse: Option<u128>,
}

/// An action plus the cost/bound data that produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    pub action: ControllerAction,
    pub reason: DecisionReason,
    pub costs: HorizonCosts,
    pub latest_safe_fuse_ns: Option<u64>,
}

/// Two-step fission/refusion controller.
///
/// Plan A launches the current batch immediately and the forecast batch after
/// both its arrival and the first launch complete. Plan B waits for the forecast
/// and fuses both batches. Both plans must meet every participating request's
/// guarded deadline; Plan B must also respect the current batch's coalescing
/// bound. Weighted flow chooses only among feasible plans.
#[derive(Clone, Copy, Debug)]
pub struct Horizon2Controller {
    config: ControllerConfig,
}

impl Horizon2Controller {
    #[must_use]
    pub const fn new(config: ControllerConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub const fn config(self) -> ControllerConfig {
        self.config
    }

    /// Returns only the serving-loop action. See [`Self::evaluate`] for traces.
    #[must_use]
    pub fn decide(
        &self,
        now_ns: u64,
        current: BatchView<'_>,
        forecast: Option<ForecastBatch<'_>>,
        profile: &LatencyProfile,
    ) -> ControllerAction {
        self.evaluate(now_ns, current, forecast, profile).action
    }

    /// Evaluates the two horizon plans without heap allocation.
    #[must_use]
    pub fn evaluate(
        &self,
        now_ns: u64,
        current: BatchView<'_>,
        forecast: Option<ForecastBatch<'_>>,
        profile: &LatencyProfile,
    ) -> Decision {
        let forecast = forecast.filter(|next| !next.batch.is_empty());
        if current.is_empty() {
            let at_ns = forecast.map_or_else(
                || now_ns.saturating_add(self.config.max_coalescing_delay_ns),
                |next| cmp::max(now_ns, next.effective_available_at()),
            );
            return decision(
                ControllerAction::WaitUntil { at_ns },
                DecisionReason::WaitingForFirstItem,
                HorizonCosts::default(),
                None,
            );
        }

        if let Some(item) = current.first_not_ready(now_ns) {
            return self.bypass(
                item,
                BypassReason::WorkNotReady,
                DecisionReason::CurrentItemNotReady,
            );
        }

        let current_units = current.service_units();
        if current_units > u64::from(self.config.max_batch_units) {
            return self.bypass(
                current.most_urgent().expect("nonempty batch has an item"),
                BypassReason::BatchCapacityExceeded,
                DecisionReason::CurrentBatchOverCapacity,
            );
        }

        let current_latency = profile.predict_ns(current_units as u32);
        let current_completion = now_ns.checked_add(current_latency);
        let guaranteed_completion = current_completion
            .and_then(|completion| completion.checked_add(self.config.deadline_guard_ns));
        if let Some(item) = current.first_deadline_violation(guaranteed_completion) {
            return self.bypass(
                item,
                BypassReason::DeadlineUnreachable,
                DecisionReason::DeadlineUnreachable,
            );
        }
        let current_completion =
            current_completion.expect("a feasible current deadline has a representable completion");

        let current_cost = current.weighted_flow_cost(current_completion);
        let Some(forecast) = forecast else {
            return decision(
                ControllerAction::DispatchNow,
                DecisionReason::NoForecast,
                HorizonCosts {
                    launch_now_then_next: Some(current_cost),
                    wait_and_fuse: None,
                },
                None,
            );
        };

        let next_units = forecast.batch.service_units();
        if next_units > u64::from(self.config.max_batch_units) {
            return decision(
                ControllerAction::DispatchNow,
                DecisionReason::ForecastOverCapacity,
                HorizonCosts {
                    launch_now_then_next: None,
                    wait_and_fuse: None,
                },
                None,
            );
        }

        let available_at = cmp::max(now_ns, forecast.effective_available_at());
        let next_latency = profile.predict_ns(next_units as u32);
        let next_start = cmp::max(available_at, current_completion);
        let next_completion = next_start.checked_add(next_latency);
        let guarded_next_completion = next_completion
            .and_then(|completion| completion.checked_add(self.config.deadline_guard_ns));
        let launch_violation = forecast
            .batch
            .first_deadline_violation(guarded_next_completion);
        let launch_feasible = next_completion.is_some()
            && guarded_next_completion.is_some()
            && launch_violation.is_none();
        let launch_cost = next_completion.map(|completion| {
            current_cost.saturating_add(forecast.batch.weighted_flow_cost(completion))
        });
        let fused_units = current_units
            .checked_add(next_units)
            .expect("two u32-bounded batches fit in u64");
        if fused_units > u64::from(self.config.max_batch_units) {
            let costs = HorizonCosts {
                launch_now_then_next: if launch_feasible { launch_cost } else { None },
                wait_and_fuse: None,
            };
            if launch_feasible {
                return decision(
                    ControllerAction::DispatchNow,
                    DecisionReason::FusionOverCapacity,
                    costs,
                    None,
                );
            }
            return self.resolve_infeasible_forecast(
                now_ns,
                available_at,
                forecast.batch,
                launch_violation,
                costs,
                None,
            );
        }

        let fused_latency = profile.predict_ns(fused_units as u32);
        let service_and_guard = fused_latency.checked_add(self.config.deadline_guard_ns);
        let coalescing_bound =
            current.earliest_coalescing_deadline(self.config.max_coalescing_delay_ns, now_ns);
        let latest_safe_fuse = service_and_guard.and_then(|duration| {
            current.latest_slo_start(duration).and_then(|current_slo| {
                forecast
                    .batch
                    .latest_slo_start(duration)
                    .map(|forecast_slo| {
                        cmp::min(coalescing_bound, cmp::min(current_slo, forecast_slo))
                    })
            })
        });
        let fused_completion = available_at.checked_add(fused_latency);
        let guarded_fused_completion = fused_completion
            .and_then(|completion| completion.checked_add(self.config.deadline_guard_ns));
        let current_fusion_violation = current.first_deadline_violation(guarded_fused_completion);
        let forecast_fusion_violation = forecast
            .batch
            .first_deadline_violation(guarded_fused_completion);
        let fusion_window_open = available_at <= coalescing_bound;
        let fusion_deadlines_met = guarded_fused_completion.is_some()
            && current_fusion_violation.is_none()
            && forecast_fusion_violation.is_none();
        let wait_feasible =
            fused_completion.is_some() && fusion_window_open && fusion_deadlines_met;
        let wait_cost = fused_completion.map(|completion| {
            current
                .weighted_flow_cost(completion)
                .saturating_add(forecast.batch.weighted_flow_cost(completion))
        });
        let costs = HorizonCosts {
            launch_now_then_next: if launch_feasible { launch_cost } else { None },
            wait_and_fuse: if wait_feasible { wait_cost } else { None },
        };

        match (launch_feasible, wait_feasible) {
            (true, true) => {
                let launch_cost = launch_cost.expect("a feasible launch plan has a cost");
                let wait_cost = wait_cost.expect("a feasible fused plan has a cost");
                if wait_cost < launch_cost {
                    decision(
                        ControllerAction::WaitUntil {
                            at_ns: available_at,
                        },
                        DecisionReason::FusionHasLowerCost,
                        costs,
                        latest_safe_fuse,
                    )
                } else {
                    let reason = if wait_cost == launch_cost {
                        DecisionReason::CostTiePrefersDispatch
                    } else {
                        DecisionReason::LaunchHasLowerCost
                    };
                    decision(
                        ControllerAction::DispatchNow,
                        reason,
                        costs,
                        latest_safe_fuse,
                    )
                }
            }
            (false, true) => decision(
                ControllerAction::WaitUntil {
                    at_ns: available_at,
                },
                DecisionReason::FusionIsOnlyFeasiblePlan,
                costs,
                latest_safe_fuse,
            ),
            (true, false) => decision(
                ControllerAction::DispatchNow,
                if fusion_window_open {
                    DecisionReason::FusionDeadlineUnreachable
                } else {
                    DecisionReason::FusionWindowClosed
                },
                costs,
                latest_safe_fuse,
            ),
            (false, false) => self.resolve_infeasible_forecast(
                now_ns,
                available_at,
                forecast.batch,
                launch_violation,
                costs,
                latest_safe_fuse,
            ),
        }
    }

    /// A future forecast is not yet owned by the action consumer, so it cannot
    /// be bypassed safely. Dispatch current work and re-evaluate that forecast
    /// when it becomes actionable. An already-actionable doomed forecast item
    /// can be removed immediately.
    fn resolve_infeasible_forecast(
        &self,
        now_ns: u64,
        available_at: u64,
        forecast: BatchView<'_>,
        violation: Option<ItemSnapshot>,
        costs: HorizonCosts,
        latest_safe_fuse_ns: Option<u64>,
    ) -> Decision {
        if available_at <= now_ns {
            let item = violation
                .or_else(|| forecast.most_urgent())
                .expect("a nonempty forecast has an item");
            decision(
                ControllerAction::Bypass {
                    request_id: item.request_id,
                    class: item.class,
                    reason: BypassReason::DeadlineUnreachable,
                },
                DecisionReason::ForecastDeadlineUnreachable,
                costs,
                latest_safe_fuse_ns,
            )
        } else {
            decision(
                ControllerAction::DispatchNow,
                DecisionReason::ForecastDeadlineUnreachable,
                costs,
                latest_safe_fuse_ns,
            )
        }
    }

    fn bypass(
        &self,
        item: ItemSnapshot,
        reason: BypassReason,
        decision_reason: DecisionReason,
    ) -> Decision {
        decision(
            ControllerAction::Bypass {
                request_id: item.request_id,
                class: item.class,
                reason,
            },
            decision_reason,
            HorizonCosts::default(),
            None,
        )
    }
}

fn decision(
    action: ControllerAction,
    reason: DecisionReason,
    costs: HorizonCosts,
    latest_safe_fuse_ns: Option<u64>,
) -> Decision {
    Decision {
        action,
        reason,
        costs,
        latest_safe_fuse_ns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BranchId, Epoch, LatencyPoint};

    fn profile(points: &[(u32, u64)]) -> LatencyProfile {
        LatencyProfile::new(
            points
                .iter()
                .map(|&(units, latency)| LatencyPoint::new(units, latency))
                .collect(),
        )
        .unwrap()
    }

    fn ready(id: u64, arrival: u64, deadline: u64, units: u32, priority: u16) -> ReadyItem {
        ReadyItem::new(id, arrival, deadline, units, Priority::new(priority)).unwrap()
    }

    fn ready_at(
        id: u64,
        original_arrival: u64,
        ready_at: u64,
        deadline: u64,
        units: u32,
        priority: u16,
    ) -> ReadyItem {
        ReadyItem::new_ready_at(
            id,
            original_arrival,
            ready_at,
            deadline,
            units,
            Priority::new(priority),
        )
        .unwrap()
    }

    fn controller(delay: u64, capacity: u32) -> Horizon2Controller {
        Horizon2Controller::new(ControllerConfig::new(delay, capacity, 0).unwrap())
    }

    #[test]
    fn fuses_when_weighted_flow_is_lower() {
        let latency = profile(&[(1, 100), (2, 110)]);
        let current = [ready(1, 0, 1_000, 1, 0)];
        let next = [ready(2, 10, 1_000, 1, 0)];
        let result = controller(50, 2).evaluate(
            0,
            BatchView::ready_only(&current),
            Some(ForecastBatch::new(10, BatchView::ready_only(&next))),
            &latency,
        );

        assert_eq!(result.action, ControllerAction::WaitUntil { at_ns: 10 });
        assert_eq!(result.reason, DecisionReason::FusionHasLowerCost);
        assert_eq!(result.costs.launch_now_then_next, Some(290));
        assert_eq!(result.costs.wait_and_fuse, Some(230));
    }

    #[test]
    fn dispatches_when_forecast_is_too_late() {
        let latency = profile(&[(1, 100), (2, 110)]);
        let current = [ready(1, 0, 1_000, 1, 0)];
        let next = [ready(2, 100, 1_000, 1, 0)];
        let action = controller(200, 2).decide(
            0,
            BatchView::ready_only(&current),
            Some(ForecastBatch::new(100, BatchView::ready_only(&next))),
            &latency,
        );
        assert_eq!(action, ControllerAction::DispatchNow);
    }

    #[test]
    fn priority_can_flip_the_plan() {
        let latency = profile(&[(1, 100), (2, 110)]);
        let urgent_current = [ready(1, 0, 1_000, 1, 99)];
        let next = [ready(2, 10, 1_000, 1, 0)];
        let result = controller(50, 2).evaluate(
            0,
            BatchView::ready_only(&urgent_current),
            Some(ForecastBatch::new(10, BatchView::ready_only(&next))),
            &latency,
        );
        assert_eq!(result.action, ControllerAction::DispatchNow);
        assert_eq!(result.reason, DecisionReason::LaunchHasLowerCost);
    }

    #[test]
    fn coalescing_age_closes_the_window() {
        let latency = profile(&[(1, 50), (2, 60)]);
        let current = [ready(1, 0, 1_000, 1, 0)];
        let next = [ready(2, 101, 1_000, 1, 0)];
        let result = controller(50, 2).evaluate(
            100,
            BatchView::ready_only(&current),
            Some(ForecastBatch::new(101, BatchView::ready_only(&next))),
            &latency,
        );
        assert_eq!(result.action, ControllerAction::DispatchNow);
        assert_eq!(result.reason, DecisionReason::FusionWindowClosed);
        assert_eq!(result.latest_safe_fuse_ns, Some(50));
    }

    #[test]
    fn earliest_slo_slack_closes_the_window() {
        let latency = profile(&[(1, 50), (2, 60)]);
        let current = [ready(1, 0, 55, 1, 0)];
        let next = [ready(2, 10, 1_000, 1, 0)];
        let result = controller(100, 2).evaluate(
            0,
            BatchView::ready_only(&current),
            Some(ForecastBatch::new(10, BatchView::ready_only(&next))),
            &latency,
        );
        assert_eq!(result.action, ControllerAction::DispatchNow);
        assert_eq!(result.reason, DecisionReason::FusionDeadlineUnreachable);
        assert_eq!(result.latest_safe_fuse_ns, None);
    }

    #[test]
    fn impossible_fused_deadline_at_time_zero_is_rejected() {
        let latency = profile(&[(1, 50), (2, 60)]);
        let current = [ready(1, 0, 55, 1, 0)];
        let next = [ready(2, 0, 1_000, 1, 0)];
        let result = controller(100, 2).evaluate(
            0,
            BatchView::ready_only(&current),
            Some(ForecastBatch::new(0, BatchView::ready_only(&next))),
            &latency,
        );

        assert_eq!(result.action, ControllerAction::DispatchNow);
        assert_eq!(result.reason, DecisionReason::FusionDeadlineUnreachable);
        assert_eq!(result.costs.launch_now_then_next, Some(150));
        assert_eq!(result.costs.wait_and_fuse, None);
    }

    #[test]
    fn feasible_fusion_beats_a_forecast_deadline_miss() {
        let latency = profile(&[(1, 100), (2, 110)]);
        let urgent_current = [ready(1, 0, 1_000, 1, 99)];
        let next = [ready(2, 0, 150, 1, 0)];
        let result = controller(100, 2).evaluate(
            0,
            BatchView::ready_only(&urgent_current),
            Some(ForecastBatch::new(0, BatchView::ready_only(&next))),
            &latency,
        );

        assert_eq!(result.action, ControllerAction::WaitUntil { at_ns: 0 });
        assert_eq!(result.reason, DecisionReason::FusionIsOnlyFeasiblePlan);
        assert_eq!(result.costs.launch_now_then_next, None);
        assert_eq!(result.costs.wait_and_fuse, Some(11_110));
    }

    #[test]
    fn checked_timestamp_overflow_is_deadline_unreachable() {
        let latency = profile(&[(1, 1)]);
        let current = [ready(1, u64::MAX, u64::MAX, 1, 0)];
        let result =
            controller(1, 1).evaluate(u64::MAX, BatchView::ready_only(&current), None, &latency);

        assert_eq!(
            result.action,
            ControllerAction::Bypass {
                request_id: 1,
                class: ItemClass::Ready,
                reason: BypassReason::DeadlineUnreachable,
            }
        );
    }

    #[test]
    fn fused_timestamp_overflow_falls_back_to_a_feasible_launch() {
        let latency = profile(&[(1, 1), (2, 3)]);
        let now = u64::MAX - 2;
        let current = [ready(1, now, u64::MAX, 1, 0)];
        let next = [ready(2, now, u64::MAX, 1, 0)];
        let result = controller(10, 2).evaluate(
            now,
            BatchView::ready_only(&current),
            Some(ForecastBatch::new(now, BatchView::ready_only(&next))),
            &latency,
        );

        assert_eq!(result.action, ControllerAction::DispatchNow);
        assert_eq!(result.reason, DecisionReason::FusionDeadlineUnreachable);
        assert_eq!(result.costs.launch_now_then_next, Some(3));
        assert_eq!(result.costs.wait_and_fuse, None);
    }

    #[test]
    fn impossible_deadline_bypasses_to_eager_lane() {
        let latency = profile(&[(1, 50)]);
        let current = [ready(7, 0, 40, 1, 4)];
        let result =
            controller(100, 2).evaluate(0, BatchView::ready_only(&current), None, &latency);
        assert_eq!(
            result.action,
            ControllerAction::Bypass {
                request_id: 7,
                class: ItemClass::Ready,
                reason: BypassReason::DeadlineUnreachable,
            }
        );
    }

    #[test]
    fn over_capacity_batch_bypasses_earliest_deadline() {
        let latency = profile(&[(1, 50), (2, 60)]);
        let current = [ready(1, 0, 500, 1, 0), ready(2, 0, 400, 1, 0)];
        let result =
            controller(100, 1).evaluate(0, BatchView::ready_only(&current), None, &latency);
        assert_eq!(
            result.action,
            ControllerAction::Bypass {
                request_id: 2,
                class: ItemClass::Ready,
                reason: BypassReason::BatchCapacityExceeded,
            }
        );
    }

    #[test]
    fn recovery_work_refuses_with_ready_work() {
        let latency = profile(&[(1, 100), (2, 110)]);
        let current = [ready(1, 0, 1_000, 1, 0)];
        let recovery = [RecoveryItem::new(
            9,
            0,
            10,
            1_000,
            1,
            Priority::NORMAL,
            TransactionStamp::new(Epoch(3), BranchId(2)),
        )
        .unwrap()];
        let action = controller(50, 2).decide(
            0,
            BatchView::ready_only(&current),
            Some(ForecastBatch::new(8, BatchView::recovery_only(&recovery))),
            &latency,
        );
        // The recovery's actual ready time (10), not optimistic forecast (8), wins.
        assert_eq!(action, ControllerAction::WaitUntil { at_ns: 10 });
    }

    #[test]
    fn current_work_that_is_not_ready_is_removed_from_the_batch() {
        let latency = profile(&[(1, 50)]);
        let current = [ready_at(1, 0, 10, 1_000, 1, 0)];
        let result = controller(20, 1).evaluate(0, BatchView::ready_only(&current), None, &latency);

        assert_eq!(
            result.action,
            ControllerAction::Bypass {
                request_id: 1,
                class: ItemClass::Ready,
                reason: BypassReason::WorkNotReady,
            }
        );
        assert_eq!(result.reason, DecisionReason::CurrentItemNotReady);
    }

    #[test]
    fn later_round_readiness_resets_the_coalescing_window() {
        let latency = profile(&[(1, 100), (2, 110)]);
        let current = [ready_at(1, 0, 100, 1_000, 1, 0)];
        let next = [ready_at(2, 0, 120, 1_000, 1, 0)];
        let result = controller(50, 2).evaluate(
            100,
            BatchView::ready_only(&current),
            Some(ForecastBatch::new(120, BatchView::ready_only(&next))),
            &latency,
        );

        assert_eq!(result.action, ControllerAction::WaitUntil { at_ns: 120 });
        assert_eq!(result.latest_safe_fuse_ns, Some(150));
    }

    #[test]
    fn fusion_over_capacity_still_reports_the_complete_launch_cost() {
        let latency = profile(&[(1, 50)]);
        let current = [ready(1, 0, 1_000, 1, 0)];
        let next = [ready(2, 10, 1_000, 1, 0)];
        let result = controller(100, 1).evaluate(
            0,
            BatchView::ready_only(&current),
            Some(ForecastBatch::new(10, BatchView::ready_only(&next))),
            &latency,
        );

        assert_eq!(result.action, ControllerAction::DispatchNow);
        assert_eq!(result.reason, DecisionReason::FusionOverCapacity);
        assert_eq!(result.costs.launch_now_then_next, Some(140));
    }

    #[test]
    fn actionable_forecast_with_no_feasible_plan_is_bypassed() {
        let latency = profile(&[(1, 100), (2, 110)]);
        let current = [ready(1, 0, 1_000, 1, 0)];
        let next = [ready(2, 0, 50, 1, 0)];
        let result = controller(100, 2).evaluate(
            0,
            BatchView::ready_only(&current),
            Some(ForecastBatch::new(0, BatchView::ready_only(&next))),
            &latency,
        );

        assert_eq!(
            result.action,
            ControllerAction::Bypass {
                request_id: 2,
                class: ItemClass::Ready,
                reason: BypassReason::DeadlineUnreachable,
            }
        );
        assert_eq!(result.reason, DecisionReason::ForecastDeadlineUnreachable);
    }

    #[test]
    fn empty_queue_waits_without_spinning() {
        let latency = profile(&[(1, 50)]);
        let result = controller(20, 1).evaluate(5, BatchView::default(), None, &latency);
        assert_eq!(result.action, ControllerAction::WaitUntil { at_ns: 25 });
        assert_eq!(result.reason, DecisionReason::WaitingForFirstItem);
    }

    #[test]
    fn empty_forecast_is_ignored_when_the_queue_is_empty() {
        let latency = profile(&[(1, 50)]);
        let result = controller(20, 1).evaluate(
            5,
            BatchView::default(),
            Some(ForecastBatch::new(5, BatchView::default())),
            &latency,
        );
        assert_eq!(result.action, ControllerAction::WaitUntil { at_ns: 25 });
    }

    #[test]
    fn validates_items_and_config() {
        assert!(Priority::NORMAL > Priority::BACKGROUND);
        assert_eq!(
            ControllerConfig::new(1, 0, 0),
            Err(ControllerConfigError::ZeroBatchCapacity)
        );
        assert_eq!(
            ReadyItem::new(1, 2, 1, 1, Priority::NORMAL),
            Err(ItemError::DeadlineBeforeArrival)
        );
        assert_eq!(
            ReadyItem::new(1, 1, 2, 0, Priority::NORMAL),
            Err(ItemError::ZeroServiceUnits)
        );
        assert_eq!(
            ReadyItem::new_ready_at(1, 2, 1, 3, 1, Priority::NORMAL),
            Err(ItemError::ReadyBeforeArrival)
        );
    }
}
