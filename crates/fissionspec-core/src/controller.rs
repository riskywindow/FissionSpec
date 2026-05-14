use crate::{LatencyProfile, TransactionStamp};
use core::cmp;
use core::fmt;

/// Static priority. Larger values carry more weighted-flow cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(u16);

impl Priority {
    pub const BACKGROUND: Self = Self(0);
    pub const NORMAL: Self = Self(0);
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
    RecoveryReadyBeforeArrival,
}

impl fmt::Display for ItemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroServiceUnits => f.write_str("an item must carry at least one service unit"),
            Self::DeadlineBeforeArrival => f.write_str("deadline precedes original arrival"),
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
    arrival_ns: u64,
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
        if service_units == 0 {
            return Err(ItemError::ZeroServiceUnits);
        }
        if deadline_ns < arrival_ns {
            return Err(ItemError::DeadlineBeforeArrival);
        }
        Ok(Self {
            request_id,
            arrival_ns,
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
    pub const fn arrival_ns(self) -> u64 {
        self.arrival_ns
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
                arrival_ns: item.arrival_ns,
                ready_at_ns: item.arrival_ns,
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
            let item_bound = item.ready_at_ns.saturating_add(delay_ns);
            deadline = Some(deadline.map_or(item_bound, |bound| cmp::min(bound, item_bound)));
        });
        deadline.unwrap_or(fallback_ns.saturating_add(delay_ns))
    }

    fn latest_slo_start(self, service_and_guard_ns: u64) -> u64 {
        let mut latest = u64::MAX;
        self.visit(|item| {
            latest = cmp::min(
                latest,
                item.deadline_ns.saturating_sub(service_and_guard_ns),
            );
        });
        latest
    }

    fn most_urgent(self) -> Option<ItemSnapshot> {
        let mut selected: Option<ItemSnapshot> = None;
        self.visit(|candidate| {
            let replace = match selected {
                None => true,
                Some(current) => {
                    candidate.deadline_ns < current.deadline_ns
                        || (candidate.deadline_ns == current.deadline_ns
                            && candidate.priority > current.priority)
                }
            };
            if replace {
                selected = Some(candidate);
            }
        });
        selected
    }

    fn first_deadline_violation(self, guaranteed_completion_ns: u64) -> Option<ItemSnapshot> {
        let mut selected: Option<ItemSnapshot> = None;
        self.visit(|candidate| {
            if candidate.deadline_ns < guaranteed_completion_ns {
                let replace = match selected {
                    None => true,
                    Some(current) => {
                        candidate.deadline_ns < current.deadline_ns
                            || (candidate.deadline_ns == current.deadline_ns
                                && candidate.priority > current.priority)
                    }
                };
                if replace {
                    selected = Some(candidate);
                }
            }
        });
        selected
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
    DeadlineUnreachable,
    CurrentBatchOverCapacity,
    NoForecast,
    FusionOverCapacity,
    FusionWindowClosed,
    FusionHasLowerCost,
    LaunchHasLowerCost,
    CostTiePrefersDispatch,
}

/// Both horizon-2 plan objectives, in priority-weighted nanoseconds.
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
/// and fuses both batches. The latter is admissible only before the oldest
/// item's coalescing bound and every member's latency-adjusted deadline.
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

        let current_units = current.service_units();
        if current_units > u64::from(self.config.max_batch_units) {
            return self.bypass(
                current.most_urgent().expect("nonempty batch has an item"),
                BypassReason::BatchCapacityExceeded,
                DecisionReason::CurrentBatchOverCapacity,
            );
        }

        let current_latency = profile.predict_ns(current_units as u32);
        let current_completion = now_ns.saturating_add(current_latency);
        let guaranteed_completion =
            current_completion.saturating_add(self.config.deadline_guard_ns);
        if let Some(item) = current.first_deadline_violation(guaranteed_completion) {
            return self.bypass(
                item,
                BypassReason::DeadlineUnreachable,
                DecisionReason::DeadlineUnreachable,
            );
        }

        let current_cost = current.weighted_flow_cost(current_completion);
        let Some(forecast) = forecast.filter(|next| !next.batch.is_empty()) else {
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
        let fused_units = current_units.saturating_add(next_units);
        if next_units > u64::from(self.config.max_batch_units)
            || fused_units > u64::from(self.config.max_batch_units)
        {
            return decision(
                ControllerAction::DispatchNow,
                DecisionReason::FusionOverCapacity,
                HorizonCosts {
                    launch_now_then_next: Some(current_cost),
                    wait_and_fuse: None,
                },
                None,
            );
        }

        let available_at = cmp::max(now_ns, forecast.effective_available_at());
        let next_latency = profile.predict_ns(next_units as u32);
        let next_start = cmp::max(available_at, current_completion);
        let next_completion = next_start.saturating_add(next_latency);
        let launch_cost =
            current_cost.saturating_add(forecast.batch.weighted_flow_cost(next_completion));

        let fused_latency = profile.predict_ns(fused_units as u32);
        let service_and_guard = fused_latency.saturating_add(self.config.deadline_guard_ns);
        let coalescing_bound =
            current.earliest_coalescing_deadline(self.config.max_coalescing_delay_ns, now_ns);
        let slo_bound = cmp::min(
            current.latest_slo_start(service_and_guard),
            forecast.batch.latest_slo_start(service_and_guard),
        );
        let latest_safe_fuse = cmp::min(coalescing_bound, slo_bound);
        if available_at > latest_safe_fuse {
            return decision(
                ControllerAction::DispatchNow,
                DecisionReason::FusionWindowClosed,
                HorizonCosts {
                    launch_now_then_next: Some(launch_cost),
                    wait_and_fuse: None,
                },
                Some(latest_safe_fuse),
            );
        }

        let fused_completion = available_at.saturating_add(fused_latency);
        let wait_cost = current
            .weighted_flow_cost(fused_completion)
            .saturating_add(forecast.batch.weighted_flow_cost(fused_completion));
        let costs = HorizonCosts {
            launch_now_then_next: Some(launch_cost),
            wait_and_fuse: Some(wait_cost),
        };

        if wait_cost < launch_cost {
            decision(
                ControllerAction::WaitUntil {
                    at_ns: available_at,
                },
                DecisionReason::FusionHasLowerCost,
                costs,
                Some(latest_safe_fuse),
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
                Some(latest_safe_fuse),
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
        assert_eq!(result.latest_safe_fuse_ns, Some(0));
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
    fn empty_queue_waits_without_spinning() {
        let latency = profile(&[(1, 50)]);
        let result = controller(20, 1).evaluate(5, BatchView::default(), None, &latency);
        assert_eq!(result.action, ControllerAction::WaitUntil { at_ns: 25 });
        assert_eq!(result.reason, DecisionReason::WaitingForFirstItem);
    }

    #[test]
    fn validates_items_and_config() {
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
    }
}
