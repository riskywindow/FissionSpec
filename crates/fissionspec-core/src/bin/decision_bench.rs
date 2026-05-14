use fissionspec_core::{
    BatchView, ControllerAction, ControllerConfig, ForecastBatch, Horizon2Controller, LatencyPoint,
    LatencyProfile, Priority, ReadyItem,
};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let iterations = std::env::args()
        .nth(1)
        .and_then(|argument| argument.parse::<u64>().ok())
        .unwrap_or(5_000_000)
        .max(1);

    let profile = LatencyProfile::new(vec![
        LatencyPoint::new(1, 42_000),
        LatencyPoint::new(4, 51_000),
        LatencyPoint::new(8, 61_000),
        LatencyPoint::new(16, 78_000),
        LatencyPoint::new(32, 109_000),
    ])
    .expect("valid profile");
    let current = [
        ReadyItem::new(1, 0, 500_000, 2, Priority::new(3)).expect("valid item"),
        ReadyItem::new(2, 2_000, 450_000, 2, Priority::NORMAL).expect("valid item"),
    ];
    let forecast_items = [
        ReadyItem::new(3, 8_000, 500_000, 2, Priority::NORMAL).expect("valid item"),
        ReadyItem::new(4, 8_000, 500_000, 2, Priority::new(1)).expect("valid item"),
    ];
    let current = BatchView::ready_only(&current);
    let forecast = ForecastBatch::new(8_000, BatchView::ready_only(&forecast_items));
    let controller = Horizon2Controller::new(
        ControllerConfig::new(20_000, 32, 2_000).expect("valid controller config"),
    );

    for _ in 0..100_000 {
        black_box(controller.decide(0, current, Some(forecast), &profile));
    }

    let started = Instant::now();
    let mut digest = 0_u64;
    for iteration in 0..iterations {
        let action = black_box(controller.decide(
            black_box(iteration & 7),
            black_box(current),
            black_box(Some(forecast)),
            black_box(&profile),
        ));
        digest ^= match action {
            ControllerAction::DispatchNow => 1,
            ControllerAction::WaitUntil { at_ns } => at_ns,
            ControllerAction::Bypass { request_id, .. } => request_id,
        };
    }
    let elapsed = started.elapsed();
    let ns_per_operation = elapsed.as_nanos() as f64 / iterations as f64;
    println!(
        "fissionspec horizon-2: {iterations} decisions in {elapsed:?}; {ns_per_operation:.2} ns/op (digest={digest})"
    );
}
