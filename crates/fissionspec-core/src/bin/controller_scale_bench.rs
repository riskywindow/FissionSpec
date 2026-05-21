use fissionspec_core::{
    BatchView, ControllerAction, ControllerConfig, ForecastBatch, Horizon2Controller, LatencyPoint,
    LatencyProfile, Priority, ReadyItem,
};
use std::hint::black_box;
use std::time::Instant;

const WARNING: &str = "LOCAL CPU TIMING / NOT GPU EVIDENCE / NOT CROSS-HOST COMPARABLE";
const SIZES: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 96, 128];

fn ready_items(count: usize, id_offset: u64, ready_at_ns: u64) -> Vec<ReadyItem> {
    (0..count)
        .map(|ordinal| {
            ReadyItem::new_ready_at(
                id_offset + ordinal as u64,
                0,
                ready_at_ns,
                1_000_000_000,
                1,
                Priority::new((ordinal % 4) as u16),
            )
            .expect("benchmark item must be valid")
        })
        .collect()
}

fn action_code(action: ControllerAction) -> u64 {
    match action {
        ControllerAction::DispatchNow => 1,
        ControllerAction::WaitUntil { at_ns } => at_ns.rotate_left(7),
        ControllerAction::Bypass { request_id, .. } => request_id.rotate_left(17),
    }
}

fn main() {
    let target_item_visits = std::env::args()
        .nth(1)
        .and_then(|argument| argument.parse::<u64>().ok())
        .unwrap_or(2_000_000)
        .max(10_000);
    let repeats = std::env::args()
        .nth(2)
        .and_then(|argument| argument.parse::<usize>().ok())
        .unwrap_or(7)
        .clamp(1, 31);
    let profile = LatencyProfile::new(vec![
        LatencyPoint::new(1, 42_000),
        LatencyPoint::new(8, 61_000),
        LatencyPoint::new(32, 109_000),
        LatencyPoint::new(64, 180_000),
        LatencyPoint::new(128, 315_000),
        LatencyPoint::new(256, 580_000),
    ])
    .expect("valid benchmark profile");
    let controller = Horizon2Controller::new(
        ControllerConfig::new(20_000, 512, 2_000).expect("valid controller config"),
    );

    println!("{{");
    println!("  \"schema_version\": 1,");
    println!("  \"measurement_warning\": \"{WARNING}\",");
    println!("  \"timer\": \"std::time::Instant\",");
    println!("  \"build_profile\": \"release\",");
    println!("  \"target_item_visits_per_repeat\": {target_item_visits},");
    println!("  \"repeats\": {repeats},");
    println!("  \"samples\": [");
    for (size_index, size) in SIZES.into_iter().enumerate() {
        let current_items = ready_items(size, 1, 0);
        let forecast_items = ready_items(size, 1_000_000, 8_000);
        let current = BatchView::ready_only(&current_items);
        let forecast = ForecastBatch::new(8_000, BatchView::ready_only(&forecast_items));
        let total_rows = (size * 2) as u64;
        let iterations = (target_item_visits / total_rows).clamp(200, 1_000_000);
        let warmup = (iterations / 10).max(100);
        let mut digest = 0_u64;
        for iteration in 0..warmup {
            let action = controller.decide(
                black_box(iteration & 7),
                black_box(current),
                black_box(Some(forecast)),
                black_box(&profile),
            );
            digest ^= action_code(black_box(action));
        }
        let mut elapsed_repeats = Vec::with_capacity(repeats);
        for repeat in 0..repeats {
            let started = Instant::now();
            for iteration in 0..iterations {
                let action = controller.decide(
                    black_box((iteration + repeat as u64) & 7),
                    black_box(current),
                    black_box(Some(forecast)),
                    black_box(&profile),
                );
                digest ^= action_code(black_box(action));
            }
            elapsed_repeats.push(started.elapsed().as_nanos());
        }
        print!(
            "    {{\"current_rows\": {size}, \"future_rows\": {size}, \
             \"iterations_per_repeat\": {iterations}, \"elapsed_ns_repeats\": ["
        );
        for (index, elapsed) in elapsed_repeats.iter().enumerate() {
            if index > 0 {
                print!(", ");
            }
            print!("{elapsed}");
        }
        print!("], \"digest\": {digest}}}");
        if size_index + 1 != SIZES.len() {
            println!(",");
        } else {
            println!();
        }
    }
    println!("  ]");
    println!("}}");
}
