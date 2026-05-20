//! Rust consumer of the reviewed cross-language fixture corpus.
//!
//! This test intentionally has its own tiny TSV reader and never invokes the
//! Python implementation. Expected values are checked into the corpus.

use fissionspec_core::{
    batch_miss_probability, expected_hol_victims, hol_amplification_factor,
    stable_survival_product, BatchView, BranchId, ControllerAction, ControllerConfig, Epoch,
    ForecastBatch, Horizon2Controller, LatencyPoint, LatencyProfile, Priority, ReadyItem,
    TransactionMeta, TransactionStamp, TransactionState,
};

const METRICS: &str = include_str!("../../../fixtures/cross_language/metrics.tsv");
const LATENCY: &str = include_str!("../../../fixtures/cross_language/latency.tsv");
const HORIZON2: &str = include_str!("../../../fixtures/cross_language/horizon2.tsv");
const TRANSACTIONS: &str = include_str!("../../../fixtures/cross_language/transactions.tsv");
const MALFORMED: &str = include_str!("../../../fixtures/cross_language/malformed.tsv");

fn fixture_rows<'a>(input: &'a str, expected_header: &str) -> Vec<Vec<&'a str>> {
    let mut lines = input
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'));
    let header = lines.next().expect("fixture has a header");
    assert_eq!(header, expected_header, "fixture header is canonical");
    let width = header.split('\t').count();
    let rows: Vec<Vec<&str>> = lines.map(|line| line.split('\t').collect()).collect();
    assert!(!rows.is_empty(), "fixture has at least one data row");
    assert!(
        rows.iter().all(|row| row.len() == width),
        "every row has the header width"
    );
    let mut cases: Vec<&str> = rows.iter().map(|row| row[0]).collect();
    cases.sort_unstable();
    cases.dedup();
    assert_eq!(cases.len(), rows.len(), "fixture case names are unique");
    rows
}

fn probabilities(encoded: &str) -> Vec<f64> {
    encoded
        .split(',')
        .map(|value| value.parse().expect("valid fixture probability"))
        .collect()
}

fn latency_points(encoded: &str) -> Vec<LatencyPoint> {
    if encoded == "-" {
        return vec![];
    }
    encoded
        .split(';')
        .map(|point| {
            let (units, latency) = point.split_once(':').expect("point contains ':'");
            LatencyPoint::new(
                units.parse().expect("valid work units"),
                latency.parse().expect("valid nanoseconds"),
            )
        })
        .collect()
}

fn close(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "{left} differs from {right} by more than {tolerance}"
    );
}

#[test]
fn reviewed_analytical_metric_fixtures() {
    for row in fixture_rows(
        METRICS,
        "case\tmiss_probabilities\tsurvival_probability\tbatch_miss_probability\t\
         hol_amplification\texpected_healthy_victims\ttolerance",
    ) {
        let misses = probabilities(row[1]);
        let tolerance: f64 = row[6].parse().unwrap();
        close(
            stable_survival_product(&misses).unwrap(),
            row[2].parse().unwrap(),
            tolerance,
        );
        close(
            batch_miss_probability(&misses).unwrap(),
            row[3].parse().unwrap(),
            tolerance,
        );
        close(
            hol_amplification_factor(&misses).unwrap(),
            row[4].parse().unwrap(),
            tolerance,
        );
        close(
            expected_hol_victims(&misses).unwrap(),
            row[5].parse().unwrap(),
            tolerance,
        );
    }
}

#[test]
fn reviewed_latency_fixtures() {
    for row in fixture_rows(LATENCY, "case\tpoints_ns\tquery_units\texpected_ns") {
        let profile = LatencyProfile::new(latency_points(row[1])).unwrap();
        let query: u32 = row[2].parse().unwrap();
        let expected: u64 = row[3].parse().unwrap();
        assert_eq!(profile.predict_ns(query), expected, "case {}", row[0]);
    }
}

#[test]
fn reviewed_flattened_horizon_fixtures() {
    for row in fixture_rows(
        HORIZON2,
        "case\tpoints_ns\tcurrent_count\tforecast_count\tforecast_at_ns\tcapacity\t\
         max_wait_ns\tcurrent_deadline_ns\tfuture_deadline_ns\texpected_action",
    ) {
        let profile = LatencyProfile::new(latency_points(row[1])).unwrap();
        let current_count: usize = row[2].parse().unwrap();
        let forecast_count: usize = row[3].parse().unwrap();
        let forecast_at = (row[4] != "-").then(|| row[4].parse::<u64>().unwrap());
        let capacity: u32 = row[5].parse().unwrap();
        let max_wait: u64 = row[6].parse().unwrap();
        let current_deadline: u64 = row[7].parse().unwrap();
        let future_deadline = (row[8] != "-").then(|| row[8].parse::<u64>().unwrap());

        let current: Vec<ReadyItem> = (0..current_count)
            .map(|index| {
                ReadyItem::new(index as u64, 0, current_deadline, 1, Priority::NORMAL).unwrap()
            })
            .collect();
        let future: Vec<ReadyItem> = (0..forecast_count)
            .map(|index| {
                ReadyItem::new(
                    10_000 + index as u64,
                    forecast_at.unwrap(),
                    future_deadline.unwrap(),
                    1,
                    Priority::NORMAL,
                )
                .unwrap()
            })
            .collect();
        let forecast = forecast_at
            .map(|at_ns| ForecastBatch::new(at_ns, BatchView::ready_only(future.as_slice())));
        let controller =
            Horizon2Controller::new(ControllerConfig::new(max_wait, capacity, 0).unwrap());
        let action = controller.decide(
            0,
            BatchView::ready_only(current.as_slice()),
            forecast,
            &profile,
        );
        let actual = match action {
            ControllerAction::WaitUntil { at_ns } if Some(at_ns) == forecast_at => "wait",
            ControllerAction::DispatchNow => "dispatch",
            other => panic!("case {} left flattened subset: {other:?}", row[0]),
        };
        assert_eq!(actual, row[9], "case {}", row[0]);
    }
}

#[test]
fn reviewed_version_fence_traces() {
    for row in fixture_rows(
        TRANSACTIONS,
        "case\tactions\texpected_results\texpected_state",
    ) {
        let stamp = TransactionStamp::new(Epoch(1), BranchId(1));
        let mut transaction = TransactionMeta::begin(stamp, None, None);
        let mut results = Vec::new();
        for encoded in row[1].split(';') {
            let (action, observed) = encoded.split_once(':').unwrap();
            let observed = Epoch(observed.parse().unwrap());
            let result = match action {
                "prepare" => transaction.prepare(observed, None),
                "commit" => transaction.commit_into(observed, None),
                "abort" => transaction.abort(observed),
                _ => panic!("unknown action {action}"),
            };
            results.push(if result.is_ok() { "ok" } else { "error" });
        }
        assert_eq!(results.join(";"), row[2], "case {}", row[0]);
        let expected_state = match row[3] {
            "committed" => TransactionState::Committed,
            "aborted" => TransactionState::Aborted,
            other => panic!("unknown expected state {other}"),
        };
        assert_eq!(transaction.state(), expected_state, "case {}", row[0]);
    }
}

#[test]
fn every_malformed_fixture_is_rejected() {
    for row in fixture_rows(MALFORMED, "case\tsubsystem\tpayload") {
        let rejected = match row[1] {
            "metrics" => stable_survival_product(&probabilities(row[2])).is_err(),
            "latency" => LatencyProfile::new(latency_points(row[2])).is_err(),
            "horizon2" => {
                let (field, value) = row[2].split_once('=').unwrap();
                let value: u32 = value.parse().unwrap();
                match field {
                    "capacity" => ControllerConfig::new(10, value, 0).is_err(),
                    "service_units" => ReadyItem::new(1, 0, 10, value, Priority::NORMAL).is_err(),
                    _ => panic!("unknown horizon field {field}"),
                }
            }
            other => panic!("unknown subsystem {other}"),
        };
        assert!(rejected, "case {} was unexpectedly accepted", row[0]);
    }
}
