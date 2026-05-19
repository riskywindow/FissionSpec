.PHONY: test lint typecheck check simulate sweep phase artifacts rust-test rust-bench clean

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v
	PYTHONPATH=src python3 -m unittest discover -s experiments -p 'test_*.py' -v

lint:
	ruff check src tests experiments tools
	ruff format --check src tests experiments tools

typecheck:
	PYTHONPATH=src mypy src/fissionspec experiments tools

check: test lint typecheck rust-test
	PYTHONPATH=src python3 -m compileall -q src tests experiments

simulate:
	PYTHONPATH=src python3 -m fissionspec simulate --policy fissionspec

sweep:
	PYTHONPATH=src python3 experiments/run_synthetic_sweep.py
	PYTHONPATH=src python3 tools/render_synthetic_results.py \
		experiments/results/synthetic_sweep.json

phase:
	PYTHONPATH=src python3 experiments/run_controller_phase_diagram.py

artifacts: sweep phase

rust-test:
	cargo test --manifest-path crates/fissionspec-core/Cargo.toml
	cargo clippy --manifest-path crates/fissionspec-core/Cargo.toml --all-targets -- -D warnings
	cargo fmt --manifest-path crates/fissionspec-core/Cargo.toml --check

rust-bench:
	cargo run --release --manifest-path crates/fissionspec-core/Cargo.toml --bin decision-bench

clean:
	cargo clean --manifest-path crates/fissionspec-core/Cargo.toml
