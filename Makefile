.PHONY: build test lint format clean dev

build:
	cargo build

build-release:
	cargo build --release

test:
	cargo test

test-python:
	cd crates/held-karp-py && maturin develop --release
	pytest tests/ -v

lint:
	cargo clippy --all-targets -- -D warnings
	cargo fmt -- --check

format:
	cargo fmt

clean:
	cargo clean

dev:
	cd crates/held-karp-py && maturin develop --release
