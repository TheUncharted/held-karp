# held-karp

Exact TSP solver using the **Held-Karp bitmask dynamic programming algorithm**, implemented in Rust with Python bindings.

## Features

- **Exact optimal solution** — guaranteed shortest round-trip tour
- **Fast** — Rust implementation, 50-300× faster than pure Python
- **Deterministic** — same input always produces the same output
- **Asymmetric support** — works with non-symmetric distance matrices
- **Dual interface** — use from Rust (crates.io) or Python (PyPI)

## Complexity

| | |
|---|---|
| **Time** | O(n² · 2ⁿ) |
| **Space** | O(n · 2ⁿ) |
| **Max nodes** | 20 |

For typical use cases (n=4–8), runs in **microseconds**.

## Installation

### Python

```bash
pip install held-karp
```

### Rust

```toml
[dependencies]
held-karp-core = "0.1"
```

## Usage

### Python

```python
import held_karp

# From a 2D distance matrix
matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]
path, cost = held_karp.solve(matrix)
print(f"Path: {path}, Cost: {cost}")
# Path: [0, 1, 3, 2], Cost: 80.0

# From a flat numpy array (faster, avoids nested list conversion)
import numpy as np
dist = np.array(matrix, dtype=float)
path, cost = held_karp.solve_flat(dist.ravel().tolist(), len(dist))
```

### Rust

```rust
use held_karp_core::solve;

let dist = vec![
    0.0, 10.0, 15.0, 20.0,
    10.0,  0.0, 35.0, 25.0,
    15.0, 35.0,  0.0, 30.0,
    20.0, 25.0, 30.0,  0.0,
];

let (path, cost) = solve(&dist, 4).unwrap();
assert_eq!(path, vec![0, 1, 3, 2]);
assert_eq!(cost, 80.0);
```

## API Reference

### Python

#### `held_karp.solve(distance_matrix) -> (list[int], float)`

Solve TSP from a list of lists (n × n).

**Args:**
- `distance_matrix`: Square matrix where `[i][j]` is the distance from node `i` to node `j`

**Returns:** `(path, cost)` — optimal node order (starting from 0) and total round-trip distance

#### `held_karp.solve_flat(flat_matrix, n) -> (list[int], float)`

Solve TSP from a flat row-major array of `n * n` floats. Useful with numpy arrays.

### Rust

#### `held_karp_core::solve(dist, n) -> Result<(Vec<usize>, f64), SolveError>`

Solve from a flat row-major slice.

#### `held_karp_core::solve_matrix(matrix) -> Result<(Vec<usize>, f64), SolveError>`

Solve from a 2D Vec.

## Performance

Benchmarked against a pure Python Held-Karp implementation:

| n (cities) | Python | Rust | Speedup |
|---|---|---|---|
| 4 | 12 µs | 0.23 µs | **52×** |
| 6 | 155 µs | 1.3 µs | **119×** |
| 8 | 2.4 ms | 8.1 µs | **296×** |

## License

MIT
