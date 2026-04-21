use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn convert_error(e: held_karp_core::SolveError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Solve TSP to optimality using the Held-Karp bitmask DP algorithm.
///
/// Time complexity: O(n² · 2ⁿ)
/// Space complexity: O(n · 2ⁿ)
///
/// Supports up to n=20 nodes. For typical use cases (n=4–8), runs in
/// microseconds.
///
/// Args:
///     distance_matrix: A list of lists (n × n) where entry [i][j] is the
///         distance from node i to node j. Does not need to be symmetric.
///
/// Returns:
///     A tuple of (permutation, distance) where permutation is a list of
///     node indices starting from 0, and distance is the total tour cost.
///
/// Example:
///     >>> import held_karp
///     >>> path, cost = held_karp.solve([[0, 10, 15], [10, 0, 35], [15, 35, 0]])
///     >>> print(f"Path: {path}, Cost: {cost}")
#[pyfunction]
fn solve(distance_matrix: Vec<Vec<f64>>) -> PyResult<(Vec<usize>, f64)> {
    held_karp_core::solve_matrix(&distance_matrix).map_err(convert_error)
}

/// Solve TSP from a flat array (row-major, n×n elements).
///
/// This avoids the overhead of converting numpy 2D arrays to nested lists.
///
/// Args:
///     flat_matrix: A flat list of n×n floats in row-major order.
///     n: The number of nodes.
///
/// Returns:
///     A tuple of (permutation, distance).
///
/// Example:
///     >>> import held_karp
///     >>> import numpy as np
///     >>> dist = np.array([[0, 10, 15], [10, 0, 35], [15, 35, 0]], dtype=float)
///     >>> path, cost = held_karp.solve_flat(dist.ravel().tolist(), 3)
#[pyfunction]
fn solve_flat(flat_matrix: Vec<f64>, n: usize) -> PyResult<(Vec<usize>, f64)> {
    held_karp_core::solve(&flat_matrix, n).map_err(convert_error)
}

/// held_karp — Exact TSP solver using the Held-Karp bitmask DP algorithm.
///
/// Provides `solve()` and `solve_flat()` for finding the shortest
/// round-trip tour through all nodes.
#[pymodule]
fn held_karp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_flat, m)?)?;
    Ok(())
}
