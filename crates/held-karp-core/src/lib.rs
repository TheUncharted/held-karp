//! # held-karp-core
//!
//! Exact TSP solver using the Held-Karp bitmask dynamic programming algorithm.
//!
//! ## Complexity
//!
//! - **Time:** O(n² · 2ⁿ)
//! - **Space:** O(n · 2ⁿ)
//!
//! Supports up to n=20 nodes. For typical use cases (n=4–8), runs in
//! microseconds.
//!
//! ## Example
//!
//! ```
//! use held_karp_core::solve;
//!
//! // 4-city distance matrix (row-major flat array)
//! let dist = vec![
//!     0.0, 10.0, 15.0, 20.0,
//!     10.0,  0.0, 35.0, 25.0,
//!     15.0, 35.0,  0.0, 30.0,
//!     20.0, 25.0, 30.0,  0.0,
//! ];
//!
//! let (path, cost) = solve(&dist, 4).unwrap();
//! assert_eq!(cost, 80.0);
//! assert_eq!(path, vec![0, 1, 3, 2]);
//! ```

/// Error type for solver failures.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveError {
    /// The distance matrix length does not match n×n.
    DimensionMismatch { expected: usize, got: usize },
    /// n exceeds the maximum supported size (20).
    TooManyNodes { n: usize },
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::DimensionMismatch { expected, got } => {
                write!(f, "Expected {} elements, got {}", expected, got)
            }
            SolveError::TooManyNodes { n } => {
                write!(f, "n={} exceeds maximum of 20 nodes", n)
            }
        }
    }
}

impl std::error::Error for SolveError {}

/// Core Held-Karp solver using top-down recursive DP with memoization.
///
/// Iterates candidate next-nodes in ascending index order and picks the
/// first minimum on ties, producing deterministic results.
fn held_karp_inner(dist: &[f64], n: usize) -> (Vec<usize>, f64) {
    let num_sets: usize = 1 << n;
    let table_size = n * num_sets;

    // dp[ni * num_sets + mask] = min cost from ni, visiting remaining set
    // (represented by mask), then returning to node 0.
    // Sentinel -1.0 means "not yet computed" (valid since all costs >= 0).
    let mut dp = vec![-1.0f64; table_size];
    // choice[ni * num_sets + mask] = best next node from ni given mask
    let mut choice = vec![0usize; table_size];

    // Initial mask: all nodes except node 0
    let initial_mask = (num_sets - 1) & !1usize;

    fn recurse(
        ni: usize,
        mask: usize,
        dist: &[f64],
        n: usize,
        num_sets: usize,
        dp: &mut [f64],
        choice: &mut [usize],
    ) -> f64 {
        if mask == 0 {
            return dist[ni * n]; // dist[ni][0] — return to start
        }
        let key = ni * num_sets + mask;
        if dp[key] >= 0.0 {
            return dp[key];
        }

        let mut min_cost = f64::INFINITY;
        let mut best_nj: usize = 0;

        // Iterate nj in ascending order (1, 2, ..., n-1) for deterministic
        // tie-breaking (first/lowest index wins).
        for nj in 1..n {
            if mask & (1 << nj) == 0 {
                continue;
            }
            let cost =
                dist[ni * n + nj] + recurse(nj, mask ^ (1 << nj), dist, n, num_sets, dp, choice);
            if cost < min_cost {
                min_cost = cost;
                best_nj = nj;
            }
        }

        dp[key] = min_cost;
        choice[key] = best_nj;
        min_cost
    }

    let best_distance = recurse(0, initial_mask, dist, n, num_sets, &mut dp, &mut choice);

    // Reconstruct path
    let mut path = vec![0usize; n];
    let mut ni = 0;
    let mut mask = initial_mask;
    #[allow(clippy::needless_range_loop)]
    for pos in 1..n {
        let nj = choice[ni * num_sets + mask];
        path[pos] = nj;
        mask ^= 1 << nj;
        ni = nj;
    }

    (path, best_distance)
}

/// Solve TSP to optimality given a flat (row-major) distance matrix.
///
/// # Arguments
///
/// * `dist` — A flat slice of `n * n` floats in row-major order, where
///   `dist[i * n + j]` is the distance from node `i` to node `j`.
///   Does not need to be symmetric.
/// * `n` — The number of nodes.
///
/// # Returns
///
/// A `(path, cost)` tuple where `path` is the optimal node visit order
/// (always starting with 0) and `cost` is the total round-trip distance.
///
/// # Errors
///
/// Returns [`SolveError::DimensionMismatch`] if `dist.len() != n * n`, or
/// [`SolveError::TooManyNodes`] if `n > 20`.
pub fn solve(dist: &[f64], n: usize) -> Result<(Vec<usize>, f64), SolveError> {
    if dist.len() != n * n {
        return Err(SolveError::DimensionMismatch {
            expected: n * n,
            got: dist.len(),
        });
    }

    match n {
        0 => Ok((vec![], 0.0)),
        1 => Ok((vec![0], 0.0)),
        2 => {
            let d = dist[1] + dist[n]; // dist[0][1] + dist[1][0]
            Ok((vec![0, 1], d))
        }
        3..=20 => Ok(held_karp_inner(dist, n)),
        _ => Err(SolveError::TooManyNodes { n }),
    }
}

/// Solve TSP from a 2D distance matrix (Vec of rows).
///
/// Convenience wrapper that flattens the matrix and calls [`solve`].
///
/// # Errors
///
/// Returns [`SolveError`] if any row length differs from `n`, or if `n > 20`.
pub fn solve_matrix(matrix: &[Vec<f64>]) -> Result<(Vec<usize>, f64), SolveError> {
    let n = matrix.len();
    for row in matrix {
        if row.len() != n {
            return Err(SolveError::DimensionMismatch {
                expected: n,
                got: row.len(),
            });
        }
    }
    let flat: Vec<f64> = matrix.iter().flatten().copied().collect();
    solve(&flat, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let (path, cost) = solve(&[], 0).unwrap();
        assert_eq!(path, vec![]);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_single_node() {
        let (path, cost) = solve(&[0.0], 1).unwrap();
        assert_eq!(path, vec![0]);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_two_nodes() {
        let dist = vec![0.0, 5.0, 3.0, 0.0];
        let (path, cost) = solve(&dist, 2).unwrap();
        assert_eq!(path, vec![0, 1]);
        assert_eq!(cost, 8.0);
    }

    #[test]
    fn test_four_cities() {
        let dist = vec![
            0.0, 10.0, 15.0, 20.0, 10.0, 0.0, 35.0, 25.0, 15.0, 35.0, 0.0, 30.0, 20.0, 25.0, 30.0,
            0.0,
        ];
        let (path, cost) = solve(&dist, 4).unwrap();
        assert_eq!(cost, 80.0);
        assert_eq!(path, vec![0, 1, 3, 2]);
    }

    #[test]
    fn test_asymmetric() {
        let dist = vec![0.0, 1.0, 100.0, 100.0, 0.0, 1.0, 1.0, 100.0, 0.0];
        let (path, cost) = solve(&dist, 3).unwrap();
        assert_eq!(cost, 3.0);
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = solve(&[0.0, 1.0, 2.0], 2).unwrap_err();
        assert_eq!(
            err,
            SolveError::DimensionMismatch {
                expected: 4,
                got: 3
            }
        );
    }

    #[test]
    fn test_too_many_nodes() {
        let err = solve(&vec![0.0; 21 * 21], 21).unwrap_err();
        assert_eq!(err, SolveError::TooManyNodes { n: 21 });
    }

    #[test]
    fn test_solve_matrix() {
        let matrix = vec![
            vec![0.0, 10.0, 15.0, 20.0],
            vec![10.0, 0.0, 35.0, 25.0],
            vec![15.0, 35.0, 0.0, 30.0],
            vec![20.0, 25.0, 30.0, 0.0],
        ];
        let (path, cost) = solve_matrix(&matrix).unwrap();
        assert_eq!(cost, 80.0);
        assert_eq!(path, vec![0, 1, 3, 2]);
    }

    #[test]
    fn test_five_cities() {
        // Known optimal for this instance
        let dist = vec![
            0.0, 3.0, 4.0, 2.0, 7.0, 3.0, 0.0, 4.0, 6.0, 3.0, 4.0, 4.0, 0.0, 5.0, 8.0, 2.0, 6.0,
            5.0, 0.0, 6.0, 7.0, 3.0, 8.0, 6.0, 0.0,
        ];
        let (path, cost) = solve(&dist, 5).unwrap();
        assert_eq!(cost, 19.0);
        assert_eq!(path[0], 0);
    }
}
