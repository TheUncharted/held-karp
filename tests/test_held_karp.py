import held_karp
import numpy as np
import pytest


def test_solve_four_cities():
    matrix = [
        [0.0, 10.0, 15.0, 20.0],
        [10.0, 0.0, 35.0, 25.0],
        [15.0, 35.0, 0.0, 30.0],
        [20.0, 25.0, 30.0, 0.0],
    ]
    path, cost = held_karp.solve(matrix)
    assert cost == 80.0
    assert path == [0, 1, 3, 2]


def test_solve_flat_four_cities():
    dist = np.array(
        [
            [0.0, 10.0, 15.0, 20.0],
            [10.0, 0.0, 35.0, 25.0],
            [15.0, 35.0, 0.0, 30.0],
            [20.0, 25.0, 30.0, 0.0],
        ]
    )
    path, cost = held_karp.solve_flat(dist.ravel().tolist(), 4)
    assert cost == 80.0
    assert path == [0, 1, 3, 2]


def test_solve_asymmetric():
    matrix = [
        [0.0, 1.0, 100.0],
        [100.0, 0.0, 1.0],
        [1.0, 100.0, 0.0],
    ]
    path, cost = held_karp.solve(matrix)
    assert cost == 3.0
    assert path == [0, 1, 2]


def test_solve_two_nodes():
    path, cost = held_karp.solve([[0.0, 5.0], [3.0, 0.0]])
    assert cost == 8.0
    assert path == [0, 1]


def test_solve_single_node():
    path, cost = held_karp.solve([[0.0]])
    assert cost == 0.0
    assert path == [0]


def test_solve_empty():
    path, cost = held_karp.solve([])
    assert cost == 0.0
    assert path == []


def test_solve_invalid_row_length():
    with pytest.raises(ValueError, match="Expected"):
        held_karp.solve([[0.0, 1.0], [2.0]])


def test_solve_too_many_nodes():
    matrix = [[0.0] * 21 for _ in range(21)]
    with pytest.raises(ValueError, match="exceeds maximum"):
        held_karp.solve(matrix)


def test_solve_flat_dimension_mismatch():
    with pytest.raises(ValueError, match="Expected"):
        held_karp.solve_flat([0.0, 1.0, 2.0], 2)


def test_deterministic_tie_breaking():
    """Ensure deterministic output when multiple tours have the same cost."""
    matrix = [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
    path1, cost1 = held_karp.solve(matrix)
    path2, cost2 = held_karp.solve(matrix)
    assert path1 == path2
    assert cost1 == cost2
