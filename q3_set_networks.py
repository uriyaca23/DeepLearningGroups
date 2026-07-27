"""Question 3 set-network implementations.

External dependency: PyTorch.

This file is developed one approved homework part at a time. It currently
contains the canonization, full group-averaging, and sampled group-averaging
constructions from parts (a), (b), and (c).
"""

from __future__ import annotations

from itertools import permutations
from math import factorial
from typing import Optional

import torch
from torch import Tensor, nn


def _build_two_layer_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    """Build the shared order-sensitive MLP architecture used in Q3(a-c)."""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def lexicographic_sort_rows(x: Tensor) -> Tensor:
    """Return a canonical copy of a 2-D tensor with rows sorted lexicographically.

    Stable sorts are applied from the last feature to the first, so the first
    feature is the primary key, the second feature is the next key, and so on.
    Identical rows need no additional tie-break because exchanging them leaves
    the returned tensor unchanged. Inputs are assumed to contain finite real
    values, as specified by the domain in the assignment.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D tensor of shape (n, d), got {tuple(x.shape)}")

    n, d = x.shape
    if n < 1 or d < 1:
        raise ValueError(f"Expected n >= 1 and d >= 1, got shape {tuple(x.shape)}")

    row_indices = torch.arange(n, device=x.device)
    for feature_index in range(d - 1, -1, -1):
        feature_order = torch.argsort(
            x[row_indices, feature_index],
            stable=True,
        )
        row_indices = row_indices[feature_order]

    return x[row_indices]


class CanonizationInvariantMLP(nn.Module):
    """Invariant network obtained by canonization followed by a two-layer MLP."""

    def __init__(
        self,
        n: int,
        d: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if n < 1 or d < 1 or output_dim < 1:
            raise ValueError("n, d, and output_dim must all be positive")

        self.n = n
        self.d = d
        self.output_dim = output_dim
        input_dim = n * d
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")

        self.hidden_dim = hidden_dim
        self.mlp = _build_two_layer_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or tuple(x.shape) != (self.n, self.d):
            raise ValueError(
                f"Expected input shape {(self.n, self.d)}, got {tuple(x.shape)}"
            )

        canonical_x = lexicographic_sort_rows(x)
        return self.mlp(canonical_x.reshape(-1))


class TwoLayerMLP(nn.Module):
    """Two-layer MLP used as the unrestricted base model h in Question 3(b)."""

    def __init__(
        self,
        n: int,
        d: int,
        output_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        if n < 1 or d < 1 or output_dim < 1 or hidden_dim < 1:
            raise ValueError(
                "n, d, output_dim, and hidden_dim must all be positive"
            )

        self.n = n
        self.d = d
        self.network = _build_two_layer_mlp(
            input_dim=n * d,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2 and tuple(x.shape) == (self.n, self.d):
            return self.network(x.reshape(-1))

        if x.ndim == 3 and tuple(x.shape[1:]) == (self.n, self.d):
            return self.network(x.reshape(x.shape[0], -1))

        raise ValueError(
            f"Expected input shape {(self.n, self.d)} or "
            f"(batch, {self.n}, {self.d}), got {tuple(x.shape)}"
        )


class SymmetrizedInvariantModel(nn.Module):
    """Invariant model obtained by averaging a base model over all row permutations."""

    def __init__(self, base_model: nn.Module, n: int, d: int) -> None:
        super().__init__()

        if n < 1 or d < 1:
            raise ValueError("n and d must both be positive")

        self.base_model = base_model
        self.n = n
        self.d = d
        permutation_indices = torch.tensor(
            list(permutations(range(n))),
            dtype=torch.long,
        )
        self.register_buffer("permutation_indices", permutation_indices)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or tuple(x.shape) != (self.n, self.d):
            raise ValueError(
                f"Expected input shape {(self.n, self.d)}, got {tuple(x.shape)}"
            )

        permuted_inputs = x[self.permutation_indices]
        outputs = self.base_model(permuted_inputs)
        return outputs.mean(dim=0)


class SampledSymmetrizedModel(nn.Module):
    """Approximately invariant model using one fixed random permutation subset."""

    def __init__(
        self,
        base_model: nn.Module,
        n: int,
        d: int,
        subset_size: int,
        subset_seed: int,
    ) -> None:
        super().__init__()

        if n < 1 or d < 1:
            raise ValueError("n and d must both be positive")

        group_size = factorial(n)
        if not 0 < subset_size < group_size:
            raise ValueError(
                f"subset_size must be between 1 and {group_size - 1}"
            )

        self.base_model = base_model
        self.n = n
        self.d = d
        self.subset_size = subset_size
        self.subset_seed = subset_seed

        all_permutation_indices = torch.tensor(
            list(permutations(range(n))),
            dtype=torch.long,
        )
        generator = torch.Generator().manual_seed(subset_seed)
        random_order = torch.randperm(
            group_size,
            generator=generator,
        )
        sampled_indices = all_permutation_indices[
            random_order[:subset_size]
        ]
        self.register_buffer("permutation_indices", sampled_indices)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or tuple(x.shape) != (self.n, self.d):
            raise ValueError(
                f"Expected input shape {(self.n, self.d)}, got {tuple(x.shape)}"
            )

        permuted_inputs = x[self.permutation_indices]
        outputs = self.base_model(permuted_inputs)
        return outputs.mean(dim=0)


def canonization_invariance_max_error(
    *,
    n: int = 7,
    d: int = 3,
    output_dim: int = 4,
    hidden_dim: int = 32,
    seed: int = 2319,
) -> float:
    """Return the Q3(a) maximum coordinate error for one row permutation."""

    torch.manual_seed(seed)

    model = CanonizationInvariantMLP(
        n=n,
        d=d,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    )
    model.eval()
    x = torch.randn(n, d)
    permutation = torch.randperm(n)
    permuted_x = x[permutation]

    with torch.no_grad():
        output = model(x)
        permuted_output = model(permuted_x)

    return float((permuted_output - output).abs().max().item())


def test_canonization_invariance(
    *,
    n: int = 7,
    d: int = 3,
    output_dim: int = 4,
    hidden_dim: int = 32,
    seed: int = 2319,
    atol: float = 1e-5,
) -> bool:
    """Test F(pi . X) = F(X) for the approved Question 3(a) configuration."""

    maximum_error = canonization_invariance_max_error(
        n=n,
        d=d,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    return maximum_error <= atol


def symmetrization_invariance_max_error(
    *,
    n: int = 7,
    d: int = 3,
    output_dim: int = 4,
    hidden_dim: int = 32,
    seed: int = 2319,
) -> float:
    """Return the Q3(b) maximum coordinate error for one row permutation."""

    torch.manual_seed(seed)

    base_model = TwoLayerMLP(
        n=n,
        d=d,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    )
    model = SymmetrizedInvariantModel(base_model=base_model, n=n, d=d)
    model.eval()
    x = torch.randn(n, d)
    permutation = torch.randperm(n)
    permuted_x = x[permutation]

    with torch.no_grad():
        output = model(x)
        permuted_output = model(permuted_x)

    return float((permuted_output - output).abs().max().item())


def test_symmetrization_invariance(
    *,
    n: int = 7,
    d: int = 3,
    output_dim: int = 4,
    hidden_dim: int = 32,
    seed: int = 2319,
    atol: float = 1e-5,
) -> bool:
    """Test F(pi . X) = F(X) for the approved Question 3(b) configuration."""

    maximum_error = symmetrization_invariance_max_error(
        n=n,
        d=d,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    return maximum_error <= atol


def test_full_permutation_group_structure(
    *,
    n: int = 7,
    d: int = 3,
    output_dim: int = 4,
    hidden_dim: int = 32,
) -> bool:
    """Check exhaustively that Q3(b) averages each element of S_n once."""

    model = SymmetrizedInvariantModel(
        base_model=TwoLayerMLP(
            n=n,
            d=d,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        ),
        n=n,
        d=d,
    )
    permutation_indices = model.permutation_indices
    expected_rows = factorial(n)
    expected_entries = torch.arange(n).expand(expected_rows, n)

    return (
        tuple(permutation_indices.shape) == (expected_rows, n)
        and torch.unique(permutation_indices, dim=0).shape[0] == expected_rows
        and torch.equal(
            torch.sort(permutation_indices, dim=1).values,
            expected_entries,
        )
    )


def test_full_symmetrization_gradients(
    *,
    n: int = 7,
    d: int = 3,
    output_dim: int = 4,
    hidden_dim: int = 32,
    seed: int = 2319,
) -> bool:
    """Check that the exact Q3(b) average remains differentiable."""

    torch.manual_seed(seed)
    base_model = TwoLayerMLP(
        n=n,
        d=d,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    )
    model = SymmetrizedInvariantModel(base_model=base_model, n=n, d=d)
    x = torch.randn(n, d)
    model(x).sum().backward()

    return all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in base_model.parameters()
    )


def sampled_symmetrization_max_error(
    *,
    n: int = 7,
    d: int = 3,
    output_dim: int = 4,
    hidden_dim: int = 32,
    subset_size: int = 2700,
    seed: int = 2319,
) -> float:
    """Return the maximum coordinate error for the approved Q3(c) test."""

    torch.manual_seed(seed)

    base_model = TwoLayerMLP(
        n=n,
        d=d,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    )
    model = SampledSymmetrizedModel(
        base_model=base_model,
        n=n,
        d=d,
        subset_size=subset_size,
        subset_seed=seed,
    )
    model.eval()
    x = torch.randn(n, d)
    permutation = torch.randperm(n)
    permuted_x = x[permutation]

    with torch.no_grad():
        output = model(x)
        permuted_output = model(permuted_x)

    return float((permuted_output - output).abs().max().item())


def test_sampled_symmetrization_approximate_invariance(
    *,
    absolute_tolerance: float = 1e-2,
) -> bool:
    """Test approximate invariance using only an absolute error condition."""

    maximum_error = sampled_symmetrization_max_error()
    return maximum_error <= absolute_tolerance


def test_sampled_subset_structure(
    *,
    n: int = 7,
    d: int = 3,
    output_dim: int = 4,
    hidden_dim: int = 32,
    subset_size: int = 2700,
    seed: int = 2319,
) -> bool:
    """Check that the sampled subset is unique, fixed, and reproducible."""

    torch.manual_seed(seed)
    model = SampledSymmetrizedModel(
        base_model=TwoLayerMLP(
            n=n,
            d=d,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        ),
        n=n,
        d=d,
        subset_size=subset_size,
        subset_seed=seed,
    )
    reproduced_model = SampledSymmetrizedModel(
        base_model=TwoLayerMLP(
            n=n,
            d=d,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        ),
        n=n,
        d=d,
        subset_size=subset_size,
        subset_seed=seed,
    )

    original_subset = model.permutation_indices.clone()
    x = torch.randn(n, d)
    with torch.no_grad():
        model(x)
        model(x)

    return (
        tuple(original_subset.shape) == (subset_size, n)
        and torch.unique(original_subset, dim=0).shape[0] == subset_size
        and torch.equal(model.permutation_indices, original_subset)
        and torch.equal(
            reproduced_model.permutation_indices,
            original_subset,
        )
    )


def test_lexicographic_sort_with_ties() -> bool:
    """Check canonization when early coordinates tie and rows repeat."""

    x = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [1.0, 1.0, 5.0],
            [0.0, 9.0, 9.0],
            [1.0, 1.0, 4.0],
            [1.0, 2.0, 0.0],
            [-1.0, 3.0, 2.0],
            [1.0, 1.0, 4.0],
        ]
    )
    expected = torch.tensor(
        [
            [-1.0, 3.0, 2.0],
            [0.0, 9.0, 9.0],
            [1.0, 1.0, 4.0],
            [1.0, 1.0, 4.0],
            [1.0, 1.0, 5.0],
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )
    permutation = torch.tensor([6, 4, 2, 0, 5, 3, 1])

    canonical_x = lexicographic_sort_rows(x)
    canonical_permuted_x = lexicographic_sort_rows(x[permutation])
    return torch.equal(canonical_x, expected) and torch.equal(
        canonical_permuted_x,
        expected,
    )


def test_canonization_over_all_permutations(
    *,
    n: int = 7,
    d: int = 3,
    seed: int = 2319,
) -> bool:
    """Check the Q3(a) canonical representative over all n! row orders."""

    torch.manual_seed(seed)
    x = torch.randn(n, d)
    expected = lexicographic_sort_rows(x)

    return all(
        torch.equal(lexicographic_sort_rows(x[list(permutation)]), expected)
        for permutation in permutations(range(n))
    )


def run_q3a_tests() -> bool:
    """Run and print all current Question 3(a) checks."""

    absolute_tolerance = 1e-5
    maximum_error = canonization_invariance_max_error()
    invariance_passed = maximum_error <= absolute_tolerance
    exhaustive_passed = test_canonization_over_all_permutations()
    ties_passed = test_lexicographic_sort_with_ties()

    print(
        "Q3(a) invariance test:",
        "PASS" if invariance_passed else "FAIL",
        "(seed=2319, n=7, d=3, p=4, hidden=32, atol=1e-5, rtol=0, "
        f"max_abs_error={maximum_error:.12g})",
    )
    print(
        "Q3(a) exhaustive canonization test over all 5040 row permutations:",
        "PASS" if exhaustive_passed else "FAIL",
    )
    print(
        "Q3(a) lexicographic tie test:",
        "PASS" if ties_passed else "FAIL",
    )
    return invariance_passed and exhaustive_passed and ties_passed


def run_q3b_tests() -> bool:
    """Run and print all current Question 3(b) checks."""

    absolute_tolerance = 1e-5
    maximum_error = symmetrization_invariance_max_error()
    invariance_passed = maximum_error <= absolute_tolerance
    group_structure_passed = test_full_permutation_group_structure()
    gradients_passed = test_full_symmetrization_gradients()

    print(
        "Q3(b) invariance test:",
        "PASS" if invariance_passed else "FAIL",
        "(seed=2319, n=7, d=3, p=4, hidden=32, permutations=5040, "
        f"atol=1e-5, rtol=0, max_abs_error={maximum_error:.12g})",
    )
    print(
        "Q3(b) exhaustive S_7 structure test (5040 unique permutations):",
        "PASS" if group_structure_passed else "FAIL",
    )
    print(
        "Q3(b) finite-gradient test:",
        "PASS" if gradients_passed else "FAIL",
    )
    return invariance_passed and group_structure_passed and gradients_passed


def run_q3c_tests() -> bool:
    """Run and print the approved Question 3(c) approximate-invariance check."""

    absolute_tolerance = 1e-2
    maximum_error = sampled_symmetrization_max_error()
    invariance_passed = maximum_error <= absolute_tolerance
    subset_structure_passed = test_sampled_subset_structure()

    print(
        "Q3(c) sampled approximate-invariance test:",
        "PASS" if invariance_passed else "FAIL",
        "(seed=2319, n=7, d=3, p=4, hidden=32, B=2700, "
        f"absolute_tolerance=1e-2, max_abs_error={maximum_error:.12g})",
    )
    print(
        "Q3(c) fixed, unique, reproducible subset test:",
        "PASS" if subset_structure_passed else "FAIL",
    )
    return invariance_passed and subset_structure_passed


def run_q3_tests() -> bool:
    """Run all implemented Question 3 checks."""

    q3a_passed = run_q3a_tests()
    q3b_passed = run_q3b_tests()
    q3c_passed = run_q3c_tests()
    return q3a_passed and q3b_passed and q3c_passed


if __name__ == "__main__":
    raise SystemExit(0 if run_q3_tests() else 1)
