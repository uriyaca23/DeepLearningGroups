"""Question 3 set-network implementations.

External dependency: PyTorch.

This file is developed one approved homework part at a time. It currently
contains the canonization and group-averaging constructions from parts (a)
and (b).
"""

from __future__ import annotations

from itertools import permutations
from typing import Optional

import torch
from torch import Tensor, nn


def _build_two_layer_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    """Build the shared order-sensitive MLP architecture used in Q3(a-b)."""

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
        if x.ndim != 2 or tuple(x.shape) != (self.n, self.d):
            raise ValueError(
                f"Expected input shape {(self.n, self.d)}, got {tuple(x.shape)}"
            )

        return self.network(x.reshape(-1))


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

        outputs = [
            self.base_model(x[permutation])
            for permutation in self.permutation_indices
        ]
        return torch.stack(outputs).mean(dim=0)


def test_canonization_invariance(
    *,
    n: int = 5,
    d: int = 3,
    output_dim: int = 4,
    hidden_dim: int = 32,
    seed: int = 2319,
    atol: float = 1e-5,
) -> bool:
    """Test F(pi . X) = F(X) for the approved Question 3(a) configuration."""

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

    return torch.allclose(
        permuted_output,
        output,
        atol=atol,
        rtol=0.0,
    )


def test_symmetrization_invariance(
    *,
    n: int = 5,
    d: int = 3,
    output_dim: int = 4,
    hidden_dim: int = 32,
    seed: int = 2319,
    atol: float = 1e-5,
) -> bool:
    """Test F(pi . X) = F(X) for the approved Question 3(b) configuration."""

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

    return torch.allclose(
        permuted_output,
        output,
        atol=atol,
        rtol=0.0,
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
        ]
    )
    expected = torch.tensor(
        [
            [0.0, 9.0, 9.0],
            [1.0, 1.0, 4.0],
            [1.0, 1.0, 5.0],
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )
    permutation = torch.tensor([4, 2, 0, 3, 1])

    canonical_x = lexicographic_sort_rows(x)
    canonical_permuted_x = lexicographic_sort_rows(x[permutation])
    return torch.equal(canonical_x, expected) and torch.equal(
        canonical_permuted_x,
        expected,
    )


def run_q3a_tests() -> bool:
    """Run and print all current Question 3(a) checks."""

    invariance_passed = test_canonization_invariance()
    ties_passed = test_lexicographic_sort_with_ties()

    print(
        "Q3(a) invariance test:",
        "PASS" if invariance_passed else "FAIL",
        "(seed=2319, n=5, d=3, p=4, hidden=32, atol=1e-5, rtol=0)",
    )
    print(
        "Q3(a) lexicographic tie test:",
        "PASS" if ties_passed else "FAIL",
    )
    return invariance_passed and ties_passed


def run_q3b_tests() -> bool:
    """Run and print all current Question 3(b) checks."""

    invariance_passed = test_symmetrization_invariance()

    print(
        "Q3(b) invariance test:",
        "PASS" if invariance_passed else "FAIL",
        "(seed=2319, n=5, d=3, p=4, hidden=32, permutations=120, "
        "atol=1e-5, rtol=0)",
    )
    return invariance_passed


def run_q3_tests() -> bool:
    """Run all implemented Question 3 checks."""

    q3a_passed = run_q3a_tests()
    q3b_passed = run_q3b_tests()
    return q3a_passed and q3b_passed


if __name__ == "__main__":
    raise SystemExit(0 if run_q3_tests() else 1)
