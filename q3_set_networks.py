"""Question 3 set-network implementations.

External dependency: PyTorch.

This file is developed one approved homework part at a time. It currently
contains only the canonization-based invariant network from Question 3(a).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


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
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or tuple(x.shape) != (self.n, self.d):
            raise ValueError(
                f"Expected input shape {(self.n, self.d)}, got {tuple(x.shape)}"
            )

        canonical_x = lexicographic_sort_rows(x)
        return self.mlp(canonical_x.reshape(-1))


def test_canonization_invariance(
    *,
    n: int = 20,
    d: int = 3,
    output_dim: int = 4,
    seed: int = 2319,
    atol: float = 1e-5,
) -> bool:
    """Test F(pi . X) = F(X) for the approved Question 3(a) configuration."""

    torch.manual_seed(seed)

    model = CanonizationInvariantMLP(n=n, d=d, output_dim=output_dim)
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
        "(seed=2319, n=20, d=3, p=4, atol=1e-5, rtol=0)",
    )
    print(
        "Q3(a) lexicographic tie test:",
        "PASS" if ties_passed else "FAIL",
    )
    return invariance_passed and ties_passed


if __name__ == "__main__":
    raise SystemExit(0 if run_q3a_tests() else 1)
