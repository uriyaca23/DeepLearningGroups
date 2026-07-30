"""Question 3 set-network implementations.

External dependency: PyTorch.

This file is developed one approved homework part at a time. It currently
contains the canonization, full group-averaging, sampled group-averaging, and
DeepSets constructions from parts (a), (b), (c), and (d).
"""

from __future__ import annotations

from itertools import permutations
from math import factorial
from typing import Optional

import torch
from torch import Tensor, nn


DEFAULT_N = 7
DEFAULT_D = 3
DEFAULT_OUTPUT_DIM = DEFAULT_D
DEFAULT_MLP_HIDDEN_DIM = 32
DEFAULT_DEEPSETS_HIDDEN_DIM = 62
DEFAULT_SUBSET_SIZE = 2700
DEFAULT_SEED = 2319


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


class DeepSetsEquivariantLinear(nn.Module):
    """S_n-equivariant linear layer with local and mean-aggregated terms."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must both be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.local_linear = nn.Linear(input_dim, output_dim, bias=True)
        self.global_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                "Expected shape (..., n, "
                f"{self.input_dim}), got {tuple(x.shape)}"
            )
        if x.shape[-2] < 1:
            raise ValueError("The row dimension n must be positive")

        row_mean = x.mean(dim=-2, keepdim=True)
        return self.local_linear(x) + self.global_linear(row_mean)


class DeepSetsEquivariantNetwork(nn.Module):
    """Two equivariant DeepSets layers with a pointwise ReLU between them."""

    def __init__(
        self,
        input_dim: int = DEFAULT_D,
        hidden_dim: int = DEFAULT_DEEPSETS_HIDDEN_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
    ) -> None:
        super().__init__()

        self.first_layer = DeepSetsEquivariantLinear(
            input_dim=input_dim,
            output_dim=hidden_dim,
        )
        self.activation = nn.ReLU()
        self.second_layer = DeepSetsEquivariantLinear(
            input_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.second_layer(self.activation(self.first_layer(x)))


class MeanPooledInvariantModel(nn.Module):
    """Invariant output obtained by mean-pooling equivariant row features."""

    def __init__(self, equivariant_model: nn.Module) -> None:
        super().__init__()
        self.equivariant_model = equivariant_model

    def forward(self, x: Tensor) -> Tensor:
        equivariant_output = self.equivariant_model(x)
        if equivariant_output.ndim < 2 or equivariant_output.shape[-2] < 1:
            raise ValueError(
                "Expected the equivariant model to return shape "
                "(..., n, output_dim) with n >= 1"
            )
        return equivariant_output.mean(dim=-2)


def canonization_invariance_max_error(
    *,
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    seed: int = DEFAULT_SEED,
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
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    seed: int = DEFAULT_SEED,
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
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    seed: int = DEFAULT_SEED,
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
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    seed: int = DEFAULT_SEED,
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
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
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
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    seed: int = DEFAULT_SEED,
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
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_SEED,
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
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    subset_size: int = DEFAULT_SUBSET_SIZE,
    seed: int = DEFAULT_SEED,
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


def deepsets_single_layer_equivariance_max_error(
    *,
    n: int = DEFAULT_N,
    input_dim: int = DEFAULT_D,
    output_dim: int = DEFAULT_DEEPSETS_HIDDEN_DIM,
    seed: int = DEFAULT_SEED,
) -> float:
    """Return the maximum equivariance error of one DeepSets linear layer."""

    torch.manual_seed(seed)
    layer = DeepSetsEquivariantLinear(
        input_dim=input_dim,
        output_dim=output_dim,
    )
    layer.eval()
    x = torch.randn(n, input_dim)
    permutation = torch.randperm(n)

    with torch.no_grad():
        transformed_output = layer(x[permutation])
        permuted_output = layer(x)[permutation]

    return float(
        (transformed_output - permuted_output).abs().max().item()
    )


def deepsets_stack_equivariance_max_error(
    *,
    n: int = DEFAULT_N,
    input_dim: int = DEFAULT_D,
    hidden_dim: int = DEFAULT_DEEPSETS_HIDDEN_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    seed: int = DEFAULT_SEED,
) -> float:
    """Return the maximum equivariance error of the complete DeepSets stack."""

    torch.manual_seed(seed)
    model = DeepSetsEquivariantNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    model.eval()
    x = torch.randn(n, input_dim)
    permutation = torch.randperm(n)

    with torch.no_grad():
        transformed_output = model(x[permutation])
        permuted_output = model(x)[permutation]

    return float(
        (transformed_output - permuted_output).abs().max().item()
    )


def deepsets_pooled_invariance_max_error(
    *,
    n: int = DEFAULT_N,
    input_dim: int = DEFAULT_D,
    hidden_dim: int = DEFAULT_DEEPSETS_HIDDEN_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    seed: int = DEFAULT_SEED,
) -> float:
    """Return the maximum invariance error after mean pooling."""

    torch.manual_seed(seed)
    model = MeanPooledInvariantModel(
        DeepSetsEquivariantNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
    )
    model.eval()
    x = torch.randn(n, input_dim)
    permutation = torch.randperm(n)

    with torch.no_grad():
        output = model(x)
        permuted_output = model(x[permutation])

    return float((permuted_output - output).abs().max().item())


def test_deepsets_finite_gradients(
    *,
    n: int = DEFAULT_N,
    input_dim: int = DEFAULT_D,
    hidden_dim: int = DEFAULT_DEEPSETS_HIDDEN_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    seed: int = DEFAULT_SEED,
) -> bool:
    """Check finite parameter and input gradients through mean pooling."""

    torch.manual_seed(seed)
    model = MeanPooledInvariantModel(
        DeepSetsEquivariantNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
    )
    x = torch.randn(n, input_dim, requires_grad=True)
    model(x).square().sum().backward()

    parameter_gradients_are_finite = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    input_gradient_is_finite = (
        x.grad is not None and torch.isfinite(x.grad).all()
    )
    return bool(parameter_gradients_are_finite and input_gradient_is_finite)


def q3d_parameter_counts(
    *,
    n: int = DEFAULT_N,
    input_dim: int = DEFAULT_D,
    hidden_dim: int = DEFAULT_DEEPSETS_HIDDEN_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    mlp_hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
) -> tuple[int, int]:
    """Return trainable parameter counts for DeepSets and the ordinary MLP."""

    deepsets_model = DeepSetsEquivariantNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    ordinary_mlp = TwoLayerMLP(
        n=n,
        d=input_dim,
        output_dim=output_dim,
        hidden_dim=mlp_hidden_dim,
    )

    deepsets_count = sum(
        parameter.numel()
        for parameter in deepsets_model.parameters()
        if parameter.requires_grad
    )
    ordinary_mlp_count = sum(
        parameter.numel()
        for parameter in ordinary_mlp.parameters()
        if parameter.requires_grad
    )
    return deepsets_count, ordinary_mlp_count


def test_shared_output_dimension_configuration() -> bool:
    """Check that every Q3 architecture uses p=d for the shared experiment."""

    canonization_model = CanonizationInvariantMLP(
        n=DEFAULT_N,
        d=DEFAULT_D,
        output_dim=DEFAULT_OUTPUT_DIM,
        hidden_dim=DEFAULT_MLP_HIDDEN_DIM,
    )
    ordinary_mlp = TwoLayerMLP(
        n=DEFAULT_N,
        d=DEFAULT_D,
        output_dim=DEFAULT_OUTPUT_DIM,
        hidden_dim=DEFAULT_MLP_HIDDEN_DIM,
    )
    deepsets_model = DeepSetsEquivariantNetwork()

    return (
        DEFAULT_OUTPUT_DIM == DEFAULT_D
        and canonization_model.mlp[-1].out_features == DEFAULT_D
        and ordinary_mlp.network[-1].out_features == DEFAULT_D
        and deepsets_model.second_layer.output_dim == DEFAULT_D
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
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
    seed: int = DEFAULT_SEED,
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
        f"(seed={DEFAULT_SEED}, n={DEFAULT_N}, d={DEFAULT_D}, "
        f"p={DEFAULT_OUTPUT_DIM}, hidden={DEFAULT_MLP_HIDDEN_DIM}, "
        "atol=1e-5, rtol=0, "
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
        f"(seed={DEFAULT_SEED}, n={DEFAULT_N}, d={DEFAULT_D}, "
        f"p={DEFAULT_OUTPUT_DIM}, hidden={DEFAULT_MLP_HIDDEN_DIM}, "
        f"permutations={factorial(DEFAULT_N)}, "
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
        f"(seed={DEFAULT_SEED}, n={DEFAULT_N}, d={DEFAULT_D}, "
        f"p={DEFAULT_OUTPUT_DIM}, hidden={DEFAULT_MLP_HIDDEN_DIM}, "
        f"B={DEFAULT_SUBSET_SIZE}, "
        f"absolute_tolerance=1e-2, max_abs_error={maximum_error:.12g})",
    )
    print(
        "Q3(c) fixed, unique, reproducible subset test:",
        "PASS" if subset_structure_passed else "FAIL",
    )
    return invariance_passed and subset_structure_passed


def run_q3d_tests() -> bool:
    """Run the approved Question 3(d) equivariance and invariance checks."""

    absolute_tolerance = 1e-5
    single_layer_error = deepsets_single_layer_equivariance_max_error()
    stack_error = deepsets_stack_equivariance_max_error()
    pooled_error = deepsets_pooled_invariance_max_error()
    single_layer_passed = single_layer_error <= absolute_tolerance
    stack_passed = stack_error <= absolute_tolerance
    pooled_passed = pooled_error <= absolute_tolerance
    gradients_passed = test_deepsets_finite_gradients()
    deepsets_count, ordinary_mlp_count = q3d_parameter_counts()
    expected_deepsets_count = (
        (2 * DEFAULT_D * DEFAULT_DEEPSETS_HIDDEN_DIM)
        + DEFAULT_DEEPSETS_HIDDEN_DIM
        + (2 * DEFAULT_DEEPSETS_HIDDEN_DIM * DEFAULT_OUTPUT_DIM)
        + DEFAULT_OUTPUT_DIM
    )
    expected_mlp_count = (
        (DEFAULT_N * DEFAULT_D * DEFAULT_MLP_HIDDEN_DIM)
        + DEFAULT_MLP_HIDDEN_DIM
        + (DEFAULT_MLP_HIDDEN_DIM * DEFAULT_OUTPUT_DIM)
        + DEFAULT_OUTPUT_DIM
    )
    parameter_counts_passed = (
        deepsets_count == expected_deepsets_count == 809
        and ordinary_mlp_count == expected_mlp_count == 803
    )

    print(
        "Q3(d) single-layer equivariance test:",
        "PASS" if single_layer_passed else "FAIL",
        f"(seed={DEFAULT_SEED}, n={DEFAULT_N}, "
        f"{DEFAULT_D}->{DEFAULT_DEEPSETS_HIDDEN_DIM}, "
        "atol=1e-5, rtol=0, "
        f"max_abs_error={single_layer_error:.12g})",
    )
    print(
        "Q3(d) full-stack equivariance test:",
        "PASS" if stack_passed else "FAIL",
        f"(seed={DEFAULT_SEED}, n={DEFAULT_N}, "
        f"{DEFAULT_D}->{DEFAULT_DEEPSETS_HIDDEN_DIM}"
        f"->{DEFAULT_OUTPUT_DIM}, pointwise ReLU, "
        f"atol=1e-5, rtol=0, max_abs_error={stack_error:.12g})",
    )
    print(
        "Q3(d) mean-pooled invariance test:",
        "PASS" if pooled_passed else "FAIL",
        f"(seed={DEFAULT_SEED}, n={DEFAULT_N}, "
        f"output_dim={DEFAULT_OUTPUT_DIM}, atol=1e-5, rtol=0, "
        f"max_abs_error={pooled_error:.12g})",
    )
    print(
        "Q3(d) finite-gradient test:",
        "PASS" if gradients_passed else "FAIL",
    )
    print(
        "Q3(d) exact parameter-count test:",
        "PASS" if parameter_counts_passed else "FAIL",
        f"(DeepSets={deepsets_count}, ordinary_n7_MLP={ordinary_mlp_count})",
    )
    return (
        single_layer_passed
        and stack_passed
        and pooled_passed
        and gradients_passed
        and parameter_counts_passed
    )


def run_q3_tests() -> bool:
    """Run all implemented Question 3 checks."""

    q3a_passed = run_q3a_tests()
    q3b_passed = run_q3b_tests()
    q3c_passed = run_q3c_tests()
    q3d_passed = run_q3d_tests()
    dimensions_passed = test_shared_output_dimension_configuration()
    print(
        "Q3 shared p=d=3 output-dimension test:",
        "PASS" if dimensions_passed else "FAIL",
    )
    return (
        q3a_passed
        and q3b_passed
        and q3c_passed
        and q3d_passed
        and dimensions_passed
    )


if __name__ == "__main__":
    raise SystemExit(0 if run_q3_tests() else 1)
