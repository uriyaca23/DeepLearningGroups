"""Question 3 set-network implementations.

External dependency: PyTorch.

This file is developed one approved homework part at a time. It currently
contains the canonization, full group-averaging, sampled group-averaging,
DeepSets, and data-augmentation constructions from parts (a)-(e).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from itertools import permutations
from math import factorial
from pathlib import Path
from typing import Optional

import matplotlib
import torch
from torch import Tensor, nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt


DEFAULT_N = 7
DEFAULT_D = 3
DEFAULT_OUTPUT_DIM = DEFAULT_D
DEFAULT_MLP_HIDDEN_DIM = 32
DEFAULT_DEEPSETS_HIDDEN_DIM = 62
DEFAULT_SUBSET_SIZE = 2700
DEFAULT_SEED = 2319
DEFAULT_DATASET_SIZE = 1000
DEFAULT_TRAIN_SIZE = 700
DEFAULT_VAL_SIZE = 150
DEFAULT_TEST_SIZE = 150
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MAX_EPOCHS = 2000
DEFAULT_EARLY_STOPPING_PATIENCE = 100
DEFAULT_Q3E_ATOL = 1e-2


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


@dataclass(frozen=True)
class Q3EDataset:
    """Synthetic Question 3(e) data and its latent Gaussian parameters."""

    inputs: Tensor
    targets: Tensor
    means: Tensor
    variances: Tensor


@dataclass(frozen=True)
class Q3EMetrics:
    """Approximation and invariance metrics on one fixed evaluation set."""

    approximation_mse: float
    mean_l2_invariance_error: float
    max_absolute_invariance_error: float


@dataclass
class Q3ETrainingRun:
    """A trained model together with its early-stopping history."""

    model: TwoLayerMLP
    training_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    best_validation_mse: float


@dataclass(frozen=True)
class Q3EExperimentResult:
    """Complete before/after comparison for the approved Q3(e) experiment."""

    initial_metrics: Q3EMetrics
    augmented_metrics: Q3EMetrics
    control_metrics: Q3EMetrics
    augmented_best_epoch: int
    control_best_epoch: int
    augmented_best_validation_mse: float
    control_best_validation_mse: float
    augmented_invariance_passed: bool
    control_invariance_passed: bool
    plot_path: Path
    results_path: Path


def coordinate_variance_target(x: Tensor) -> Tensor:
    """Compute the assignment's coordinate-wise variance using the 1/n divisor."""

    if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError(
            "Expected a batch with shape (examples, n, d), "
            f"got {tuple(x.shape)}"
        )
    return x.var(dim=1, unbiased=False)


def _sample_standard_rayleigh(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample Rayleigh(scale=1) values by inverse-transform sampling."""

    uniform = torch.rand(shape, generator=generator, dtype=dtype)
    epsilon = torch.finfo(dtype).eps
    uniform = uniform.clamp(min=epsilon, max=1.0 - epsilon)
    return torch.sqrt(-2.0 * torch.log(uniform))


def generate_q3e_dataset(
    *,
    dataset_size: int = DEFAULT_DATASET_SIZE,
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
    seed: int = DEFAULT_SEED,
) -> Q3EDataset:
    """Generate independent Gaussian sets with per-example means and variances."""

    if dataset_size < 1 or n < 1 or d < 1:
        raise ValueError("dataset_size, n, and d must all be positive")

    generator = torch.Generator().manual_seed(seed)
    means = torch.randn(dataset_size, d, generator=generator)
    variances = _sample_standard_rayleigh(
        (dataset_size, d),
        generator=generator,
    )
    standard_normal_noise = torch.randn(
        dataset_size,
        n,
        d,
        generator=generator,
    )
    inputs = (
        means.unsqueeze(1)
        + variances.sqrt().unsqueeze(1) * standard_normal_noise
    )
    targets = coordinate_variance_target(inputs)
    return Q3EDataset(
        inputs=inputs,
        targets=targets,
        means=means,
        variances=variances,
    )


def draw_row_permutations(
    batch_size: int,
    n: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Draw one independent uniform row permutation for every batch example."""

    if batch_size < 1 or n < 1:
        raise ValueError("batch_size and n must both be positive")
    return torch.stack(
        [torch.randperm(n, generator=generator) for _ in range(batch_size)]
    )


def apply_row_permutations(x: Tensor, permutation_indices: Tensor) -> Tensor:
    """Apply a separately supplied row permutation to every set in a batch."""

    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (batch, n, d), got {tuple(x.shape)}")
    if permutation_indices.shape != x.shape[:2]:
        raise ValueError(
            "Expected permutation_indices with shape "
            f"{tuple(x.shape[:2])}, got {tuple(permutation_indices.shape)}"
        )

    gather_indices = permutation_indices.unsqueeze(-1).expand(-1, -1, x.shape[2])
    return torch.gather(x, dim=1, index=gather_indices)


@torch.no_grad()
def q3e_metrics(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    permutation_indices: Tensor,
) -> Q3EMetrics:
    """Evaluate approximation and invariance on fixed examples/permutations."""

    model.eval()
    predictions = model(inputs)
    permuted_inputs = apply_row_permutations(inputs, permutation_indices)
    permuted_predictions = model(permuted_inputs)
    prediction_difference = permuted_predictions - predictions

    return Q3EMetrics(
        approximation_mse=float(torch.mean((predictions - targets) ** 2).item()),
        mean_l2_invariance_error=float(
            torch.linalg.vector_norm(prediction_difference, dim=1).mean().item()
        ),
        max_absolute_invariance_error=float(
            prediction_difference.abs().max().item()
        ),
    )


@torch.no_grad()
def _q3e_dataset_mse(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
) -> float:
    model.eval()
    return float(torch.mean((model(inputs) - targets) ** 2).item())


def train_q3e_model(
    model: TwoLayerMLP,
    *,
    training_inputs: Tensor,
    training_targets: Tensor,
    validation_inputs: Tensor,
    validation_targets: Tensor,
    augment: bool,
    maximum_epochs: int,
    training_order_seed: int,
    augmentation_seed: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
) -> Q3ETrainingRun:
    """Train one Q3(e) model with validation-based early stopping."""

    if training_inputs.shape[0] != training_targets.shape[0]:
        raise ValueError("Training inputs and targets must have equal lengths")
    if validation_inputs.shape[0] != validation_targets.shape[0]:
        raise ValueError("Validation inputs and targets must have equal lengths")
    if maximum_epochs < 1:
        raise ValueError("maximum_epochs must be positive")
    if batch_size < 1 or learning_rate <= 0 or patience < 1:
        raise ValueError("batch_size, learning_rate, and patience must be positive")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    training_order_generator = torch.Generator().manual_seed(training_order_seed)
    augmentation_generator = torch.Generator().manual_seed(augmentation_seed)

    training_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_mse = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch_index in range(maximum_epochs):
        epoch_order = torch.randperm(
            training_inputs.shape[0],
            generator=training_order_generator,
        )
        model.train()
        total_squared_error = 0.0
        total_coordinates = 0

        for batch_start in range(0, len(epoch_order), batch_size):
            batch_indices = epoch_order[batch_start : batch_start + batch_size]
            batch_inputs = training_inputs[batch_indices]
            batch_targets = training_targets[batch_indices]

            if augment:
                row_permutations = draw_row_permutations(
                    batch_size=batch_inputs.shape[0],
                    n=batch_inputs.shape[1],
                    generator=augmentation_generator,
                )
                batch_inputs = apply_row_permutations(
                    batch_inputs,
                    row_permutations,
                )

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            total_squared_error += float(
                torch.sum((predictions.detach() - batch_targets) ** 2).item()
            )
            total_coordinates += batch_targets.numel()

        training_mse = total_squared_error / total_coordinates
        validation_mse = _q3e_dataset_mse(
            model,
            validation_inputs,
            validation_targets,
        )
        training_losses.append(training_mse)
        validation_losses.append(validation_mse)

        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch_index + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return Q3ETrainingRun(
        model=model,
        training_losses=training_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        best_validation_mse=best_validation_mse,
    )


def save_q3e_training_plot(
    augmented_run: Q3ETrainingRun,
    control_run: Q3ETrainingRun,
    output_path: Path,
) -> None:
    """Save the approved two-panel training/validation MSE figure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for axis, title, run in (
        (axes[0], "Permutation augmentation", augmented_run),
        (axes[1], "No-augmentation control", control_run),
    ):
        epochs = range(1, len(run.training_losses) + 1)
        axis.plot(epochs, run.training_losses, label="Training MSE")
        axis.plot(epochs, run.validation_losses, label="Validation MSE")
        axis.axvline(
            run.best_epoch,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Best epoch: {run.best_epoch}",
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend()

    axes[0].set_ylabel("Mean squared error")
    figure.suptitle("Question 3(e): training and validation loss")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _metrics_to_dict(metrics: Q3EMetrics) -> dict[str, float]:
    return {
        "approximation_mse": metrics.approximation_mse,
        "mean_l2_invariance_error": metrics.mean_l2_invariance_error,
        "max_absolute_invariance_error": metrics.max_absolute_invariance_error,
    }


def test_q3e_data_and_augmentation(
    dataset: Q3EDataset,
    *,
    dataset_size: int = DEFAULT_DATASET_SIZE,
    n: int = DEFAULT_N,
    d: int = DEFAULT_D,
    seed: int = DEFAULT_SEED,
) -> bool:
    """Check shapes, targets, per-example parameters, and row permutations."""

    expected_input_shape = (dataset_size, n, d)
    expected_parameter_shape = (dataset_size, d)
    targets_match = torch.allclose(
        dataset.targets,
        coordinate_variance_target(dataset.inputs),
        atol=0.0,
        rtol=0.0,
    )
    parameters_vary = (
        torch.unique(dataset.means, dim=0).shape[0] > 1
        and torch.unique(dataset.variances, dim=0).shape[0] > 1
    )

    generator = torch.Generator().manual_seed(seed + 3000)
    first_permutations = draw_row_permutations(
        batch_size=32,
        n=n,
        generator=generator,
    )
    second_permutations = draw_row_permutations(
        batch_size=32,
        n=n,
        generator=generator,
    )
    expected_indices = torch.arange(n).expand(32, -1)
    permutations_valid = (
        torch.equal(torch.sort(first_permutations, dim=1).values, expected_indices)
        and torch.equal(
            torch.sort(second_permutations, dim=1).values,
            expected_indices,
        )
        and not torch.equal(first_permutations, second_permutations)
    )
    target_is_permutation_invariant = torch.equal(
        coordinate_variance_target(
            apply_row_permutations(dataset.inputs[:32], first_permutations)
        ),
        dataset.targets[:32],
    )

    return (
        tuple(dataset.inputs.shape) == expected_input_shape
        and tuple(dataset.targets.shape) == expected_parameter_shape
        and tuple(dataset.means.shape) == expected_parameter_shape
        and tuple(dataset.variances.shape) == expected_parameter_shape
        and bool(torch.all(dataset.variances > 0))
        and bool(torch.all(torch.isfinite(dataset.inputs)))
        and bool(torch.all(torch.isfinite(dataset.targets)))
        and targets_match
        and parameters_vary
        and permutations_valid
        and target_is_permutation_invariant
    )


def run_q3e_experiment(
    *,
    output_dir: Optional[Path] = None,
    dataset_size: int = DEFAULT_DATASET_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    maximum_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    seed: int = DEFAULT_SEED,
) -> tuple[Q3EExperimentResult, bool]:
    """Run the approved augmented/control training comparison and save results."""

    if dataset_size < 20 or dataset_size % 20 != 0:
        raise ValueError(
            "dataset_size must be a positive multiple of 20 for a 70/15/15 split"
        )
    training_size = 7 * dataset_size // 10
    validation_size = 3 * dataset_size // 20
    test_size = dataset_size - training_size - validation_size

    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent
            / "_qa"
            / "q3e"
            / (
                f"size-{dataset_size}-batch-{batch_size}"
                f"-epochs-{maximum_epochs}-patience-{patience}"
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_q3e_dataset(dataset_size=dataset_size, seed=seed)
    data_checks_passed = test_q3e_data_and_augmentation(
        dataset,
        dataset_size=dataset_size,
        seed=seed,
    )

    training_end = training_size
    validation_end = training_end + validation_size
    test_end = validation_end + test_size
    if test_end != dataset_size:
        raise RuntimeError("The Q3(e) split sizes do not cover the dataset")

    training_inputs = dataset.inputs[:training_end]
    training_targets = dataset.targets[:training_end]
    validation_inputs = dataset.inputs[training_end:validation_end]
    validation_targets = dataset.targets[training_end:validation_end]
    test_inputs = dataset.inputs[validation_end:test_end]
    test_targets = dataset.targets[validation_end:test_end]

    torch.manual_seed(seed)
    initial_model = TwoLayerMLP(
        n=DEFAULT_N,
        d=DEFAULT_D,
        output_dim=DEFAULT_OUTPUT_DIM,
        hidden_dim=DEFAULT_MLP_HIDDEN_DIM,
    )
    initial_state = copy.deepcopy(initial_model.state_dict())

    evaluation_generator = torch.Generator().manual_seed(seed + 4000)
    evaluation_permutations = draw_row_permutations(
        batch_size=test_size,
        n=DEFAULT_N,
        generator=evaluation_generator,
    )
    initial_metrics = q3e_metrics(
        initial_model,
        test_inputs,
        test_targets,
        evaluation_permutations,
    )

    augmented_model = TwoLayerMLP(
        n=DEFAULT_N,
        d=DEFAULT_D,
        output_dim=DEFAULT_OUTPUT_DIM,
        hidden_dim=DEFAULT_MLP_HIDDEN_DIM,
    )
    augmented_model.load_state_dict(initial_state)
    augmented_run = train_q3e_model(
        augmented_model,
        training_inputs=training_inputs,
        training_targets=training_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        augment=True,
        maximum_epochs=maximum_epochs,
        training_order_seed=seed + 1000,
        augmentation_seed=seed + 2000,
        batch_size=batch_size,
        patience=patience,
    )

    control_model = TwoLayerMLP(
        n=DEFAULT_N,
        d=DEFAULT_D,
        output_dim=DEFAULT_OUTPUT_DIM,
        hidden_dim=DEFAULT_MLP_HIDDEN_DIM,
    )
    control_model.load_state_dict(initial_state)
    control_run = train_q3e_model(
        control_model,
        training_inputs=training_inputs,
        training_targets=training_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        augment=False,
        maximum_epochs=maximum_epochs,
        training_order_seed=seed + 1000,
        augmentation_seed=seed + 2000,
        batch_size=batch_size,
        patience=patience,
    )

    augmented_metrics = q3e_metrics(
        augmented_run.model,
        test_inputs,
        test_targets,
        evaluation_permutations,
    )
    control_metrics = q3e_metrics(
        control_run.model,
        test_inputs,
        test_targets,
        evaluation_permutations,
    )
    augmented_invariance_passed = (
        augmented_metrics.max_absolute_invariance_error <= DEFAULT_Q3E_ATOL
    )
    control_invariance_passed = (
        control_metrics.max_absolute_invariance_error <= DEFAULT_Q3E_ATOL
    )

    plot_path = output_dir / "q3e_training_curves.png"
    results_path = output_dir / "q3e_results.json"
    save_q3e_training_plot(augmented_run, control_run, plot_path)

    result_payload = {
        "configuration": {
            "seed": seed,
            "n": DEFAULT_N,
            "d": DEFAULT_D,
            "output_dim": DEFAULT_OUTPUT_DIM,
            "hidden_dim": DEFAULT_MLP_HIDDEN_DIM,
            "dataset_size": dataset_size,
            "training_size": training_size,
            "validation_size": validation_size,
            "test_size": test_size,
            "batch_size": batch_size,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "maximum_epochs": maximum_epochs,
            "early_stopping_patience": patience,
            "maximum_error_tolerance": DEFAULT_Q3E_ATOL,
            "loss": "MSELoss",
            "optimizer": "Adam",
            "mean_distribution": "Normal(0, 1)",
            "variance_distribution": "Rayleigh(scale=1)",
            "target_variance_divisor": "n",
        },
        "data_and_augmentation_checks_passed": data_checks_passed,
        "initial_metrics": _metrics_to_dict(initial_metrics),
        "augmented_model": {
            "metrics": _metrics_to_dict(augmented_metrics),
            "maximum_error_test_passed": augmented_invariance_passed,
            "best_epoch": augmented_run.best_epoch,
            "best_validation_mse": augmented_run.best_validation_mse,
            "training_losses": augmented_run.training_losses,
            "validation_losses": augmented_run.validation_losses,
        },
        "control_model": {
            "metrics": _metrics_to_dict(control_metrics),
            "maximum_error_test_passed": control_invariance_passed,
            "best_epoch": control_run.best_epoch,
            "best_validation_mse": control_run.best_validation_mse,
            "training_losses": control_run.training_losses,
            "validation_losses": control_run.validation_losses,
        },
    }
    results_path.write_text(
        json.dumps(result_payload, indent=2),
        encoding="utf-8",
    )

    result = Q3EExperimentResult(
        initial_metrics=initial_metrics,
        augmented_metrics=augmented_metrics,
        control_metrics=control_metrics,
        augmented_best_epoch=augmented_run.best_epoch,
        control_best_epoch=control_run.best_epoch,
        augmented_best_validation_mse=augmented_run.best_validation_mse,
        control_best_validation_mse=control_run.best_validation_mse,
        augmented_invariance_passed=augmented_invariance_passed,
        control_invariance_passed=control_invariance_passed,
        plot_path=plot_path,
        results_path=results_path,
    )
    return result, data_checks_passed


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


def run_q3e_tests() -> bool:
    """Run the approved Q3(e) augmented/control training experiment."""

    result, data_checks_passed = run_q3e_experiment()
    metric_values = (
        *_metrics_to_dict(result.initial_metrics).values(),
        *_metrics_to_dict(result.augmented_metrics).values(),
        *_metrics_to_dict(result.control_metrics).values(),
        result.augmented_best_validation_mse,
        result.control_best_validation_mse,
    )
    metrics_are_finite = bool(torch.all(torch.isfinite(torch.tensor(metric_values))))
    output_files_exist = result.plot_path.is_file() and result.results_path.is_file()

    print(
        "Q3(e) data-generation and fresh-permutation checks:",
        "PASS" if data_checks_passed else "FAIL",
    )
    print(
        "Q3(e) augmented training:",
        f"best_epoch={result.augmented_best_epoch}, "
        f"best_validation_mse={result.augmented_best_validation_mse:.12g}, "
        f"test_mse={result.augmented_metrics.approximation_mse:.12g}",
    )
    print(
        "Q3(e) no-augmentation control:",
        f"best_epoch={result.control_best_epoch}, "
        f"best_validation_mse={result.control_best_validation_mse:.12g}, "
        f"test_mse={result.control_metrics.approximation_mse:.12g}",
    )
    print(
        "Q3(e) augmented mean-L2 invariance error:",
        f"before={result.initial_metrics.mean_l2_invariance_error:.12g}, "
        f"after={result.augmented_metrics.mean_l2_invariance_error:.12g}",
    )
    print(
        "Q3(e) control mean-L2 invariance error:",
        f"before={result.initial_metrics.mean_l2_invariance_error:.12g}, "
        f"after={result.control_metrics.mean_l2_invariance_error:.12g}",
    )
    print(
        "Q3(e) augmented maximum-coordinate invariance test:",
        "PASS" if result.augmented_invariance_passed else "FAIL",
        f"(atol={DEFAULT_Q3E_ATOL:g}, "
        f"before={result.initial_metrics.max_absolute_invariance_error:.12g}, "
        f"after={result.augmented_metrics.max_absolute_invariance_error:.12g})",
    )
    print(
        "Q3(e) control maximum-coordinate invariance test:",
        "PASS" if result.control_invariance_passed else "FAIL",
        f"(atol={DEFAULT_Q3E_ATOL:g}, "
        f"before={result.initial_metrics.max_absolute_invariance_error:.12g}, "
        f"after={result.control_metrics.max_absolute_invariance_error:.12g})",
    )
    print("Q3(e) training curves:", result.plot_path)
    print("Q3(e) full numerical results:", result.results_path)
    return data_checks_passed and metrics_are_finite and output_files_exist


def run_q3_tests() -> bool:
    """Run all implemented Question 3 checks."""

    q3a_passed = run_q3a_tests()
    q3b_passed = run_q3b_tests()
    q3c_passed = run_q3c_tests()
    q3d_passed = run_q3d_tests()
    q3e_completed = run_q3e_tests()
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
        and q3e_completed
        and dimensions_passed
    )


if __name__ == "__main__":
    raise SystemExit(0 if run_q3_tests() else 1)
