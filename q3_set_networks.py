"""HW4 Question 3. Dependencies: CUDA-enabled PyTorch and Matplotlib."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import torch
from torch import Tensor, nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt


DEFAULT_SEED = 2319
DEFAULT_D = 3
DEFAULT_CLASS_OUTPUT_DIM = 40
DEFAULT_VARIANCE_OUTPUT_DIM = DEFAULT_D

DEFAULT_SCALABLE_N = 256
DEFAULT_SYMMETRIZATION_N = 7
DEFAULT_SCALABLE_HIDDEN_DIMS = (64, 64)
DEFAULT_SYMMETRIZATION_HIDDEN_DIMS = (207, 207)
DEFAULT_DEEPSETS_HIDDEN_DIM = 128
DEFAULT_Q3E_COMPARISON_N = 7
DEFAULT_Q3E_COMPARISON_HIDDEN_DIMS = (349, 130)

DEFAULT_FULL_GROUP_SIZE = math.factorial(DEFAULT_SYMMETRIZATION_N)
DEFAULT_SUBSET_SIZE = DEFAULT_FULL_GROUP_SIZE // 20  # exactly 5% = 252
DEFAULT_Q3C_TRIALS = 100
DEFAULT_EXACT_ATOL = 1e-5
DEFAULT_APPROXIMATE_ATOL = 1e-2

DEFAULT_DATASET_SIZE = 1000
DEFAULT_TRAIN_SIZE = 700
DEFAULT_VAL_SIZE = 150
DEFAULT_TEST_SIZE = 150
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MAX_EPOCHS = 2000
DEFAULT_EARLY_STOPPING_PATIENCE = 100


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA-enabled PyTorch build and an available NVIDIA GPU are "
            "required for the Q3/Q4 experiments."
        )
    return torch.device("cuda:0")


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cuda_generator(seed: int) -> torch.Generator:
    return torch.Generator(device=require_cuda()).manual_seed(seed)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
) -> nn.Sequential:
    if input_dim < 1 or output_dim < 1 or any(width < 1 for width in hidden_dims):
        raise ValueError("All MLP dimensions must be positive")

    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def lexicographic_sort_rows(x: Tensor) -> Tensor:
    if x.ndim < 2 or x.shape[-2] < 1 or x.shape[-1] < 1:
        raise ValueError(f"Expected shape (..., n, d), got {tuple(x.shape)}")

    n, d = x.shape[-2:]
    batch_shape = x.shape[:-2]
    indices = torch.arange(n, device=x.device).expand(*batch_shape, n)
    for feature in range(d - 1, -1, -1):
        values = x[..., feature]
        ordered_values = torch.gather(values, -1, indices)
        order = torch.argsort(ordered_values, dim=-1, stable=True)
        indices = torch.gather(indices, -1, order)

    gather_indices = indices.unsqueeze(-1).expand(*batch_shape, n, d)
    return torch.gather(x, -2, gather_indices)


class FlattenedMLP(nn.Module):
    def __init__(
        self,
        n: int,
        d: int,
        output_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        if n < 1 or d < 1:
            raise ValueError("n and d must be positive")
        self.n = n
        self.d = d
        self.output_dim = output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.network = _build_mlp(n * d, hidden_dims, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2 or tuple(x.shape[-2:]) != (self.n, self.d):
            raise ValueError(
                f"Expected shape (..., {self.n}, {self.d}), got {tuple(x.shape)}"
            )
        return self.network(x.reshape(*x.shape[:-2], self.n * self.d))


class CanonizationInvariantMLP(nn.Module):
    def __init__(
        self,
        n: int = DEFAULT_SCALABLE_N,
        d: int = DEFAULT_D,
        output_dim: int = DEFAULT_CLASS_OUTPUT_DIM,
        hidden_dims: Sequence[int] = DEFAULT_SCALABLE_HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        self.n = n
        self.d = d
        self.mlp = FlattenedMLP(n, d, output_dim, hidden_dims)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(lexicographic_sort_rows(x))


def _permutation_outputs(
    base_model: FlattenedMLP,
    x: Tensor,
    permutation_indices: Tensor,
    *,
    chunk_size: int,
) -> Tensor:
    single = x.ndim == 2
    if single:
        x = x.unsqueeze(0)
    if x.ndim != 3 or tuple(x.shape[-2:]) != (base_model.n, base_model.d):
        raise ValueError("Unexpected set shape for permutation evaluation")

    batch_size = x.shape[0]
    chunks: list[Tensor] = []
    for start in range(0, permutation_indices.shape[0], chunk_size):
        index_chunk = permutation_indices[start : start + chunk_size]
        permuted = x[:, index_chunk, :]
        outputs = base_model(
            permuted.reshape(-1, base_model.n, base_model.d)
        ).reshape(batch_size, index_chunk.shape[0], -1)
        chunks.append(outputs)
    combined = torch.cat(chunks, dim=1)
    return combined.squeeze(0) if single else combined


class SymmetrizedInvariantModel(nn.Module):
    def __init__(
        self,
        base_model: FlattenedMLP,
        *,
        permutation_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.n = base_model.n
        self.d = base_model.d
        self.permutation_chunk_size = permutation_chunk_size
        self.register_buffer(
            "permutation_indices",
            torch.tensor(list(permutations(range(self.n))), dtype=torch.long),
        )

    def forward(self, x: Tensor) -> Tensor:
        outputs = _permutation_outputs(
            self.base_model,
            x,
            self.permutation_indices,
            chunk_size=self.permutation_chunk_size,
        )
        permutation_axis = -2 if x.ndim == 3 else 0
        return outputs.mean(dim=permutation_axis)


class SampledSymmetrizedModel(nn.Module):
    def __init__(
        self,
        base_model: FlattenedMLP,
        subset_size: int = DEFAULT_SUBSET_SIZE,
        subset_seed: int = DEFAULT_SEED,
        *,
        permutation_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.n = base_model.n
        self.d = base_model.d
        self.subset_size = subset_size
        self.subset_seed = subset_seed
        self.permutation_chunk_size = permutation_chunk_size

        all_indices = torch.tensor(
            list(permutations(range(self.n))), dtype=torch.long
        )
        group_size = all_indices.shape[0]
        if not 0 < subset_size < group_size:
            raise ValueError("subset_size must be a proper nonempty subset")
        generator = torch.Generator().manual_seed(subset_seed)
        selected = torch.randperm(group_size, generator=generator)[:subset_size]
        self.register_buffer("permutation_indices", all_indices[selected])

    def forward(self, x: Tensor) -> Tensor:
        outputs = _permutation_outputs(
            self.base_model,
            x,
            self.permutation_indices,
            chunk_size=self.permutation_chunk_size,
        )
        permutation_axis = -2 if x.ndim == 3 else 0
        return outputs.mean(dim=permutation_axis)


class DeepSetsEquivariantLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.local_linear = nn.Linear(input_dim, output_dim, bias=True)
        self.global_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected (..., n, {self.input_dim})")
        row_mean = x.mean(dim=-2, keepdim=True)
        return self.local_linear(x) + self.global_linear(row_mean)


class DeepSetsEquivariantBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int = DEFAULT_D,
        hidden_dim: int = DEFAULT_DEEPSETS_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.first_layer = DeepSetsEquivariantLinear(input_dim, hidden_dim)
        self.second_layer = DeepSetsEquivariantLinear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(
            self.second_layer(self.activation(self.first_layer(x)))
        )


class DeepSetsInvariantClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = DEFAULT_D,
        hidden_dim: int = DEFAULT_DEEPSETS_HIDDEN_DIM,
        output_dim: int = DEFAULT_CLASS_OUTPUT_DIM,
    ) -> None:
        super().__init__()
        self.backbone = DeepSetsEquivariantBackbone(input_dim, hidden_dim)
        self.head = _build_mlp(hidden_dim, (hidden_dim,), output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x).mean(dim=-2))


@dataclass(frozen=True)
class Q3CMonteCarloMetrics:
    empirical_rms_error: float
    predicted_rms_error: float
    relative_prediction_gap: float
    theoretical_coefficient: float
    p95_example_rms_error: float
    maximum_absolute_error: float


@dataclass(frozen=True)
class Q3EDataset:
    inputs: Tensor
    targets: Tensor
    means: Tensor
    variances: Tensor


@dataclass(frozen=True)
class Q3EMetrics:
    approximation_mse: float
    mean_l2_invariance_error: float
    rms_invariance_error: float
    p95_example_rms_error: float
    max_absolute_invariance_error: float


@dataclass
class Q3ETrainingRun:
    model: FlattenedMLP
    training_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    best_validation_mse: float


@dataclass(frozen=True)
class Q3EExperimentResult:
    initial_metrics: Q3EMetrics
    augmented_metrics: Q3EMetrics
    control_metrics: Q3EMetrics
    augmented_best_epoch: int
    control_best_epoch: int
    augmented_best_validation_mse: float
    control_best_validation_mse: float
    n7_initial_metrics: Q3EMetrics
    n7_augmented_metrics: Q3EMetrics
    n7_control_metrics: Q3EMetrics
    n7_augmented_best_epoch: int
    n7_control_best_epoch: int
    n7_augmented_best_validation_mse: float
    n7_control_best_validation_mse: float
    plot_path: Path
    results_path: Path


def coordinate_variance_target(x: Tensor) -> Tensor:
    if x.ndim != 3:
        raise ValueError("Expected (examples, n, d)")
    return x.var(dim=1, unbiased=False)


def _sample_standard_rayleigh(
    shape: tuple[int, ...], *, generator: torch.Generator
) -> Tensor:
    uniform = torch.rand(shape, device=require_cuda(), generator=generator)
    epsilon = torch.finfo(uniform.dtype).eps
    return torch.sqrt(-2.0 * torch.log(uniform.clamp(epsilon, 1.0 - epsilon)))


def generate_q3e_dataset(
    *,
    dataset_size: int = DEFAULT_DATASET_SIZE,
    n: int = DEFAULT_SCALABLE_N,
    d: int = DEFAULT_D,
    seed: int = DEFAULT_SEED,
) -> Q3EDataset:
    generator = _cuda_generator(seed)
    device = require_cuda()
    means = torch.randn(dataset_size, d, device=device, generator=generator)
    variances = _sample_standard_rayleigh(
        (dataset_size, d), generator=generator
    )
    noise = torch.randn(
        dataset_size, n, d, device=device, generator=generator
    )
    inputs = means[:, None, :] + variances.sqrt()[:, None, :] * noise
    return Q3EDataset(
        inputs=inputs,
        targets=coordinate_variance_target(inputs),
        means=means,
        variances=variances,
    )


def draw_row_permutations(
    batch_size: int,
    n: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    random_keys = torch.rand(
        batch_size, n, device=require_cuda(), generator=generator
    )
    return torch.argsort(random_keys, dim=1)


def apply_row_permutations(x: Tensor, permutation_indices: Tensor) -> Tensor:
    gather = permutation_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, 1, gather)


def test_q3e_data_and_augmentation(
    dataset: Q3EDataset,
    *,
    dataset_size: int = DEFAULT_DATASET_SIZE,
    n: int = DEFAULT_SCALABLE_N,
    d: int = DEFAULT_D,
    seed: int = DEFAULT_SEED,
) -> bool:
    expected_inputs = (dataset_size, n, d)
    expected_vectors = (dataset_size, d)
    generator = _cuda_generator(seed + 3000)
    first = draw_row_permutations(
        32, n, generator=generator
    )
    second = draw_row_permutations(
        32, n, generator=generator
    )
    expected_indices = torch.arange(
        n, device=require_cuda()
    ).expand(32, -1)
    permuted_targets = coordinate_variance_target(
        apply_row_permutations(dataset.inputs[:32], first)
    )
    return (
        dataset.inputs.is_cuda
        and dataset.targets.is_cuda
        and tuple(dataset.inputs.shape) == expected_inputs
        and tuple(dataset.targets.shape) == expected_vectors
        and tuple(dataset.means.shape) == expected_vectors
        and tuple(dataset.variances.shape) == expected_vectors
        and bool(torch.all(dataset.variances > 0))
        and bool(torch.isfinite(dataset.inputs).all())
        and bool(torch.isfinite(dataset.targets).all())
        and torch.equal(
            dataset.targets, coordinate_variance_target(dataset.inputs)
        )
        and torch.equal(torch.sort(first, dim=1).values, expected_indices)
        and torch.equal(torch.sort(second, dim=1).values, expected_indices)
        and not torch.equal(first, second)
        and torch.allclose(
            permuted_targets, dataset.targets[:32], atol=1e-6, rtol=0.0
        )
    )


@torch.no_grad()
def q3e_metrics(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    permutation_indices: Tensor,
) -> Q3EMetrics:
    model.eval()
    predictions = model(inputs)
    permuted_predictions = model(
        apply_row_permutations(inputs, permutation_indices)
    )
    differences = permuted_predictions - predictions
    per_example_l2 = torch.linalg.vector_norm(differences, dim=1)
    per_example_rms = differences.square().mean(dim=1).sqrt()
    return Q3EMetrics(
        approximation_mse=float((predictions - targets).square().mean().item()),
        mean_l2_invariance_error=float(per_example_l2.mean().item()),
        rms_invariance_error=float(differences.square().mean().sqrt().item()),
        p95_example_rms_error=float(
            torch.quantile(per_example_rms, 0.95).item()
        ),
        max_absolute_invariance_error=float(differences.abs().max().item()),
    )


@torch.no_grad()
def _dataset_mse(model: nn.Module, inputs: Tensor, targets: Tensor) -> float:
    model.eval()
    return float((model(inputs) - targets).square().mean().item())


def train_q3e_model(
    model: FlattenedMLP,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    order_generator = _cuda_generator(training_order_seed)
    augmentation_generator = _cuda_generator(augmentation_seed)
    training_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_mse = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(1, maximum_epochs + 1):
        order = torch.randperm(
            training_inputs.shape[0],
            device=require_cuda(),
            generator=order_generator,
        )
        model.train()
        squared_error = 0.0
        coordinate_count = 0
        for start in range(0, order.numel(), batch_size):
            indices = order[start : start + batch_size]
            batch_inputs = training_inputs[indices]
            batch_targets = training_targets[indices]
            if augment:
                row_permutations = draw_row_permutations(
                    batch_inputs.shape[0],
                    batch_inputs.shape[1],
                    generator=augmentation_generator,
                )
                batch_inputs = apply_row_permutations(
                    batch_inputs, row_permutations
                )

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            squared_error += float(
                (predictions.detach() - batch_targets).square().sum().item()
            )
            coordinate_count += batch_targets.numel()

        training_mse = squared_error / coordinate_count
        validation_mse = _dataset_mse(
            model, validation_inputs, validation_targets
        )
        training_losses.append(training_mse)
        validation_losses.append(validation_mse)
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
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
    n7_augmented_run: Q3ETrainingRun,
    n7_control_run: Q3ETrainingRun,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharey=True)
    for axis, title, run in (
        (axes[0, 0], "n=256: permutation augmentation", augmented_run),
        (axes[0, 1], "n=256: no-augmentation control", control_run),
        (axes[1, 0], "n=7: permutation augmentation", n7_augmented_run),
        (axes[1, 1], "n=7: no-augmentation control", n7_control_run),
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
    axes[0, 0].set_ylabel("Mean squared error")
    axes[1, 0].set_ylabel("Mean squared error")
    figure.suptitle("Question 3(e): training and validation loss")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _metrics_to_dict(metrics: Q3EMetrics) -> dict[str, float]:
    return {
        "approximation_mse": metrics.approximation_mse,
        "mean_l2_invariance_error": metrics.mean_l2_invariance_error,
        "rms_invariance_error": metrics.rms_invariance_error,
        "p95_example_rms_error": metrics.p95_example_rms_error,
        "max_absolute_invariance_error": metrics.max_absolute_invariance_error,
    }


def run_q3e_experiment(
    *,
    output_dir: Optional[Path] = None,
    dataset_size: int = DEFAULT_DATASET_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    maximum_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    seed: int = DEFAULT_SEED,
) -> Q3EExperimentResult:
    device = require_cuda()
    if dataset_size < 20 or dataset_size % 20:
        raise ValueError("dataset_size must be a multiple of 20")
    train_size = 7 * dataset_size // 10
    val_size = 3 * dataset_size // 20
    test_size = dataset_size - train_size - val_size
    output_dir = output_dir or (
        Path(__file__).parent
        / "_qa"
        / "q3e-q4-aligned"
        / f"size-{dataset_size}-batch-{batch_size}-epochs-{maximum_epochs}-patience-{patience}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_q3e_dataset(dataset_size=dataset_size, seed=seed)
    data_checks_passed = test_q3e_data_and_augmentation(
        dataset, dataset_size=dataset_size, seed=seed
    )
    if not data_checks_passed:
        raise RuntimeError("Q3(e) CUDA data or augmentation checks failed")
    train_end = train_size
    val_end = train_end + val_size
    train_x, train_y = dataset.inputs[:train_end], dataset.targets[:train_end]
    val_x, val_y = dataset.inputs[train_end:val_end], dataset.targets[train_end:val_end]
    test_x, test_y = dataset.inputs[val_end:], dataset.targets[val_end:]

    _seed_all(seed)
    initial_model = FlattenedMLP(
        DEFAULT_SCALABLE_N,
        DEFAULT_D,
        DEFAULT_VARIANCE_OUTPUT_DIM,
        DEFAULT_SCALABLE_HIDDEN_DIMS,
    ).to(device)
    initial_state = copy.deepcopy(initial_model.state_dict())
    evaluation_permutations = draw_row_permutations(
        test_size,
        DEFAULT_SCALABLE_N,
        generator=_cuda_generator(seed + 4000),
    )
    initial_metrics = q3e_metrics(
        initial_model, test_x, test_y, evaluation_permutations
    )

    augmented_model = FlattenedMLP(
        DEFAULT_SCALABLE_N,
        DEFAULT_D,
        DEFAULT_VARIANCE_OUTPUT_DIM,
        DEFAULT_SCALABLE_HIDDEN_DIMS,
    ).to(device)
    augmented_model.load_state_dict(initial_state)
    augmented_run = train_q3e_model(
        augmented_model,
        training_inputs=train_x,
        training_targets=train_y,
        validation_inputs=val_x,
        validation_targets=val_y,
        augment=True,
        maximum_epochs=maximum_epochs,
        training_order_seed=seed + 1000,
        augmentation_seed=seed + 2000,
        batch_size=batch_size,
        patience=patience,
    )

    control_model = FlattenedMLP(
        DEFAULT_SCALABLE_N,
        DEFAULT_D,
        DEFAULT_VARIANCE_OUTPUT_DIM,
        DEFAULT_SCALABLE_HIDDEN_DIMS,
    ).to(device)
    control_model.load_state_dict(initial_state)
    control_run = train_q3e_model(
        control_model,
        training_inputs=train_x,
        training_targets=train_y,
        validation_inputs=val_x,
        validation_targets=val_y,
        augment=False,
        maximum_epochs=maximum_epochs,
        training_order_seed=seed + 1000,
        augmentation_seed=seed + 2000,
        batch_size=batch_size,
        patience=patience,
    )

    augmented_metrics = q3e_metrics(
        augmented_run.model, test_x, test_y, evaluation_permutations
    )
    control_metrics = q3e_metrics(
        control_run.model, test_x, test_y, evaluation_permutations
    )

    # Match the n=7 and n=256 samples.
    n7_inputs = dataset.inputs[:, :DEFAULT_Q3E_COMPARISON_N, :].contiguous()
    n7_targets = coordinate_variance_target(n7_inputs)
    n7_train_x, n7_train_y = n7_inputs[:train_end], n7_targets[:train_end]
    n7_val_x, n7_val_y = (
        n7_inputs[train_end:val_end],
        n7_targets[train_end:val_end],
    )
    n7_test_x, n7_test_y = n7_inputs[val_end:], n7_targets[val_end:]

    _seed_all(seed)
    n7_initial_model = FlattenedMLP(
        DEFAULT_Q3E_COMPARISON_N,
        DEFAULT_D,
        DEFAULT_VARIANCE_OUTPUT_DIM,
        DEFAULT_Q3E_COMPARISON_HIDDEN_DIMS,
    ).to(device)
    if parameter_count(n7_initial_model) != parameter_count(initial_model):
        raise RuntimeError("The n=7 and n=256 Q3(e) models must match in parameters")
    n7_initial_state = copy.deepcopy(n7_initial_model.state_dict())
    n7_evaluation_permutations = draw_row_permutations(
        test_size,
        DEFAULT_Q3E_COMPARISON_N,
        generator=_cuda_generator(seed + 5000),
    )
    n7_initial_metrics = q3e_metrics(
        n7_initial_model,
        n7_test_x,
        n7_test_y,
        n7_evaluation_permutations,
    )

    n7_augmented_model = FlattenedMLP(
        DEFAULT_Q3E_COMPARISON_N,
        DEFAULT_D,
        DEFAULT_VARIANCE_OUTPUT_DIM,
        DEFAULT_Q3E_COMPARISON_HIDDEN_DIMS,
    ).to(device)
    n7_augmented_model.load_state_dict(n7_initial_state)
    n7_augmented_run = train_q3e_model(
        n7_augmented_model,
        training_inputs=n7_train_x,
        training_targets=n7_train_y,
        validation_inputs=n7_val_x,
        validation_targets=n7_val_y,
        augment=True,
        maximum_epochs=maximum_epochs,
        training_order_seed=seed + 1000,
        augmentation_seed=seed + 2000,
        batch_size=batch_size,
        patience=patience,
    )

    n7_control_model = FlattenedMLP(
        DEFAULT_Q3E_COMPARISON_N,
        DEFAULT_D,
        DEFAULT_VARIANCE_OUTPUT_DIM,
        DEFAULT_Q3E_COMPARISON_HIDDEN_DIMS,
    ).to(device)
    n7_control_model.load_state_dict(n7_initial_state)
    n7_control_run = train_q3e_model(
        n7_control_model,
        training_inputs=n7_train_x,
        training_targets=n7_train_y,
        validation_inputs=n7_val_x,
        validation_targets=n7_val_y,
        augment=False,
        maximum_epochs=maximum_epochs,
        training_order_seed=seed + 1000,
        augmentation_seed=seed + 2000,
        batch_size=batch_size,
        patience=patience,
    )

    n7_augmented_metrics = q3e_metrics(
        n7_augmented_run.model,
        n7_test_x,
        n7_test_y,
        n7_evaluation_permutations,
    )
    n7_control_metrics = q3e_metrics(
        n7_control_run.model,
        n7_test_x,
        n7_test_y,
        n7_evaluation_permutations,
    )
    plot_path = output_dir / "q3e_training_curves.png"
    results_path = output_dir / "q3e_results.json"
    save_q3e_training_plot(
        augmented_run,
        control_run,
        n7_augmented_run,
        n7_control_run,
        plot_path,
    )

    result = Q3EExperimentResult(
        initial_metrics=initial_metrics,
        augmented_metrics=augmented_metrics,
        control_metrics=control_metrics,
        augmented_best_epoch=augmented_run.best_epoch,
        control_best_epoch=control_run.best_epoch,
        augmented_best_validation_mse=augmented_run.best_validation_mse,
        control_best_validation_mse=control_run.best_validation_mse,
        n7_initial_metrics=n7_initial_metrics,
        n7_augmented_metrics=n7_augmented_metrics,
        n7_control_metrics=n7_control_metrics,
        n7_augmented_best_epoch=n7_augmented_run.best_epoch,
        n7_control_best_epoch=n7_control_run.best_epoch,
        n7_augmented_best_validation_mse=n7_augmented_run.best_validation_mse,
        n7_control_best_validation_mse=n7_control_run.best_validation_mse,
        plot_path=plot_path,
        results_path=results_path,
    )
    payload = {
        "configuration": {
            "seed": seed,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "n": DEFAULT_SCALABLE_N,
            "d": DEFAULT_D,
            "output_dim": DEFAULT_VARIANCE_OUTPUT_DIM,
            "hidden_dims": list(DEFAULT_SCALABLE_HIDDEN_DIMS),
            "parameter_count": parameter_count(initial_model),
            "dataset_size": dataset_size,
            "training_size": train_size,
            "validation_size": val_size,
            "test_size": test_size,
            "batch_size": batch_size,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "maximum_epochs": maximum_epochs,
            "early_stopping_patience": patience,
            "invariance_metric": (
                "Mean over examples of the L2 norm between predictions on "
                "the original and permuted inputs"
            ),
            "loss": "MSELoss",
            "optimizer": "Adam",
        },
        "data_and_augmentation_checks_passed": data_checks_passed,
        "initial_metrics": _metrics_to_dict(initial_metrics),
        "augmented_model": {
            "metrics": _metrics_to_dict(augmented_metrics),
            "best_epoch": augmented_run.best_epoch,
            "best_validation_mse": augmented_run.best_validation_mse,
            "training_losses": augmented_run.training_losses,
            "validation_losses": augmented_run.validation_losses,
        },
        "control_model": {
            "metrics": _metrics_to_dict(control_metrics),
            "best_epoch": control_run.best_epoch,
            "best_validation_mse": control_run.best_validation_mse,
            "training_losses": control_run.training_losses,
            "validation_losses": control_run.validation_losses,
        },
        "n7_matched_parameter_experiment": {
            "configuration": {
                "n": DEFAULT_Q3E_COMPARISON_N,
                "d": DEFAULT_D,
                "output_dim": DEFAULT_VARIANCE_OUTPUT_DIM,
                "hidden_dims": list(DEFAULT_Q3E_COMPARISON_HIDDEN_DIMS),
                "parameter_count": parameter_count(n7_initial_model),
                "data_relationship": (
                    "First seven rows of each reproducible n=256 example; "
                    "targets recomputed with divisor n=7"
                ),
                "shared_training_configuration": True,
            },
            "initial_metrics": _metrics_to_dict(n7_initial_metrics),
            "augmented_model": {
                "metrics": _metrics_to_dict(n7_augmented_metrics),
                "best_epoch": n7_augmented_run.best_epoch,
                "best_validation_mse": n7_augmented_run.best_validation_mse,
                "training_losses": n7_augmented_run.training_losses,
                "validation_losses": n7_augmented_run.validation_losses,
            },
            "control_model": {
                "metrics": _metrics_to_dict(n7_control_metrics),
                "best_epoch": n7_control_run.best_epoch,
                "best_validation_mse": n7_control_run.best_validation_mse,
                "training_losses": n7_control_run.training_losses,
                "validation_losses": n7_control_run.validation_losses,
            },
        },
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result


def _new_symmetrization_base() -> FlattenedMLP:
    return FlattenedMLP(
        DEFAULT_SYMMETRIZATION_N,
        DEFAULT_D,
        DEFAULT_CLASS_OUTPUT_DIM,
        DEFAULT_SYMMETRIZATION_HIDDEN_DIMS,
    ).to(require_cuda())


@torch.no_grad()
def q3c_monte_carlo_validation(
    trials: int = DEFAULT_Q3C_TRIALS,
    seed: int = DEFAULT_SEED,
) -> Q3CMonteCarloMetrics:
    device = require_cuda()
    all_indices = torch.tensor(
        list(permutations(range(DEFAULT_SYMMETRIZATION_N))),
        device=device,
        dtype=torch.long,
    )
    group_size = all_indices.shape[0]
    factor = (group_size - DEFAULT_SUBSET_SIZE) / (
        DEFAULT_SUBSET_SIZE * group_size
    )
    observed_squared: list[float] = []
    predicted_variances: list[float] = []
    example_rms: list[float] = []
    maxima: list[float] = []

    for trial in range(trials):
        trial_seed = seed + trial
        _seed_all(trial_seed)
        model = _new_symmetrization_base().eval()
        generator = _cuda_generator(trial_seed + 10000)
        x = torch.randn(
            DEFAULT_SYMMETRIZATION_N,
            DEFAULT_D,
            device=device,
            generator=generator,
        )
        sigma = torch.randperm(
            DEFAULT_SYMMETRIZATION_N,
            device=device,
            generator=generator,
        )
        selected = torch.randperm(
            group_size, device=device, generator=generator
        )[:DEFAULT_SUBSET_SIZE]
        x_outputs = _permutation_outputs(
            model, x, all_indices, chunk_size=1024
        )
        sigma_outputs = _permutation_outputs(
            model, x[sigma], all_indices, chunk_size=1024
        )
        difference = (
            sigma_outputs[selected].mean(dim=0)
            - x_outputs[selected].mean(dim=0)
        )
        paired_population = sigma_outputs - x_outputs
        predicted_variance = factor * paired_population.var(
            dim=0, unbiased=True
        ).mean()
        observed_squared.append(float(difference.square().mean().item()))
        predicted_variances.append(float(predicted_variance.item()))
        example_rms.append(float(difference.square().mean().sqrt().item()))
        maxima.append(float(difference.abs().max().item()))

    empirical = math.sqrt(sum(observed_squared) / trials)
    predicted = math.sqrt(sum(predicted_variances) / trials)
    ordered = sorted(example_rms)
    p95 = ordered[math.ceil(0.95 * trials) - 1]
    finite_population_scale = math.sqrt(
        (group_size - DEFAULT_SUBSET_SIZE)
        / (DEFAULT_SUBSET_SIZE * (group_size - 1))
    )
    return Q3CMonteCarloMetrics(
        empirical_rms_error=empirical,
        predicted_rms_error=predicted,
        relative_prediction_gap=abs(empirical - predicted) / predicted,
        theoretical_coefficient=predicted / finite_population_scale,
        p95_example_rms_error=p95,
        maximum_absolute_error=max(maxima),
    )


def test_lexicographic_sort_with_ties() -> bool:
    device = require_cuda()
    x = torch.tensor(
        [[1, 2, 0], [1, 1, 5], [0, 9, 9], [1, 1, 4],
         [1, 2, 0], [-1, 3, 2], [1, 1, 4]],
        dtype=torch.float32,
        device=device,
    )
    permutation = torch.tensor([6, 4, 2, 0, 5, 3, 1], device=device)
    return torch.equal(
        lexicographic_sort_rows(x),
        lexicographic_sort_rows(x[permutation]),
    )


@torch.no_grad()
def canonization_invariance_max_error(trials: int = 100) -> float:
    device = require_cuda()
    _seed_all(DEFAULT_SEED)
    model = CanonizationInvariantMLP().to(device).eval()
    generator = _cuda_generator(DEFAULT_SEED)
    maximum = 0.0
    for _ in range(trials):
        x = torch.randn(
            DEFAULT_SCALABLE_N, DEFAULT_D, device=device, generator=generator
        )
        permutation = torch.randperm(
            DEFAULT_SCALABLE_N, device=device, generator=generator
        )
        maximum = max(
            maximum,
            float((model(x[permutation]) - model(x)).abs().max().item()),
        )
    return maximum


@torch.no_grad()
def symmetrization_invariance_max_error() -> tuple[float, float]:
    device = require_cuda()
    _seed_all(DEFAULT_SEED)
    base = _new_symmetrization_base().eval()
    model = SymmetrizedInvariantModel(base).to(device).eval()
    generator = _cuda_generator(DEFAULT_SEED)
    x = torch.randn(
        DEFAULT_SYMMETRIZATION_N, DEFAULT_D, device=device, generator=generator
    )
    sigma = torch.randperm(
        DEFAULT_SYMMETRIZATION_N, device=device, generator=generator
    )
    invariant_error = float((model(x[sigma]) - model(x)).abs().max().item())
    base_error = float((base(x[sigma]) - base(x)).abs().max().item())
    return invariant_error, base_error


def test_full_permutation_group_structure() -> bool:
    model = SymmetrizedInvariantModel(_new_symmetrization_base()).to(require_cuda())
    indices = model.permutation_indices
    expected = torch.arange(
        DEFAULT_SYMMETRIZATION_N, device=indices.device
    ).expand(DEFAULT_FULL_GROUP_SIZE, -1)
    return (
        tuple(indices.shape)
        == (DEFAULT_FULL_GROUP_SIZE, DEFAULT_SYMMETRIZATION_N)
        and torch.unique(indices, dim=0).shape[0] == DEFAULT_FULL_GROUP_SIZE
        and torch.equal(torch.sort(indices, dim=1).values, expected)
    )


def test_full_symmetrization_gradients() -> bool:
    device = require_cuda()
    _seed_all(DEFAULT_SEED)
    base = _new_symmetrization_base()
    model = SymmetrizedInvariantModel(base).to(device)
    x = torch.randn(
        DEFAULT_SYMMETRIZATION_N, DEFAULT_D, device=device
    )
    model(x).sum().backward()
    return all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in base.parameters()
    )


def test_sampled_subset_structure() -> bool:
    first = SampledSymmetrizedModel(
        _new_symmetrization_base(), DEFAULT_SUBSET_SIZE, DEFAULT_SEED
    ).to(require_cuda())
    second = SampledSymmetrizedModel(
        _new_symmetrization_base(), DEFAULT_SUBSET_SIZE, DEFAULT_SEED
    ).to(require_cuda())
    indices = first.permutation_indices
    return (
        tuple(indices.shape)
        == (DEFAULT_SUBSET_SIZE, DEFAULT_SYMMETRIZATION_N)
        and torch.unique(indices, dim=0).shape[0] == DEFAULT_SUBSET_SIZE
        and torch.equal(indices, second.permutation_indices)
        and DEFAULT_SUBSET_SIZE / DEFAULT_FULL_GROUP_SIZE == 0.05
    )


@torch.no_grad()
def deepsets_errors() -> tuple[float, float, float]:
    device = require_cuda()
    _seed_all(DEFAULT_SEED)
    layer = DeepSetsEquivariantLinear(
        DEFAULT_D, DEFAULT_DEEPSETS_HIDDEN_DIM
    ).to(device).eval()
    backbone = DeepSetsEquivariantBackbone().to(device).eval()
    classifier = DeepSetsInvariantClassifier().to(device).eval()
    generator = _cuda_generator(DEFAULT_SEED)
    x = torch.randn(
        DEFAULT_SCALABLE_N, DEFAULT_D, device=device, generator=generator
    )
    permutation = torch.randperm(
        DEFAULT_SCALABLE_N, device=device, generator=generator
    )
    single = float(
        (layer(x[permutation]) - layer(x)[permutation]).abs().max().item()
    )
    stack = float(
        (backbone(x[permutation]) - backbone(x)[permutation]).abs().max().item()
    )
    pooled = float(
        (classifier(x[permutation]) - classifier(x)).abs().max().item()
    )
    return single, stack, pooled


def test_deepsets_finite_gradients() -> bool:
    device = require_cuda()
    _seed_all(DEFAULT_SEED)
    model = DeepSetsInvariantClassifier().to(device)
    x = torch.randn(
        DEFAULT_SCALABLE_N,
        DEFAULT_D,
        device=device,
        requires_grad=True,
    )
    model(x).square().sum().backward()
    return (
        x.grad is not None
        and bool(torch.isfinite(x.grad).all())
        and all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        )
    )


def architecture_parameter_counts() -> dict[str, int]:
    device = require_cuda()
    return {
        "q3a_canonization": parameter_count(
            CanonizationInvariantMLP().to(device)
        ),
        "q3b_full_symmetrization": parameter_count(
            _new_symmetrization_base()
        ),
        "q3c_sampled_symmetrization": parameter_count(
            _new_symmetrization_base()
        ),
        "q3d_deepsets": parameter_count(
            DeepSetsInvariantClassifier().to(device)
        ),
        "q3e_augmentation": parameter_count(
            FlattenedMLP(
                DEFAULT_SCALABLE_N,
                DEFAULT_D,
                DEFAULT_VARIANCE_OUTPUT_DIM,
                DEFAULT_SCALABLE_HIDDEN_DIMS,
            ).to(device)
        ),
        "q3e_n7_matched": parameter_count(
            FlattenedMLP(
                DEFAULT_Q3E_COMPARISON_N,
                DEFAULT_D,
                DEFAULT_VARIANCE_OUTPUT_DIM,
                DEFAULT_Q3E_COMPARISON_HIDDEN_DIMS,
            ).to(device)
        ),
    }


def run_q3a_tests() -> bool:
    error = canonization_invariance_max_error()
    passed = error <= DEFAULT_EXACT_ATOL and test_lexicographic_sort_with_ties()
    print(
        "Q3(a):",
        "PASS" if passed else "FAIL",
        f"(cuda, n=256, MLP=768->64->64->40, params=55976, "
        f"atol=1e-5, max_abs_error={error:.12g})",
    )
    return passed


def run_q3b_tests() -> bool:
    invariant_error, base_error = symmetrization_invariance_max_error()
    structure = test_full_permutation_group_structure()
    gradients = test_full_symmetrization_gradients()
    passed = (
        invariant_error <= DEFAULT_EXACT_ATOL
        and base_error > DEFAULT_EXACT_ATOL
        and structure
        and gradients
    )
    print(
        "Q3(b):",
        "PASS" if passed else "FAIL",
        f"(cuda, n=7, 5040 permutations, MLP=21->207->207->40, "
        f"params=55930, atol=1e-5, max_abs_error={invariant_error:.12g}, "
        f"base_error={base_error:.12g})",
    )
    return passed


def run_q3c_tests() -> tuple[bool, Q3CMonteCarloMetrics]:
    metrics = q3c_monte_carlo_validation()
    structure = test_sampled_subset_structure()
    passed = (
        metrics.empirical_rms_error <= DEFAULT_APPROXIMATE_ATOL
        and metrics.relative_prediction_gap <= 0.15
        and structure
    )
    print(
        "Q3(c):",
        "PASS" if passed else "FAIL",
        f"(cuda, n=7, B=252=5%, MLP=21->207->207->40, params=55930, "
        f"average-RMS atol=1e-2, predicted={metrics.predicted_rms_error:.12g}, "
        f"empirical={metrics.empirical_rms_error:.12g}, "
        f"gap={100*metrics.relative_prediction_gap:.3f}%, "
        f"p95={metrics.p95_example_rms_error:.12g}, "
        f"max_diagnostic={metrics.maximum_absolute_error:.12g})",
    )
    return passed, metrics


def run_q3d_tests() -> bool:
    single, stack, pooled = deepsets_errors()
    gradients = test_deepsets_finite_gradients()
    counts = architecture_parameter_counts()
    passed = (
        max(single, stack, pooled) <= DEFAULT_EXACT_ATOL
        and gradients
        and counts["q3d_deepsets"] == 55464
    )
    print(
        "Q3(d):",
        "PASS" if passed else "FAIL",
        f"(cuda, n=256, equivariant 3->128->128, mean pool, "
        f"head 128->128->40, params=55464, atol=1e-5, "
        f"single={single:.12g}, stack={stack:.12g}, pooled={pooled:.12g})",
    )
    return passed


def run_q3e_tests() -> tuple[bool, Q3EExperimentResult]:
    result = run_q3e_experiment()
    finite = all(
        math.isfinite(value)
        for metrics in (
            result.initial_metrics,
            result.augmented_metrics,
            result.control_metrics,
            result.n7_initial_metrics,
            result.n7_augmented_metrics,
            result.n7_control_metrics,
        )
        for value in _metrics_to_dict(metrics).values()
    )
    passed = finite and result.plot_path.is_file() and result.results_path.is_file()
    print(
        "Q3(e):",
        "COMPLETE" if passed else "FAIL",
        f"(cuda, n=256, MLP=768->64->64->3, params=53571, "
        f"direct mean-L2 invariance metric, augmented test MSE="
        f"{result.augmented_metrics.approximation_mse:.12g}, "
        f"augmented E_inv="
        f"{result.augmented_metrics.mean_l2_invariance_error:.12g}, "
        f"control test MSE={result.control_metrics.approximation_mse:.12g}, "
        f"control E_inv="
        f"{result.control_metrics.mean_l2_invariance_error:.12g}; "
        f"n=7 matched params, augmented test MSE="
        f"{result.n7_augmented_metrics.approximation_mse:.12g}, "
        f"augmented E_inv="
        f"{result.n7_augmented_metrics.mean_l2_invariance_error:.12g}, "
        f"control test MSE={result.n7_control_metrics.approximation_mse:.12g}, "
        f"control E_inv="
        f"{result.n7_control_metrics.mean_l2_invariance_error:.12g})",
    )
    return passed, result


def run_q3_tests() -> bool:
    device = require_cuda()
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    counts = architecture_parameter_counts()
    expected_counts = {
        "q3a_canonization": 55976,
        "q3b_full_symmetrization": 55930,
        "q3c_sampled_symmetrization": 55930,
        "q3d_deepsets": 55464,
        "q3e_augmentation": 53571,
        "q3e_n7_matched": 53571,
    }
    counts_passed = counts == expected_counts
    print("Q3/Q4 architecture parameter counts:", counts)
    q3c_passed, _ = run_q3c_tests()
    q3e_passed, _ = run_q3e_tests()
    return all(
        (
            counts_passed,
            run_q3a_tests(),
            run_q3b_tests(),
            q3c_passed,
            run_q3d_tests(),
            q3e_passed,
        )
    )


if __name__ == "__main__":
    raise SystemExit(0 if run_q3_tests() else 1)
