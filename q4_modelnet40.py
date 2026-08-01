"""HW4 Question 4 ModelNet40 benchmark.

This module implements the mandatory comparison of five permutation-invariant
approaches and the approved surface-normal bonus rerun. It deliberately reuses
the Question 3 model components so the symmetry mechanisms remain aligned.

The large ModelNet40 archive and all generated experiment artifacts live under
the ignored ``_qa`` directory. Neural-network execution is CUDA-only and never
silently falls back to the CPU.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence
from zipfile import ZipFile

import matplotlib
import numpy as np
import torch
from torch import Tensor, nn

from q3_set_networks import (
    CanonizationInvariantMLP,
    DeepSetsInvariantClassifier,
    FlattenedMLP,
    SampledSymmetrizedModel,
    SymmetrizedInvariantModel,
    apply_row_permutations,
    draw_row_permutations,
    parameter_count,
    require_cuda,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


SEED = 2319
CLASS_COUNT = 40
POSITION_DIM = 3
MANDATORY_FEATURE_DIM = 3
NORMAL_FEATURE_DIM = 6
FEATURE_DIM = MANDATORY_FEATURE_DIM
SCALABLE_POINT_COUNT = 256
SYMMETRIZATION_POINT_COUNT = 7
SELECTED_SIZES = (5, 10, 50)
MAX_SELECTED_PER_CLASS = max(SELECTED_SIZES)

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 2000
EARLY_STOPPING_PATIENCE = 100
TIME_LIMIT_SECONDS = 30 * 60

FULL_PERMUTATION_COUNT = math.factorial(SYMMETRIZATION_POINT_COUNT)
SAMPLED_PERMUTATION_COUNT = FULL_PERMUTATION_COUNT // 20
PERMUTATION_CHUNK_SIZE = 1024

MODELNET40_URL = (
    "https://www.dropbox.com/scl/fi/qcddtiathbfcp6v3wimoi/"
    "modelnet40_normal_resampled.zip?"
    "rlkey=3t9jz6lr954wxmfj4xp6u9m2a&dl=1"
)
EXPECTED_ARCHIVE_SHA256 = (
    "DCA19B495658331BDD1656527AF7EA6D8CC4162D871E00444A7BFD945C96C9D3"
)

ARCHITECTURES = (
    "canonization",
    "full_symmetrization",
    "sampled_symmetrization",
    "equivariant",
    "augmentation",
)
ARCHITECTURE_LABELS = {
    "canonization": "Canonization",
    "full_symmetrization": "Full symmetrization",
    "sampled_symmetrization": "Sampled symmetrization",
    "equivariant": "Equivariant + mean pooling",
    "augmentation": "Permutation augmentation",
}
ARCHITECTURE_POINT_COUNTS = {
    "canonization": SCALABLE_POINT_COUNT,
    "full_symmetrization": SYMMETRIZATION_POINT_COUNT,
    "sampled_symmetrization": SYMMETRIZATION_POINT_COUNT,
    "equivariant": SCALABLE_POINT_COUNT,
    "augmentation": SCALABLE_POINT_COUNT,
}


def _mlp_parameter_count(
    input_dim: int, hidden_dims: Sequence[int], output_dim: int
) -> int:
    widths = (input_dim, *hidden_dims, output_dim)
    return sum(
        input_width * output_width + output_width
        for input_width, output_width in zip(widths, widths[1:])
    )


def _expected_parameter_counts(feature_dim: int) -> dict[str, int]:
    flattened = _mlp_parameter_count(
        SCALABLE_POINT_COUNT * feature_dim, (64, 64), CLASS_COUNT
    )
    symmetrized = _mlp_parameter_count(
        SYMMETRIZATION_POINT_COUNT * feature_dim,
        (207, 207),
        CLASS_COUNT,
    )
    equivariant = (
        2 * feature_dim * 128
        + 128
        + 2 * 128 * 128
        + 128
        + _mlp_parameter_count(128, (128,), CLASS_COUNT)
    )
    return {
        "canonization": flattened,
        "full_symmetrization": symmetrized,
        "sampled_symmetrization": symmetrized,
        "equivariant": equivariant,
        "augmentation": flattened,
    }


EXPECTED_PARAMETER_COUNTS = _expected_parameter_counts(FEATURE_DIM)


def configure_feature_dimension(feature_dim: int) -> None:
    """Select the approved XYZ-only or XYZ-plus-normal experiment mode."""

    if feature_dim not in {MANDATORY_FEATURE_DIM, NORMAL_FEATURE_DIM}:
        raise ValueError(f"Unsupported feature dimension: {feature_dim}")
    global FEATURE_DIM, EXPECTED_PARAMETER_COUNTS
    FEATURE_DIM = feature_dim
    EXPECTED_PARAMETER_COUNTS = _expected_parameter_counts(feature_dim)


@dataclass(frozen=True)
class DatasetSplit:
    selected_per_class: int
    training_indices: Tensor
    validation_indices: Tensor


@dataclass(frozen=True)
class EvaluationMetrics:
    cross_entropy: float
    overall_accuracy: float
    mean_class_accuracy: float
    inference_seconds: float
    fixed_permutation_accuracy: float | None = None


@dataclass(frozen=True)
class TrainingOutcome:
    status: str
    stop_reason: str
    completed_epochs: int
    best_epoch: int
    best_validation_cross_entropy: float
    training_seconds: float
    training_cross_entropy: tuple[float, ...]
    validation_cross_entropy: tuple[float, ...]


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_modelnet40(archive_path: Path) -> None:
    """Download the assignment-provided archive with a resumable temp target."""

    if archive_path.is_file():
        return
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_suffix(archive_path.suffix + ".part")
    request = urllib.request.Request(
        MODELNET40_URL,
        headers={"User-Agent": "DeepLearningGroups-HW4/1.0"},
    )
    mode = "ab" if partial.exists() else "wb"
    start = partial.stat().st_size if partial.exists() else 0
    if start:
        request.add_header("Range", f"bytes={start}-")
    response = urllib.request.urlopen(request)
    if start and getattr(response, "status", None) != 206:
        mode = "wb"
    with response, partial.open(mode) as target:
        while block := response.read(8 * 1024 * 1024):
            target.write(block)
    partial.replace(archive_path)


def _class_from_shape_id(shape_id: str) -> str:
    class_name, separator, numeric_id = shape_id.rpartition("_")
    if not separator or not class_name or not numeric_id.isdigit():
        raise ValueError(f"Unexpected ModelNet40 shape id: {shape_id!r}")
    return class_name


def _unique_member_ending(names: Sequence[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one ZIP member ending in {suffix!r}, got {matches}"
        )
    return matches[0]


def _read_nonempty_lines(archive: ZipFile, member: str) -> list[str]:
    return [
        line.strip()
        for line in archive.read(member).decode("utf-8-sig").splitlines()
        if line.strip()
    ]


def _point_member_map(names: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in names:
        path = PurePosixPath(name)
        if path.suffix.lower() != ".txt" or len(path.parts) < 2:
            continue
        if path.stem in mapping:
            raise RuntimeError(f"Duplicate point-cloud file stem: {path.stem}")
        mapping[path.stem] = name
    return mapping


def _read_point_features256(archive: ZipFile, member: str) -> np.ndarray:
    with archive.open(member) as binary_stream:
        text_stream = io.TextIOWrapper(binary_stream, encoding="utf-8")
        point_features = np.loadtxt(
            text_stream,
            delimiter=",",
            dtype=np.float32,
            max_rows=SCALABLE_POINT_COUNT,
            usecols=tuple(range(FEATURE_DIM)),
        )
    if point_features.shape != (SCALABLE_POINT_COUNT, FEATURE_DIM):
        raise RuntimeError(
            f"Expected {(SCALABLE_POINT_COUNT, FEATURE_DIM)} point features in "
            f"{member}, got {point_features.shape}"
        )
    if not np.isfinite(point_features).all():
        raise RuntimeError(f"Non-finite point feature found in {member}")
    return point_features


def prepare_compact_dataset(
    archive_path: Path,
    cache_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Cache only the approved training subset and complete official test set."""

    if cache_path.is_file() and not force:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        validate_compact_dataset(cached)
        return cached
    if not archive_path.is_file():
        raise FileNotFoundError(f"ModelNet40 archive not found: {archive_path}")

    source_sha256 = _sha256(archive_path)
    if source_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            "The ModelNet40 archive hash differs from the assignment-provided "
            f"file: {source_sha256}"
        )

    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        train_member = _unique_member_ending(names, "modelnet40_train.txt")
        test_member = _unique_member_ending(names, "modelnet40_test.txt")
        class_member = _unique_member_ending(
            names, "modelnet40_shape_names.txt"
        )
        train_ids = _read_nonempty_lines(archive, train_member)
        test_ids = _read_nonempty_lines(archive, test_member)
        class_names = _read_nonempty_lines(archive, class_member)
        if len(class_names) != CLASS_COUNT or len(set(class_names)) != CLASS_COUNT:
            raise RuntimeError("Expected exactly 40 unique ModelNet40 classes")

        training_by_class: dict[str, list[str]] = {name: [] for name in class_names}
        for shape_id in train_ids:
            training_by_class[_class_from_shape_id(shape_id)].append(shape_id)

        selected_training_ids: list[str] = []
        selected_training_labels: list[int] = []
        for class_index, class_name in enumerate(class_names):
            candidates = sorted(training_by_class[class_name])
            if len(candidates) < MAX_SELECTED_PER_CLASS:
                raise RuntimeError(
                    f"Class {class_name} has only {len(candidates)} training samples"
                )
            generator = torch.Generator().manual_seed(SEED + class_index)
            order = torch.randperm(len(candidates), generator=generator).tolist()
            selected = [candidates[index] for index in order[:MAX_SELECTED_PER_CLASS]]
            selected_training_ids.extend(selected)
            selected_training_labels.extend(
                [class_index] * MAX_SELECTED_PER_CLASS
            )

        class_to_index = {name: index for index, name in enumerate(class_names)}
        test_labels = [
            class_to_index[_class_from_shape_id(shape_id)] for shape_id in test_ids
        ]
        member_by_shape = _point_member_map(names)
        required_ids = selected_training_ids + test_ids
        missing = [shape_id for shape_id in required_ids if shape_id not in member_by_shape]
        if missing:
            raise RuntimeError(f"Missing point-cloud members: {missing[:5]}")

        arrays: list[np.ndarray] = []
        for index, shape_id in enumerate(required_ids, start=1):
            arrays.append(
                _read_point_features256(archive, member_by_shape[shape_id])
            )
            if index % 250 == 0 or index == len(required_ids):
                print(f"Prepared {index}/{len(required_ids)} point clouds")

    all_points = torch.from_numpy(np.stack(arrays)).contiguous()
    training_count = len(selected_training_ids)
    payload: dict[str, Any] = {
        "format_version": 2,
        "source_url": MODELNET40_URL,
        "source_archive_sha256": source_sha256,
        "seed": SEED,
        "feature_dimension": FEATURE_DIM,
        "feature_columns": (
            ["x", "y", "z"]
            if FEATURE_DIM == MANDATORY_FEATURE_DIM
            else ["x", "y", "z", "nx", "ny", "nz"]
        ),
        "class_names": class_names,
        "training_ids": selected_training_ids,
        "training_points": all_points[:training_count],
        "training_labels": torch.tensor(selected_training_labels, dtype=torch.long),
        "test_ids": test_ids,
        "test_points": all_points[training_count:],
        "test_labels": torch.tensor(test_labels, dtype=torch.long),
    }
    validate_compact_dataset(payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    _write_json(
        cache_path.with_suffix(".manifest.json"),
        {
            key: value
            for key, value in payload.items()
            if key not in {
                "training_points",
                "training_labels",
                "test_points",
                "test_labels",
            }
        }
        | {
            "training_tensor_shape": list(payload["training_points"].shape),
            "test_tensor_shape": list(payload["test_points"].shape),
        },
    )
    return payload


def validate_compact_dataset(dataset: dict[str, Any]) -> None:
    class_names = dataset["class_names"]
    training_points = dataset["training_points"]
    training_labels = dataset["training_labels"]
    test_points = dataset["test_points"]
    test_labels = dataset["test_labels"]
    recorded_feature_dim = dataset.get(
        "feature_dimension", int(training_points.shape[-1])
    )
    if recorded_feature_dim != FEATURE_DIM:
        raise RuntimeError(
            f"Cache contains d={recorded_feature_dim}, but the active mode "
            f"requires d={FEATURE_DIM}"
        )
    if len(class_names) != CLASS_COUNT:
        raise RuntimeError("Compact dataset does not contain 40 classes")
    expected_training = CLASS_COUNT * MAX_SELECTED_PER_CLASS
    if tuple(training_points.shape) != (
        expected_training,
        SCALABLE_POINT_COUNT,
        FEATURE_DIM,
    ):
        raise RuntimeError(f"Unexpected training tensor shape: {training_points.shape}")
    if tuple(test_points.shape) != (2468, SCALABLE_POINT_COUNT, FEATURE_DIM):
        raise RuntimeError(f"Unexpected test tensor shape: {test_points.shape}")
    if tuple(training_labels.shape) != (expected_training,):
        raise RuntimeError("Unexpected training-label shape")
    if tuple(test_labels.shape) != (2468,):
        raise RuntimeError("Unexpected test-label shape")
    expected_labels = torch.arange(CLASS_COUNT).repeat_interleave(
        MAX_SELECTED_PER_CLASS
    )
    if not torch.equal(training_labels, expected_labels):
        raise RuntimeError("Cached training examples are not class-grouped and balanced")
    if not (torch.isfinite(training_points).all() and torch.isfinite(test_points).all()):
        raise RuntimeError("Compact dataset contains non-finite point features")


def make_split(selected_per_class: int) -> DatasetSplit:
    if selected_per_class not in SELECTED_SIZES:
        raise ValueError(f"Unsupported selected size: {selected_per_class}")
    training_indices: list[int] = []
    validation_indices: list[int] = []
    for class_index in range(CLASS_COUNT):
        offset = class_index * MAX_SELECTED_PER_CLASS
        for position in range(selected_per_class):
            target = (
                validation_indices if position % 5 == 4 else training_indices
            )
            target.append(offset + position)
    return DatasetSplit(
        selected_per_class=selected_per_class,
        training_indices=torch.tensor(training_indices, dtype=torch.long),
        validation_indices=torch.tensor(validation_indices, dtype=torch.long),
    )


def validate_nested_splits(dataset: dict[str, Any]) -> None:
    splits = {size: make_split(size) for size in SELECTED_SIZES}
    labels = dataset["training_labels"]
    for size, split in splits.items():
        expected_train_per_class = 4 * (size // 5)
        expected_validation_per_class = size // 5
        train_counts = torch.bincount(
            labels[split.training_indices], minlength=CLASS_COUNT
        )
        validation_counts = torch.bincount(
            labels[split.validation_indices], minlength=CLASS_COUNT
        )
        if not torch.equal(
            train_counts, torch.full_like(train_counts, expected_train_per_class)
        ):
            raise RuntimeError(f"Unbalanced training split for selected size {size}")
        if not torch.equal(
            validation_counts,
            torch.full_like(validation_counts, expected_validation_per_class),
        ):
            raise RuntimeError(f"Unbalanced validation split for selected size {size}")
        if set(split.training_indices.tolist()) & set(
            split.validation_indices.tolist()
        ):
            raise RuntimeError("Training and validation splits overlap")

    for smaller, larger in zip(SELECTED_SIZES, SELECTED_SIZES[1:]):
        if not set(splits[smaller].training_indices.tolist()) <= set(
            splits[larger].training_indices.tolist()
        ):
            raise RuntimeError("Training splits are not nested")
        if not set(splits[smaller].validation_indices.tolist()) <= set(
            splits[larger].validation_indices.tolist()
        ):
            raise RuntimeError("Validation splits are not nested")


def _lexicographic_sort_by_position(features: Tensor) -> Tensor:
    """Sort rows by XYZ while carrying every attached feature with its point."""

    if (
        features.ndim < 2
        or features.shape[-2] < 1
        or features.shape[-1] < POSITION_DIM
    ):
        raise ValueError(
            f"Expected shape (..., n, d) with d >= {POSITION_DIM}, got "
            f"{tuple(features.shape)}"
        )
    point_count, feature_dim = features.shape[-2:]
    batch_shape = features.shape[:-2]
    indices = torch.arange(point_count, device=features.device).expand(
        *batch_shape, point_count
    )
    for feature in range(POSITION_DIM - 1, -1, -1):
        values = features[..., feature]
        ordered_values = torch.gather(values, -1, indices)
        order = torch.argsort(ordered_values, dim=-1, stable=True)
        indices = torch.gather(indices, -1, order)
    gather_indices = indices.unsqueeze(-1).expand(
        *batch_shape, point_count, feature_dim
    )
    return torch.gather(features, -2, gather_indices)


class PositionCanonizationInvariantMLP(nn.Module):
    """XYZ canonization that keeps normals attached to their spatial points."""

    def __init__(
        self,
        n: int,
        d: int,
        output_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.n = n
        self.d = d
        self.mlp = FlattenedMLP(n, d, output_dim, hidden_dims)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim < 2 or tuple(features.shape[-2:]) != (self.n, self.d):
            raise ValueError(
                f"Expected shape (..., {self.n}, {self.d}), got "
                f"{tuple(features.shape)}"
            )
        return self.mlp(_lexicographic_sort_by_position(features))


def build_model(architecture: str, *, seed: int = SEED) -> nn.Module:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    _seed_all(seed)
    if architecture == "canonization":
        if FEATURE_DIM == MANDATORY_FEATURE_DIM:
            model: nn.Module = CanonizationInvariantMLP(
                n=SCALABLE_POINT_COUNT,
                d=FEATURE_DIM,
                output_dim=CLASS_COUNT,
                hidden_dims=(64, 64),
            )
        else:
            model = PositionCanonizationInvariantMLP(
                n=SCALABLE_POINT_COUNT,
                d=FEATURE_DIM,
                output_dim=CLASS_COUNT,
                hidden_dims=(64, 64),
            )
    elif architecture == "full_symmetrization":
        base = FlattenedMLP(
            SYMMETRIZATION_POINT_COUNT,
            FEATURE_DIM,
            CLASS_COUNT,
            (207, 207),
        )
        model = SymmetrizedInvariantModel(
            base, permutation_chunk_size=PERMUTATION_CHUNK_SIZE
        )
    elif architecture == "sampled_symmetrization":
        base = FlattenedMLP(
            SYMMETRIZATION_POINT_COUNT,
            FEATURE_DIM,
            CLASS_COUNT,
            (207, 207),
        )
        model = SampledSymmetrizedModel(
            base,
            subset_size=SAMPLED_PERMUTATION_COUNT,
            subset_seed=SEED,
            permutation_chunk_size=PERMUTATION_CHUNK_SIZE,
        )
    elif architecture == "equivariant":
        model = DeepSetsInvariantClassifier(
            input_dim=FEATURE_DIM,
            hidden_dim=128,
            output_dim=CLASS_COUNT,
        )
    else:
        model = FlattenedMLP(
            SCALABLE_POINT_COUNT,
            FEATURE_DIM,
            CLASS_COUNT,
            (64, 64),
        )
    count = parameter_count(model)
    if count != EXPECTED_PARAMETER_COUNTS[architecture]:
        raise RuntimeError(
            f"{architecture} has {count} parameters; expected "
            f"{EXPECTED_PARAMETER_COUNTS[architecture]}"
        )
    return model.to(require_cuda())


def _model_input(points: Tensor, architecture: str) -> Tensor:
    return points[:, : ARCHITECTURE_POINT_COUNTS[architecture], :]


def _validation_cross_entropy(
    model: nn.Module,
    points: Tensor,
    labels: Tensor,
    architecture: str,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for start in range(0, labels.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, labels.shape[0])
            batch_points = _model_input(
                points[start:end].to(require_cuda(), non_blocking=True), architecture
            )
            batch_labels = labels[start:end].to(require_cuda(), non_blocking=True)
            logits = model(batch_points)
            total_loss += float(
                nn.functional.cross_entropy(logits, batch_labels, reduction="sum").item()
            )
    return total_loss / labels.shape[0]


def train_model(
    model: nn.Module,
    training_points: Tensor,
    training_labels: Tensor,
    validation_points: Tensor,
    validation_labels: Tensor,
    architecture: str,
    *,
    selected_per_class: int,
    maximum_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    time_limit_seconds: float = TIME_LIMIT_SECONDS,
) -> TrainingOutcome:
    device = require_cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    batch_order_generator = torch.Generator().manual_seed(
        SEED + 100_000 + selected_per_class
    )
    augmentation_generator = torch.Generator(device=device).manual_seed(
        SEED + 200_000 + selected_per_class
    )
    training_points = training_points.contiguous().pin_memory()
    training_labels = training_labels.contiguous().pin_memory()
    validation_points = validation_points.contiguous().pin_memory()
    validation_labels = validation_labels.contiguous().pin_memory()

    best_validation = math.inf
    best_epoch = 0
    best_state = _cpu_state_dict(model)
    last_completed_state = copy.deepcopy(best_state)
    last_completed_epoch = 0
    stale_epochs = 0
    training_history: list[float] = []
    validation_history: list[float] = []
    status = "complete"
    stop_reason = "maximum_epochs"

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        order = torch.randperm(
            training_labels.shape[0], generator=batch_order_generator
        )
        epoch_loss = 0.0
        processed = 0
        interrupted = False
        for start in range(0, order.shape[0], BATCH_SIZE):
            if time.perf_counter() - started >= time_limit_seconds:
                interrupted = True
                break
            indices = order[start : start + BATCH_SIZE]
            batch_points = _model_input(
                training_points[indices].to(device, non_blocking=True), architecture
            )
            batch_labels = training_labels[indices].to(device, non_blocking=True)
            if architecture == "augmentation":
                permutations = draw_row_permutations(
                    batch_points.shape[0],
                    SCALABLE_POINT_COUNT,
                    generator=augmentation_generator,
                )
                batch_points = apply_row_permutations(batch_points, permutations)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_points)
            loss = nn.functional.cross_entropy(logits, batch_labels)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss for {architecture}, size "
                    f"{selected_per_class}, epoch {epoch}"
                )
            loss.backward()
            optimizer.step()
            batch_count = batch_labels.shape[0]
            epoch_loss += float(loss.item()) * batch_count
            processed += batch_count

        if interrupted:
            model.load_state_dict(last_completed_state)
            status = "OOT"
            stop_reason = "time_limit_during_epoch"
            break
        if processed != training_labels.shape[0]:
            raise RuntimeError("An epoch did not process the complete training split")

        training_loss = epoch_loss / processed
        validation_loss = _validation_cross_entropy(
            model,
            validation_points,
            validation_labels,
            architecture,
        )
        training_history.append(training_loss)
        validation_history.append(validation_loss)
        last_completed_epoch = epoch
        last_completed_state = _cpu_state_dict(model)

        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(last_completed_state)
            stale_epochs = 0
        else:
            stale_epochs += 1

        elapsed = time.perf_counter() - started
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"{architecture} selected={selected_per_class}: epoch={epoch}, "
                f"train_ce={training_loss:.6f}, val_ce={validation_loss:.6f}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        if elapsed >= time_limit_seconds:
            status = "OOT"
            stop_reason = "time_limit_after_completed_epoch"
            break
        if stale_epochs >= patience:
            stop_reason = "early_stopping"
            break

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    if status == "OOT":
        model.load_state_dict(last_completed_state)
    else:
        model.load_state_dict(best_state)
    return TrainingOutcome(
        status=status,
        stop_reason=stop_reason,
        completed_epochs=last_completed_epoch,
        best_epoch=best_epoch,
        best_validation_cross_entropy=best_validation,
        training_seconds=training_seconds,
        training_cross_entropy=tuple(training_history),
        validation_cross_entropy=tuple(validation_history),
    )


def evaluate_test_once(
    model: nn.Module,
    test_points: Tensor,
    test_labels: Tensor,
    architecture: str,
    *,
    fixed_permutation_diagnostic: bool,
) -> EvaluationMetrics:
    device = require_cuda()
    test_points = test_points.contiguous().pin_memory()
    test_labels = test_labels.contiguous().pin_memory()
    correct_by_class = torch.zeros(CLASS_COUNT, dtype=torch.long, device=device)
    total_by_class = torch.zeros(CLASS_COUNT, dtype=torch.long, device=device)
    permuted_correct = 0
    total_loss = 0.0
    total_correct = 0
    model.eval()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, test_labels.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, test_labels.shape[0])
            batch_points = _model_input(
                test_points[start:end].to(device, non_blocking=True), architecture
            )
            batch_labels = test_labels[start:end].to(device, non_blocking=True)
            logits = model(batch_points)
            total_loss += float(
                nn.functional.cross_entropy(logits, batch_labels, reduction="sum").item()
            )
            predictions = logits.argmax(dim=1)
            matches = predictions.eq(batch_labels)
            total_correct += int(matches.sum().item())
            total_by_class += torch.bincount(batch_labels, minlength=CLASS_COUNT)
            correct_by_class += torch.bincount(
                batch_labels[matches], minlength=CLASS_COUNT
            )

    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started

    if fixed_permutation_diagnostic:
        permutation_generator = torch.Generator(device=device).manual_seed(
            SEED + 900_000
        )
        with torch.inference_mode():
            for start in range(0, test_labels.shape[0], BATCH_SIZE):
                end = min(start + BATCH_SIZE, test_labels.shape[0])
                batch_points = _model_input(
                    test_points[start:end].to(device, non_blocking=True), architecture
                )
                batch_labels = test_labels[start:end].to(device, non_blocking=True)
                permutations = draw_row_permutations(
                    batch_points.shape[0],
                    ARCHITECTURE_POINT_COUNTS[architecture],
                    generator=permutation_generator,
                )
                permuted_logits = model(
                    apply_row_permutations(batch_points, permutations)
                )
                permuted_correct += int(
                    permuted_logits.argmax(dim=1).eq(batch_labels).sum().item()
                )
    class_accuracies = correct_by_class.float() / total_by_class.clamp_min(1)
    return EvaluationMetrics(
        cross_entropy=total_loss / test_labels.shape[0],
        overall_accuracy=total_correct / test_labels.shape[0],
        mean_class_accuracy=float(class_accuracies.mean().item()),
        inference_seconds=inference_seconds,
        fixed_permutation_accuracy=(
            permuted_correct / test_labels.shape[0]
            if fixed_permutation_diagnostic
            else None
        ),
    )


def _run_result_path(output_dir: Path, architecture: str, selected: int) -> Path:
    return output_dir / "runs" / f"{architecture}-selected-{selected}.json"


def run_experiment(
    dataset: dict[str, Any],
    output_dir: Path,
    architecture: str,
    selected_per_class: int,
    *,
    maximum_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    time_limit_seconds: float = TIME_LIMIT_SECONDS,
    force: bool = False,
) -> dict[str, Any]:
    result_path = _run_result_path(output_dir, architecture, selected_per_class)
    if result_path.is_file() and not force:
        print(f"Reusing completed result: {result_path}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    split = make_split(selected_per_class)
    points = dataset["training_points"]
    labels = dataset["training_labels"]
    training_points = points[split.training_indices]
    training_labels = labels[split.training_indices]
    validation_points = points[split.validation_indices]
    validation_labels = labels[split.validation_indices]

    model = build_model(architecture, seed=SEED)
    outcome = train_model(
        model,
        training_points,
        training_labels,
        validation_points,
        validation_labels,
        architecture,
        selected_per_class=selected_per_class,
        maximum_epochs=maximum_epochs,
        patience=patience,
        time_limit_seconds=time_limit_seconds,
    )
    metrics = evaluate_test_once(
        model,
        dataset["test_points"],
        dataset["test_labels"],
        architecture,
        fixed_permutation_diagnostic=architecture == "augmentation",
    )
    checkpoint_path = (
        output_dir
        / "checkpoints"
        / f"{architecture}-selected-{selected_per_class}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cpu_state_dict(model), checkpoint_path)

    payload: dict[str, Any] = {
        "architecture": architecture,
        "architecture_label": ARCHITECTURE_LABELS[architecture],
        "selected_examples_per_class": selected_per_class,
        "training_examples_per_class": 4 * (selected_per_class // 5),
        "validation_examples_per_class": selected_per_class // 5,
        "training_examples": int(training_labels.shape[0]),
        "validation_examples": int(validation_labels.shape[0]),
        "test_examples": int(dataset["test_labels"].shape[0]),
        "point_count": ARCHITECTURE_POINT_COUNTS[architecture],
        "feature_dimension": FEATURE_DIM,
        "uses_surface_normals": FEATURE_DIM == NORMAL_FEATURE_DIM,
        "canonization_sort_features": ["x", "y", "z"],
        "class_count": CLASS_COUNT,
        "parameter_count": parameter_count(model),
        "seed": SEED,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": maximum_epochs,
        "early_stopping_patience": patience,
        "time_limit_seconds": time_limit_seconds,
        "loss": "CrossEntropyLoss",
        "learning_rate_scheduler": None,
        "dropout": None,
        "training": asdict(outcome),
        "test": asdict(metrics),
        "checkpoint": str(checkpoint_path),
    }
    _write_json(result_path, payload)
    print(
        f"Completed {architecture}, selected={selected_per_class}: "
        f"status={outcome.status}, test_accuracy={metrics.overall_accuracy:.4f}, "
        f"train_seconds={outcome.training_seconds:.1f}",
        flush=True,
    )
    return payload


def save_sanity_plot(dataset: dict[str, Any], output_path: Path) -> None:
    class_names: list[str] = dataset["class_names"]
    labels: Tensor = dataset["test_labels"]
    points: Tensor = dataset["test_points"]
    selected_classes = ("airplane", "chair")
    figure = plt.figure(figsize=(10, 4.8))
    for panel, class_name in enumerate(selected_classes, start=1):
        class_index = class_names.index(class_name)
        example_index = int(torch.nonzero(labels == class_index, as_tuple=False)[0])
        cloud = points[example_index].numpy()
        axis = figure.add_subplot(1, 2, panel, projection="3d")
        axis.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=5, alpha=0.8)
        axis.set_title(class_name)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_box_aspect((1, 1, 1))
    figure.suptitle("ModelNet40 sanity check: first 256 XYZ points")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _load_results(output_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for selected in SELECTED_SIZES:
            path = _run_result_path(output_dir, architecture, selected)
            if path.is_file():
                results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def save_accuracy_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 6))
    for architecture in ARCHITECTURES:
        rows = sorted(
            (row for row in results if row["architecture"] == architecture),
            key=lambda row: row["selected_examples_per_class"],
        )
        if not rows:
            continue
        axis.plot(
            [row["selected_examples_per_class"] for row in rows],
            [100 * row["test"]["overall_accuracy"] for row in rows],
            marker="o",
            linewidth=2,
            label=ARCHITECTURE_LABELS[architecture],
        )
        for row in rows:
            if row["training"]["status"] == "OOT":
                axis.scatter(
                    [row["selected_examples_per_class"]],
                    [100 * row["test"]["overall_accuracy"]],
                    marker="x",
                    color="black",
                    s=70,
                    zorder=5,
                )
    axis.set_xticks(SELECTED_SIZES)
    axis.set_xlabel("Selected examples per class (before 20% validation holdout)")
    axis.set_ylabel("Official test accuracy (%)")
    axis.set_title("ModelNet40: test accuracy versus selected training-pool size")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for architecture in ARCHITECTURES:
        rows = sorted(
            (row for row in results if row["architecture"] == architecture),
            key=lambda row: row["selected_examples_per_class"],
        )
        if not rows:
            continue
        figure, axes = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 4.5))
        if len(rows) == 1:
            axes = [axes]
        for axis, row in zip(axes, rows):
            train_values = row["training"]["training_cross_entropy"]
            validation_values = row["training"]["validation_cross_entropy"]
            epochs = range(1, len(train_values) + 1)
            axis.plot(epochs, train_values, label="Training")
            axis.plot(epochs, validation_values, label="Validation")
            if row["training"]["best_epoch"]:
                axis.axvline(
                    row["training"]["best_epoch"],
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    label="Best validation epoch",
                )
            axis.set_yscale("log")
            axis.set_title(
                f"Selected per class: {row['selected_examples_per_class']}"
            )
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Cross-entropy")
        axes[0].legend()
        figure.suptitle(f"{ARCHITECTURE_LABELS[architecture]} learning curves")
        figure.tight_layout()
        figure.savefig(
            output_dir / f"{architecture}.png", dpi=160, bbox_inches="tight"
        )
        plt.close(figure)


def save_runtime_table(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "architecture",
        "selected_examples_per_class",
        "parameter_count",
        "status",
        "completed_epochs",
        "training_seconds",
        "inference_seconds",
        "overall_test_accuracy",
        "mean_class_accuracy",
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            results,
            key=lambda item: (
                ARCHITECTURES.index(item["architecture"]),
                item["selected_examples_per_class"],
            ),
        ):
            writer.writerow(
                {
                    "architecture": row["architecture"],
                    "selected_examples_per_class": row[
                        "selected_examples_per_class"
                    ],
                    "parameter_count": row["parameter_count"],
                    "status": row["training"]["status"],
                    "completed_epochs": row["training"]["completed_epochs"],
                    "training_seconds": row["training"]["training_seconds"],
                    "inference_seconds": row["test"]["inference_seconds"],
                    "overall_test_accuracy": row["test"]["overall_accuracy"],
                    "mean_class_accuracy": row["test"]["mean_class_accuracy"],
                }
            )


def save_results_markdown(results: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "| Architecture | Selected per class | Parameters | Overall accuracy | Mean-class accuracy | Epochs | Training time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        results,
        key=lambda item: (
            ARCHITECTURES.index(item["architecture"]),
            item["selected_examples_per_class"],
        ),
    ):
        lines.append(
            "| {label} | {selected} | {parameters:,} | {overall:.2f}% | "
            "{mean_class:.2f}% | {epochs} | {seconds:.1f} |".format(
                label=row["architecture_label"],
                selected=row["selected_examples_per_class"],
                parameters=row["parameter_count"],
                overall=100 * row["test"]["overall_accuracy"],
                mean_class=100 * row["test"]["mean_class_accuracy"],
                epochs=row["training"]["completed_epochs"],
                seconds=row["training"]["training_seconds"],
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_results(output_dir: Path) -> list[dict[str, Any]]:
    results = _load_results(output_dir)
    save_accuracy_plot(results, output_dir / "modelnet40_accuracy.png")
    save_training_curves(results, output_dir / "training_curves")
    save_runtime_table(results, output_dir / "modelnet40_runtime.csv")
    save_results_markdown(results, output_dir / "modelnet40_results_table.md")
    _write_json(
        output_dir / "modelnet40_results.json",
        {
            "source_url": MODELNET40_URL,
            "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "seed": SEED,
            "feature_dimension": FEATURE_DIM,
            "uses_surface_normals": FEATURE_DIM == NORMAL_FEATURE_DIM,
            "selected_sizes": list(SELECTED_SIZES),
            "results": results,
        },
    )
    return results


def reevaluate_saved_runs(
    dataset: dict[str, Any], output_dir: Path
) -> list[dict[str, Any]]:
    """Refresh test metrics from saved checkpoints without retraining."""

    results = _load_results(output_dir)
    for payload in results:
        architecture = payload["architecture"]
        checkpoint_path = Path(payload["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        model = build_model(architecture, seed=SEED)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        metrics = evaluate_test_once(
            model,
            dataset["test_points"],
            dataset["test_labels"],
            architecture,
            fixed_permutation_diagnostic=architecture == "augmentation",
        )
        payload["test"] = asdict(metrics)
        _write_json(
            _run_result_path(
                output_dir,
                architecture,
                payload["selected_examples_per_class"],
            ),
            payload,
        )
        del model
        torch.cuda.empty_cache()
    return summarize_results(output_dir)


def run_smoke_tests(dataset: dict[str, Any]) -> None:
    device = require_cuda()
    validate_compact_dataset(dataset)
    validate_nested_splits(dataset)
    sample_points = dataset["training_points"][:4].to(device)
    sample_labels = dataset["training_labels"][:4].to(device)

    if FEATURE_DIM == NORMAL_FEATURE_DIM:
        normals = sample_points[..., POSITION_DIM:]
        if torch.unique(normals.reshape(-1, POSITION_DIM), dim=0).shape[0] <= 1:
            raise RuntimeError("Surface normals are unexpectedly identical")
        synthetic = torch.tensor(
            [
                [2.0, 0.0, 0.0, 20.0, 21.0, 22.0],
                [0.0, 0.0, 0.0, 10.0, 11.0, 12.0],
                [1.0, 0.0, 0.0, 30.0, 31.0, 32.0],
            ],
            device=device,
        )
        sorted_synthetic = _lexicographic_sort_by_position(synthetic)
        expected_normal_ids = torch.tensor([10.0, 30.0, 20.0], device=device)
        if not torch.equal(
            sorted_synthetic[:, POSITION_DIM], expected_normal_ids
        ):
            raise RuntimeError("XYZ canonization detached normals from points")

    sampled_a = build_model("sampled_symmetrization")
    sampled_b = build_model("sampled_symmetrization")
    if not torch.equal(sampled_a.permutation_indices, sampled_b.permutation_indices):
        raise RuntimeError("Sampled-symmetrization subset is not reproducible")
    if sampled_a.permutation_indices.shape != (
        SAMPLED_PERMUTATION_COUNT,
        SYMMETRIZATION_POINT_COUNT,
    ):
        raise RuntimeError("Unexpected sampled permutation subset shape")
    if torch.unique(sampled_a.permutation_indices, dim=0).shape[0] != (
        SAMPLED_PERMUTATION_COUNT
    ):
        raise RuntimeError("Sampled permutation subset contains duplicates")
    del sampled_a, sampled_b

    canonization = build_model("canonization")
    augmentation = build_model("augmentation")
    canonization_values = list(canonization.mlp.network.state_dict().values())
    augmentation_values = list(augmentation.network.state_dict().values())
    if not all(
        torch.equal(left, right)
        for left, right in zip(canonization_values, augmentation_values)
    ):
        raise RuntimeError("Canonization and augmentation initial weights differ")
    full = build_model("full_symmetrization")
    sampled = build_model("sampled_symmetrization")
    if not all(
        torch.equal(full.base_model.state_dict()[key], value)
        for key, value in sampled.base_model.state_dict().items()
    ):
        raise RuntimeError("Full and sampled symmetrization initial weights differ")
    del canonization, augmentation, full, sampled

    for architecture in ARCHITECTURES:
        model = build_model(architecture)
        count = 2 if architecture == "full_symmetrization" else 4
        points = _model_input(sample_points[:count], architecture)
        logits = model(points)
        if logits.shape != (count, CLASS_COUNT):
            raise RuntimeError(
                f"Unexpected {architecture} output shape: {logits.shape}"
            )
        loss = nn.functional.cross_entropy(logits, sample_labels[:count])
        loss.backward()
        if not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"Invalid gradients for {architecture}")

        model.eval()
        generator = torch.Generator(device=device).manual_seed(SEED + 777)
        permutations = draw_row_permutations(
            count,
            ARCHITECTURE_POINT_COUNTS[architecture],
            generator=generator,
        )
        with torch.inference_mode():
            original = model(points)
            permuted = model(apply_row_permutations(points, permutations))
        max_error = float((original - permuted).abs().max().item())
        if architecture in {"canonization", "full_symmetrization", "equivariant"}:
            if max_error > 1e-5:
                raise RuntimeError(
                    f"{architecture} failed exact invariance smoke test: {max_error}"
                )
        print(
            f"Smoke {architecture}: params={parameter_count(model)}, "
            f"max_permutation_error={max_error:.8g}"
        )
        del model
        torch.cuda.empty_cache()

    first = draw_row_permutations(
        4,
        SCALABLE_POINT_COUNT,
        generator=torch.Generator(device=device).manual_seed(SEED + 888),
    )
    if torch.unique(first, dim=0).shape[0] != first.shape[0]:
        raise RuntimeError("Per-example augmentation permutations were not distinct")
    print("Q4 smoke tests: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "smoke", "run", "reevaluate", "summarize"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("_qa/data/modelnet40_normal_resampled.zip"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--use-normals",
        action="store_true",
        help=(
            "Use XYZ plus the attached surface normal as six input features. "
            "Use a separate cache and output directory for this mode."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--architectures", nargs="+", choices=ARCHITECTURES, default=ARCHITECTURES
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, choices=SELECTED_SIZES, default=SELECTED_SIZES
    )
    parser.add_argument("--maximum-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--time-limit-seconds", type=float, default=TIME_LIMIT_SECONDS
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_feature_dimension(
        NORMAL_FEATURE_DIM if args.use_normals else MANDATORY_FEATURE_DIM
    )
    if args.output_dir is None:
        args.output_dir = Path(
            "_qa/q4_modelnet40_normals"
            if args.use_normals
            else "_qa/q4_modelnet40"
        )
    if args.cache is None:
        args.cache = args.output_dir / (
            "modelnet40_xyz_normals256.pt"
            if args.use_normals
            else "modelnet40_xyz256.pt"
        )
    if args.download:
        download_modelnet40(args.archive)
    dataset = prepare_compact_dataset(
        args.archive, args.cache, force=args.force and args.command == "prepare"
    )
    validate_nested_splits(dataset)
    save_sanity_plot(dataset, args.output_dir / "modelnet40_sanity.png")

    if args.command == "prepare":
        print(f"Compact dataset ready: {args.cache}")
        return 0
    if args.command == "smoke":
        run_smoke_tests(dataset)
        return 0
    if args.command == "reevaluate":
        results = reevaluate_saved_runs(dataset, args.output_dir)
        print(f"Re-evaluated {len(results)} completed experiments")
        return 0
    if args.command == "run":
        for architecture in args.architectures:
            for selected in args.sizes:
                run_experiment(
                    dataset,
                    args.output_dir,
                    architecture,
                    selected,
                    maximum_epochs=args.maximum_epochs,
                    patience=args.patience,
                    time_limit_seconds=args.time_limit_seconds,
                    force=args.force,
                )
                summarize_results(args.output_dir)
    results = summarize_results(args.output_dir)
    print(f"Summarized {len(results)} completed experiments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
