"""Build Q4 result tables. Dependencies: Python standard library."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SELECTED_SIZES = (5, 10, 50)
VARIANTS = (
    ("Canonization (XYZ)", "q4_modelnet40", "canonization", False),
    (
        "Full symmetrization (XYZ)",
        "q4_modelnet40",
        "full_symmetrization",
        False,
    ),
    (
        "Sampled symmetrization (XYZ)",
        "q4_modelnet40",
        "sampled_symmetrization",
        False,
    ),
    ("Equivariant network (XYZ)", "q4_modelnet40", "equivariant", False),
    ("Data augmentation (XYZ)", "q4_modelnet40", "augmentation", False),
    (
        "Bonus: Canonization + Transformer (XYZ)",
        "q4_transformer_bonus",
        "canonization_transformer",
        True,
    ),
    (
        "Bonus: Full symmetrization + Transformer (XYZ)",
        "q4_transformer_bonus",
        "full_symmetrization_transformer",
        True,
    ),
    (
        "Bonus: Canonization + surface normals",
        "q4_modelnet40_normals",
        "canonization",
        True,
    ),
    (
        "Bonus: Full symmetrization + surface normals",
        "q4_modelnet40_normals",
        "full_symmetrization",
        True,
    ),
    (
        "Bonus: Sampled symmetrization + surface normals",
        "q4_modelnet40_normals",
        "sampled_symmetrization",
        True,
    ),
    (
        "Bonus: Equivariant network + surface normals",
        "q4_modelnet40_normals",
        "equivariant",
        True,
    ),
    (
        "Bonus: Data augmentation + surface normals",
        "q4_modelnet40_normals",
        "augmentation",
        True,
    ),
    (
        "Bonus: O(3)-invariant pairwise-distance features",
        "q4_modelnet40_rotation",
        "rotation_invariant",
        True,
    ),
)


def load_payload(
    experiment_root: Path,
    directory: str,
    architecture: str,
    selected: int,
) -> dict[str, Any]:
    path = (
        experiment_root
        / directory
        / "runs"
        / f"{architecture}-selected-{selected}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment record: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["architecture"] != architecture:
        raise ValueError(f"Architecture mismatch in {path}")
    if payload["selected_examples_per_class"] != selected:
        raise ValueError(f"Selected-size mismatch in {path}")
    return payload


def consolidate(experiment_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, directory, architecture, bonus in VARIANTS:
        for selected in SELECTED_SIZES:
            payload = load_payload(
                experiment_root, directory, architecture, selected
            )
            rows.append(
                {
                    "variant": label,
                    "bonus": bonus,
                    "architecture": architecture,
                    "selected_examples_per_class": selected,
                    "parameter_count": payload["parameter_count"],
                    "point_count": payload["point_count"],
                    "feature_dimension": payload["feature_dimension"],
                    "status": payload["training"]["status"],
                    "overall_accuracy": payload["test"]["overall_accuracy"],
                    "mean_class_accuracy": payload["test"][
                        "mean_class_accuracy"
                    ],
                    "training_seconds": payload["training"][
                        "training_seconds"
                    ],
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root", type=Path, default=Path("_qa")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("_qa/q4_joint_results")
    )
    args = parser.parse_args()

    rows = consolidate(args.experiment_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "q4_joint_results.csv")
    (args.output_dir / "q4_joint_results.json").write_text(
        json.dumps(
            {
                "selected_sizes": list(SELECTED_SIZES),
                "result_count": len(rows),
                "results": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Consolidated {len(rows)} experiment records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
