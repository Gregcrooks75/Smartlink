from __future__ import annotations

from pathlib import Path
import pandas as pd
from mtx.training.build_ranker_dataset import (
    build_ranker_training_dataset,
)
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "meter_to_transformer"
    / "05_training_data"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
def build_training_datasets(
    tile_feature_files: dict[str, Path],
    labels_df: pd.DataFrame,
) -> dict[str, Path]:
    outputs = {}
    for (
        division,
        feature_file,
    ) in tile_feature_files.items():
        print(
            "\n"
            + "=" * 80
        )
        print(
            f"TRAINING DATA: {division}"
        )
        features = pd.read_parquet(
            feature_file
        )
        if (
            "Division"
            in labels_df.columns
        ):
            division_labels = (
                labels_df[
                    labels_df[
                        "Division"
                    ]
                    == division
                ]
                .copy()
            )
        else:
            division_labels = (
                labels_df.copy()
            )
        (
            training,
            missing_truth,
        ) = (
            build_ranker_training_dataset(
                features=features,
                labels=division_labels,
            )
        )
        training_file = (
            OUTPUT_DIR
            / (
                f"{division}"
                "_ranker_training_data.parquet"
            )
        )
        missing_file = (
            OUTPUT_DIR
            / (
                f"{division}"
                "_truth_not_in_candidate_set.parquet"
            )
        )
        training.to_parquet(
            training_file,
            index=False,
        )
        missing_truth.to_parquet(
            missing_file,
            index=False,
        )
        outputs[
            division
        ] = training_file
        print(
            f"Training rows: "
            f"{len(training):,}"
        )
        print(
            "Meters whose Truth was "
            "not found in candidate set: "
            f"{len(missing_truth):,}"
        )
    return outputs