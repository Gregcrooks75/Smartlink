from pathlib import Path
import json
import pandas as pd


def review_corrections(
    model_directory,
    training_directory,
    target_precision=0.95,
    min_changes=100,
    evaluate_test=False,
):
    folder = Path(model_directory)
    settings_file = folder / "correction_filter.json"
    split = "test" if evaluate_test else "validation"

    if evaluate_test:
        settings = json.loads(settings_file.read_text())
        threshold = settings["min_score_gap"]

        if threshold is None:
            raise ValueError("No validation threshold was approved.")

        from meter_to_transformer.training.run_model_training import (
            predict_transformers,
        )

        test = pd.read_parquet(Path(training_directory) / "test.parquet")
        predictions = predict_transformers(test, folder)

        truth = test.loc[
            test["y"].eq(1),
            ["meter_number", "candidate_transformer"],
        ].rename(
            columns={
                "candidate_transformer": "verified_transformer"
            }
        )

        predictions = predictions.merge(
            truth,
            on="meter_number",
            validate="one_to_one",
        )

    else:
        predictions = pd.read_parquet(
            folder / "validation_predictions.parquet"
        )

    known = predictions["current_transformer"].notna()

    was_correct = predictions["current_transformer"].eq(
        predictions["verified_transformer"]
    )

    incorrect_meters = int((known & ~was_correct).sum())

    changes = predictions.loc[
        known
        & predictions["recommended_transformer"].ne(
            predictions["current_transformer"]
        )
        & predictions["score_gap"].gt(0)
    ].copy()

    changes["correct"] = changes["recommended_transformer"].eq(
        changes["verified_transformer"]
    )

    changes["wrongly_changed"] = changes["current_transformer"].eq(
        changes["verified_transformer"]
    )

    if not evaluate_test:
        table = (
            changes.groupby("score_gap")
            .agg(
                proposed_changes=("correct", "size"),
                correct_changes=("correct", "sum"),
                correctly_assigned_meters_wrongly_changed=(
                    "wrongly_changed",
                    "sum",
                ),
            )
            .sort_index(ascending=False)
            .cumsum()
            .reset_index()
        )

        table["precision"] = (
            table["correct_changes"] / table["proposed_changes"]
        )

        table.to_csv(
            folder / "validation_thresholds.csv",
            index=False,
        )

        eligible = table.loc[
            table["precision"].ge(target_precision)
            & table["proposed_changes"].ge(min_changes)
        ]

        threshold = (
            None
            if eligible.empty
            else float(eligible.iloc[-1]["score_gap"])
        )

        settings_file.write_text(
            json.dumps(
                {
                    "min_score_gap": threshold,
                    "target_precision": target_precision,
                    "min_changes": min_changes,
                },
                indent=2,
            )
        )

    if threshold is None:
        print("No cutoff meets the validation target and minimum volume.")
        return None

    selected = changes.loc[
        changes["score_gap"].ge(threshold)
    ]

    correct = int(selected["correct"].sum())

    summary = {
        "split": split,
        "min_score_gap": threshold,
        "proposed_changes": len(selected),
        "correct_changes": correct,
        "incorrect_changes": len(selected) - correct,
        "precision": (
            correct / len(selected)
            if len(selected)
            else None
        ),
        "incorrect_meters_recovered": (
            correct / incorrect_meters
            if incorrect_meters
            else None
        ),
        "correctly_assigned_meters_wrongly_changed": int(
            selected["wrongly_changed"].sum()
        ),
    }

    (folder / f"{split}_correction_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )

    selected.to_parquet(
        folder / f"{split}_selected_corrections.parquet",
        index=False,
    )

    print(json.dumps(summary, indent=2))

    return summary