"""
SmartLink: train an XGBoost ranker on existing train/validation/test Parquets.

Requirements:
    numpy
    pandas
    pyarrow
    xgboost >= 2.0

The training files already contain y, qid, and the subgroup-quality exclusions.
"""

from __future__ import annotations

import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


ID_AND_LABEL_COLUMNS = {
    "meter_number",
    "candidate_transformer",
    "tx_group_id",
    "subgroup_id",
    "current_meter_floc",
    "current_transformer",
    "split_component",
    "qid",
    "y",
    "verified_transformer",
    "truth_transformer",
    "correct_transformer_floc",
    "split",
    "training_status",
    "label_source",
    "verification_date",
    "validationsource",
    "validationdate",
    "truth_count",
    "positive_count",
    "baseline_is_correct",
    "current_is_correct",
    "previously_recommended",
    "baseline_transformer",
    "ami_rows",
    "assignment_count",
    "label",
    "target",
}


REQUIRED = {
    "meter_number",
    "candidate_transformer",
    "current_transformer",
    "qid",
    "y",
    "division",
    "subgroup_id",
    "split_component",
    "distance_ft",
    "tx_current_transformer_count",
}


def validate_split(
    frame: pd.DataFrame,
    name: str,
    max_subgroup_transformers: int,
) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)

    if missing or frame.empty:
        raise ValueError(
            f"{name}: empty dataset or missing columns {sorted(missing)}"
        )

    keys = [
        "meter_number",
        "candidate_transformer",
        "qid",
        "split_component",
    ]

    if frame[keys].isna().any().any():
        raise ValueError(f"{name}: null identifiers")

    if frame.duplicated(
        ["meter_number", "candidate_transformer"]
    ).any():
        raise ValueError(f"{name}: duplicate meter/candidate pairs")

    if not frame["y"].isin([0, 1]).all():
        raise ValueError(f"{name}: y must contain only 0 and 1")

    groups = frame.groupby("qid", sort=False)

    if (
        (groups["meter_number"].nunique() != 1).any()
        or (groups["y"].sum() != 1).any()
        or (groups.size() < 2).any()
        or (frame.groupby("meter_number")["qid"].nunique() != 1).any()
    ):
        raise ValueError(
            f"{name}: each qid must identify one meter with "
            ">=2 candidates and one positive"
        )

    if groups["current_transformer"].nunique(dropna=False).gt(1).any():
        raise ValueError(
            f"{name}: inconsistent current assignments within a meter"
        )

    quality = pd.to_numeric(
        frame["tx_current_transformer_count"],
        errors="raise",
    )

    invalid_quality = (
        (frame["subgroup_id"].notna() & quality.isna())
        | quality.lt(0)
        | quality.gt(max_subgroup_transformers)
    )

    if invalid_quality.any():
        raise ValueError(
            f"{name}: invalid subgroup quality or transformer limit exceeded"
        )

    distances = frame["distance_ft"].to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError(
            f"{name}: distance_ft must be finite and nonnegative"
        )

    return (
        frame.sort_values(
            ["qid", "candidate_transformer"],
            kind="stable",
        )
        .reset_index(drop=True)
    )



def check_overlap(
    frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, int]]:

    identities = {}

    for name, frame in frames.items():
        subgroups = (
            frame.loc[
                frame["subgroup_id"].notna(),
                ["division", "subgroup_id"],
            ]
            .drop_duplicates()
        )

        identities[name] = {
            "meters": set(frame["meter_number"]),
            "components": set(frame["split_component"]),
            "verified_transformers": set(
                frame.loc[
                    frame["y"] == 1,
                    "candidate_transformer",
                ]
            ),
            "subgroups": set(
                subgroups.itertuples(index=False, name=None)
            ),
        }

    for split_a, split_b in combinations(identities, 2):
        for key in identities[split_a]:
            overlap = (
                identities[split_a][key]
                & identities[split_b][key]
            )

            if overlap:
                raise ValueError(
                    f"{split_a}/{split_b}: "
                    f"{len(overlap):,} overlapping {key}. "
                    "Rebuild the splits."
                )

    return {
        name: {
            key: len(values)
            for key, values in groups.items()
        }
        for name, groups in identities.items()
    }

def feature_schema(
    train: pd.DataFrame,
    exclude_features: tuple[str, ...],
) -> dict:
    excluded = (
        ID_AND_LABEL_COLUMNS
        | {column.lower() for column in exclude_features}
    )

    features = [
        column
        for column in train.columns
        if column.lower() not in excluded
        and not column.lower().endswith(
            ("_id", "_floc", "_number")
        )
        and not column.lower().startswith(
            (
                "truth_",
                "verified_",
                "label_",
                "validation_",
            )
        )
    ]

    if not features:
        raise ValueError(
            "No model features remain after filtering"
        )

    numeric = [
        column
        for column in features
        if is_numeric_dtype(train[column])
    ]

    categories = {
        column: sorted(
            train[column]
            .astype("string")
            .fillna("__MISSING__")
            .unique()
            .tolist()
        )
        for column in features
        if column not in numeric
    }

    encoded = numeric + [
        f"{column}__cat_{index}"
        for column, levels in categories.items()
        for index in range(len(levels))
    ]

    if len(encoded) != len(set(encoded)):
        raise ValueError("Encoded feature names collide")

    return {
        "source_features": features,
        "numeric": numeric,
        "categories": categories,
        "encoded_features": encoded,
        "excluded_columns": [
            column
            for column in train.columns
            if column not in features
        ],
    }


def _encode(
    frame: pd.DataFrame,
    schema: dict,
) -> pd.DataFrame:
    """
    Use training-only category levels.

    An unseen category becomes an all-zero block.
    """

    missing = (
        set(schema["source_features"])
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"Missing prediction features: {sorted(missing)}"
        )

    columns = {}

    for column in schema["numeric"]:
        values = pd.to_numeric(
            frame[column],
            errors="raise",
        ).to_numpy(
            dtype=np.float32,
            na_value=np.nan,
            copy=True,
        )

        values[~np.isfinite(values)] = np.nan
        columns[column] = values

    for column, levels in schema["categories"].items():
        values = (
            frame[column]
            .astype("string")
            .fillna("__MISSING__")
        )

        for index, level in enumerate(levels):
            encoded_name = f"{column}__cat_{index}"

            columns[encoded_name] = (
                values.eq(level)
                .to_numpy(dtype=np.float32)
            )

    return pd.DataFrame(
        columns,
        index=frame.index,
    )[schema["encoded_features"]]


def rank_candidates(
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank candidates using deterministic tie-breaking.

    Ties are resolved using:
        1. Shorter distance
        2. Transformer ID

    Rank scores are not probabilities.
    """

    columns = [
        column
        for column in [
            "meter_number",
            "candidate_transformer",
            "current_transformer",
            "division",
            "subgroup_id",
            "tx_current_transformer_count",
            "distance_ft",
            "y",
        ]
        if column in frame.columns
    ]

    ranked = frame[columns].copy()
    ranked["rank_score"] = np.asarray(scores)

    if not np.isfinite(ranked["rank_score"]).all():
        raise ValueError(
            "The model returned non-finite scores"
        )

    ranked = ranked.sort_values(
        [
            "meter_number",
            "rank_score",
            "distance_ft",
            "candidate_transformer",
        ],
        ascending=[True, False, True, True],
        kind="stable",
    )

    ranked["rank"] = (
        ranked.groupby(
            "meter_number",
            sort=False,
        )
        .cumcount()
        .add(1)
    )

    top = (
        ranked.loc[ranked["rank"] == 1]
        .copy()
        .set_index("meter_number")
    )

    top = top.rename(
        columns={
            "candidate_transformer": "recommended_transformer",
            "distance_ft": "recommended_distance_ft",
        }
    ).drop(
        columns=[
            column
            for column in ["y", "rank"]
            if column in top.columns
        ]
    )

    second = (
        ranked.loc[ranked["rank"] == 2]
        .set_index("meter_number")
    )

    top["second_transformer"] = second[
        "candidate_transformer"
    ]
    top["second_score"] = second["rank_score"]
    top["score_gap"] = (
        top["rank_score"] - top["second_score"]
    )
    top["top_score_tied"] = top["score_gap"].eq(0)

    top["candidate_count"] = (
        ranked.groupby("meter_number").size()
    )

    nearest = (
        ranked.sort_values(
            [
                "meter_number",
                "distance_ft",
                "candidate_transformer",
            ],
            kind="stable",
        )
        .drop_duplicates("meter_number")
        .set_index("meter_number")
    )

    top["nearest_transformer"] = nearest[
        "candidate_transformer"
    ]

    return ranked, top


def _evaluate(
    frame: pd.DataFrame,
    scores: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    ranked, top = rank_candidates(frame, scores)

    truth = (
        ranked.loc[ranked["y"] == 1]
        .set_index("meter_number")
    )

    top["verified_transformer"] = truth[
        "candidate_transformer"
    ]
    top["truth_rank"] = truth["rank"]

    top["correct"] = top[
        "recommended_transformer"
    ].eq(top["verified_transformer"])

    top["nearest_correct"] = top[
        "nearest_transformer"
    ].eq(top["verified_transformer"])

    known = top["current_transformer"].notna()

    current_correct = (
        top["current_transformer"]
        .eq(top["verified_transformer"])
        & known
    )

    incorrect = known & ~current_correct

    top["current_correct"] = current_correct

    top["proposed_change"] = (
        known
        & top["recommended_transformer"].ne(
            top["current_transformer"]
        )
    )

    top["new_assignment"] = ~known

    top["false_change"] = (
        current_correct
        & top["proposed_change"]
    )

    def mean(values: pd.Series) -> float | None:
        return (
            float(values.mean())
            if len(values)
            else None
        )

    metrics = {
        "meters": len(top),
        "candidate_rows": len(frame),
        "top1_accuracy": mean(top["correct"]),
        "top3_recall": mean(top["truth_rank"].le(3)),
        "mrr": mean(1.0 / top["truth_rank"]),
        "ndcg_at_3": float(
            np.where(
                top["truth_rank"] <= 3,
                1.0 / np.log2(top["truth_rank"] + 1),
                0.0,
            ).mean()
        ),
        "nearest_baseline_accuracy": mean(
            top["nearest_correct"]
        ),
        "current_assignment_coverage": mean(known),
        "current_baseline_accuracy_known": mean(
            current_correct[known]
        ),
        "currently_incorrect_meters": int(
            incorrect.sum()
        ),
        "top1_on_currently_incorrect": mean(
            top.loc[incorrect, "correct"]
        ),
        "unassigned_meters": int((~known).sum()),
        "top1_on_unassigned": mean(
            top.loc[~known, "correct"]
        ),
        "proposed_changes": int(
            top["proposed_change"].sum()
        ),
        "proposed_change_precision": mean(
            top.loc[
                top["proposed_change"],
                "correct",
            ]
        ),
        "correct_assignments_wrongly_changed": int(
            top["false_change"].sum()
        ),
        "wrong_change_rate_on_correct_assignments": mean(
            top.loc[
                current_correct,
                "false_change",
            ]
        ),
        "top_score_ties": int(
            top["top_score_tied"].sum()
        ),
    }

    return metrics, top.reset_index()


def predict_transformers(
    frame: pd.DataFrame,
    model_directory: str | Path,
) -> pd.DataFrame:
    """
    Rank candidates containing the same source features
    used during training.
    """

    import xgboost as xgb

    folder = Path(model_directory)

    metadata = json.loads(
        (folder / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    if frame["tx_current_transformer_count"].gt(
        metadata["max_subgroup_transformers"]
    ).any():
        raise ValueError(
            "Prediction data contains subgroups above "
            "the training-quality limit"
        )

    model = xgb.XGBRanker()
    model.load_model(folder / "model.ubj")

    scores = model.predict(
        _encode(
            frame,
            metadata["feature_schema"],
        ),
        iteration_range=(
            0,
            metadata["best_iteration"] + 1,
        ),
    )

    return (
        rank_candidates(frame, scores)[1]
        .reset_index()
    )


def run_model_training_stage(
    *,
    project_root: str | Path,
    training_directory: str | Path | None = None,
    model_params: dict | None = None,
    exclude_features: tuple[str, ...] = (),
    evaluate_test: bool = False,
    max_subgroup_transformers: int = 3,
    seed: int = 42,
) -> Path:
    """
    Fit only on train and use validation only for early stopping.

    Test reporting is opt-in.

    Use the following for a second ranking objective:

        model_params={"objective": "rank:pairwise"}

    This stage ranks individual meters. Connection-object
    enforcement comes later.
    """

    import xgboost as xgb

    root = Path(project_root)

    dataset = (
        Path(training_directory)
        if training_directory
        else root / "data/processed/training/current"
    )

    frames = {
        name: validate_split(
            pd.read_parquet(
                dataset / f"{name}.parquet"
            ),
            name,
            max_subgroup_transformers,
        )
        for name in (
            "train",
            "validation",
            "test",
        )
    }

    split_counts = check_overlap(frames)

    train = frames["train"]
    validation = frames["validation"]

    schema = feature_schema(
        train,
        exclude_features,
    )

    x_train = _encode(train, schema)
    x_validation = _encode(validation, schema)

    params = {
        "objective": "rank:ndcg",
        "tree_method": "hist",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "lambdarank_pair_method": "topk",
        "lambdarank_num_pair_per_sample": 8,
        "eval_metric": [
            "ndcg@3",
            "ndcg@1",
        ],
        "early_stopping_rounds": 75,
        "random_state": seed,
        "n_jobs": 4,
    }

    params.update(model_params or {})

    valid_objectives = {
        "rank:ndcg",
        "rank:pairwise",
        "rank:map",
    }

    if params["objective"] not in valid_objectives:
        raise ValueError(
            "Use a ranking objective: "
            "rank:ndcg, rank:pairwise, or rank:map"
        )

    patience = params.pop(
        "early_stopping_rounds"
    )

    if (
        not isinstance(patience, int)
        or patience < 1
    ):
        raise ValueError(
            "early_stopping_rounds must be "
            "a positive integer"
        )

    # Keep only the best trees in the saved model
    # and its feature importance.
    model = xgb.XGBRanker(
        **params,
        callbacks=[
            xgb.callback.EarlyStopping(
                rounds=patience,
                save_best=True,
            )
        ],
    )

    model.fit(
        x_train,
        train["y"].to_numpy(),
        qid=pd.factorize(
            train["qid"],
            sort=False,
        )[0],
        eval_set=[
            (
                x_validation,
                validation["y"].to_numpy(),
            )
        ],
        eval_qid=[
            pd.factorize(
                validation["qid"],
                sort=False,
            )[0]
        ],
        verbose=100,
    )

    best_iteration = int(
        model.best_iteration
    )

    rounds = (
        0,
        best_iteration + 1,
    )

    run_id = datetime.now().strftime(
        "%Y-%m-%d_%H%M%S_%f"
    )

    output = (
        root
        / "data/processed/models/transformer_ranker"
        / run_id
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        output / "model.ubj"
    )

    metrics = {}

    evaluation_splits = (
        ("train", "validation", "test")
        if evaluate_test
        else ("train", "validation")
    )

    for name in evaluation_splits:
        frame = frames[name]

        if name == "train":
            matrix = x_train
        elif name == "validation":
            matrix = x_validation
        else:
            matrix = _encode(
                frame,
                schema,
            )

        scores = model.predict(
            matrix,
            iteration_range=rounds,
        )

        metrics[name], predictions = _evaluate(
            frame,
            scores,
        )

        predictions.to_parquet(
            output / f"{name}_predictions.parquet",
            index=False,
        )

        print(
            f"{name}: "
            f"top1={metrics[name]['top1_accuracy']:.4f}; "
            f"nearest="
            f"{metrics[name]['nearest_baseline_accuracy']:.4f}; "
            f"change precision="
            f"{metrics[name]['proposed_change_precision']}"
        )

    gain = (
        model.get_booster()
        .get_score(importance_type="gain")
    )

    importance = pd.DataFrame(
        {
            "feature": schema["encoded_features"],
            "gain": [
                gain.get(feature, 0.0)
                for feature in schema[
                    "encoded_features"
                ]
            ],
        }
    )

    importance.sort_values(
        "gain",
        ascending=False,
    ).to_csv(
        output / "feature_importance.csv",
        index=False,
    )

    stopping_metric = params["eval_metric"]

    if isinstance(stopping_metric, list):
        stopping_metric = stopping_metric[-1]

    metadata = {
        "training_directory": str(
            dataset.resolve()
        ),
        "xgboost_version": xgb.__version__,
        "pandas_version": pd.__version__,
        "best_iteration": best_iteration,
        "best_validation_score": float(
            model.best_score
        ),
        "early_stopping_metric": stopping_metric,
        "early_stopping_rounds": patience,
        "parameters": params,
        "feature_schema": schema,
        "split_counts": split_counts,
        "max_subgroup_transformers": (
            max_subgroup_transformers
        ),
        "test_evaluated": evaluate_test,
    }

    output_files = [
        ("metadata.json", metadata),
        ("metrics.json", metrics),
        (
            "learning_curve.json",
            model.evals_result(),
        ),
    ]

    for filename, value in output_files:
        (output / filename).write_text(
            json.dumps(
                value,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    print(
        f"Saved "
        f"{len(schema['encoded_features'])} "
        f"encoded features, "
        f"{best_iteration + 1} selected trees "
        f"-> {output}"
    )

    return output