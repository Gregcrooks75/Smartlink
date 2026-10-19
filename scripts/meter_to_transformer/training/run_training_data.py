from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date, datetime
from pathlib import Path

import polars as pl


def _id(column: str) -> pl.Expr:
    value = (
        pl.col(column)
        .cast(pl.String)
        .str.strip_chars()
        .str.to_uppercase()
    )

    return (
        pl.when(value.is_in(["", "NULL", "NONE", "NAN", "N/A", "..."]))
        .then(None)
        .otherwise(value)
    )


def meter_splits(
    meters: pl.DataFrame,
    seed: int,
) -> pl.DataFrame:
    parent = {}
    owners = {}

    def find(meter):
        parent.setdefault(meter, meter)

        while parent[meter] != meter:
            parent[meter] = parent[parent[meter]]
            meter = parent[meter]

        return meter

    for meter, truth, division, subgroup in meters.select(
        "meter_number",
        "verified_transformer",
        "division",
        "subgroup_id",
    ).iter_rows():
        keys = [("transformer", truth)]

        if subgroup is not None:
            keys.append(("subgroup", division, subgroup))

        for key in keys:
            a = find(meter)
            b = find(owners.setdefault(key, meter))
            parent[max(a, b)] = min(a, b)

    rows = []

    for meter in sorted(meters["meter_number"].to_list()):
        component = find(meter)

        digest = hashlib.sha256(
            f"{seed}|{component}".encode()
        ).digest()

        fraction = int.from_bytes(digest[:8], "big") / 2**64

        if fraction < 0.70:
            split = "train"
        elif fraction < 0.85:
            split = "validation"
        else:
            split = "test"

        rows.append((meter, component, split))

    return (
        pl.DataFrame(
            rows,
            schema=[
                "meter_number",
                "split_component",
                "split",
            ],
            orient="row",
        )
        .with_row_index("qid")
    )


def run_training_data_stage(
    *,
    project_root: str | Path,
    adam_labels: pl.DataFrame,
    current_ami: pl.DataFrame,
    run_date: str | None = None,
    truth_col: str = "Correct_Transformer_FLOC",
    ami_meter_col: str = "meter_number",
    ami_transformer_col: str = "SuperiorFLOC",
    subgroup_col: str = "tx_group_id",
    quality_col: str = "tx_current_transformer_count",
    approved_label_sources: tuple[str, ...] = ("MAP",),
    max_subgroup_transformers: int = 3,
    exclude_previous_recommendations: bool = True,
    exclude_features: tuple[str, ...] = (),
    seed: int = 42,
) -> Path:
    """
    Build a current training snapshot.

    current_ami must match the feature data for run_date.

    ADAM requires:
        - meter_number
        - truth_col
        - ValidationSource
        - ValidationDate

    All numeric and Boolean source features are selected except IDs,
    label fields, and audit fields.

    Other source columns remain available in the output Parquet files
    as metadata.
    """

    root = Path(project_root)

    day = date.fromisoformat(run_date) if run_date else date.today()
    run_date = day.isoformat()

    feature_dir = (
        root
        / "data"
        / "cache"
        / "final_features"
        / run_date
    )

    corrections = (
        root
        / "data"
        / "cache"
        / "historical_corrections"
        / run_date
        / "all_historical_corrections.parquet"
    )

    output = (
        root
        / "data"
        / "processed"
        / "training"
    )

    # ---------------------------------------------------------
    # Prepare ADAM labels
    # ---------------------------------------------------------

    labels_raw = (
        adam_labels.select(
            _id("meter_number").alias("meter_number"),
            _id(truth_col).alias("verified_transformer"),
            _id("ValidationSource").alias("label_source"),
            (
                pl.col("ValidationDate")
                .cast(pl.String)
                .str.slice(0, 10)
                .str.to_date("%Y-%m-%d", strict=False)
                .alias("verification_date")
            ),
        )
        .filter(pl.col("meter_number").is_not_null())
    )

    reviewed = (
        labels_raw.select("meter_number")
        .unique()
    )

    truth = (
        labels_raw.filter(
            pl.col("verified_transformer").is_not_null()
            & pl.col("label_source").is_in(
                [
                    source.strip().upper()
                    for source in approved_label_sources
                ]
            )
            & (pl.col("verification_date") <= day)
        )
        .sort(
            "verification_date",
            descending=True,
        )
        .group_by("meter_number")
        .agg(
            pl.col("verified_transformer")
            .n_unique()
            .alias("truth_count"),
            pl.col("verified_transformer").first(),
            pl.col("label_source").first(),
            pl.col("verification_date").first(),
        )
    )

    # ---------------------------------------------------------
    # Prepare current AMI assignments
    # ---------------------------------------------------------

    ami = (
        current_ami.select(
            _id(ami_meter_col).alias("meter_number"),
            _id(ami_transformer_col).alias(
                "current_transformer"
            ),
        )
        .join(
            reviewed,
            on="meter_number",
            how="semi",
        )
    )

    assignments = (
        ami.group_by("meter_number")
        .agg(
            pl.len().alias("ami_rows"),
            (
                pl.col("current_transformer")
                .drop_nulls()
                .n_unique()
                .alias("assignment_count")
            ),
            (
                pl.col("current_transformer")
                .drop_nulls()
                .first()
            ),
        )
    )

    # ---------------------------------------------------------
    # Load historical recommendation exclusions
    # ---------------------------------------------------------

    if exclude_previous_recommendations and not corrections.exists():
        raise FileNotFoundError(
            f"Missing recommendation exclusions: {corrections}"
        )

    if corrections.exists():
        sent = pl.read_parquet(
            corrections,
            columns=["meter_number"],
        )
    else:
        sent = pl.DataFrame(
            schema={"meter_number": pl.String}
        )

    sent = (
        sent.select(
            _id("meter_number").alias("meter_number")
        )
        .unique()
    )

    # ---------------------------------------------------------
    # Build the meter-level label table
    # ---------------------------------------------------------

    labels = (
        reviewed.join(
            truth,
            on="meter_number",
            how="left",
        )
        .join(
            assignments,
            on="meter_number",
            how="left",
        )
        .join(
            sent.with_columns(
                pl.lit(True).alias(
                    "previously_recommended"
                )
            ),
            on="meter_number",
            how="left",
        )
        .with_columns(
            pl.col("previously_recommended")
            .fill_null(False)
        )
    )

    eligible = labels.filter(
        (pl.col("truth_count") == 1)
        & pl.col("ami_rows").is_not_null()
        & (pl.col("assignment_count") <= 1)
        & ~(
            pl.col("previously_recommended")
            & pl.lit(exclude_previous_recommendations)
        )
    )

    # ---------------------------------------------------------
    # Find division feature files
    # ---------------------------------------------------------

    files = sorted(
        feature_dir.glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No division feature files: {feature_dir}"
        )

    scans = [
        (path, pl.scan_parquet(path))
        for path in files
    ]

    schemas = [
        scan.collect_schema()
        for _, scan in scans
    ]

    generated = [
        "candidate_matches_current",
        "candidate_same_current_parent",
    ]

    # Drop supervised and audit fields if an upstream file
    # accidentally contains them.
    drop = {
        "y",
        "label",
        "target",
        "truth",
        "truth_transformer",
        "verified_transformer",
        truth_col.lower(),
        "is_correct",
        "baseline_is_correct",
        "current_is_correct",
        "label_source",
        "verification_date",
        "validationsource",
        "validationdate",
        "truth_count",
        "positive_count",
        "training_status",
        "previously_recommended",
        "qid",
        "split",
        "split_component",
        "baseline_transformer",
        "current_transformer",
        "ami_rows",
        "assignment_count",
        "candidate_matches_baseline",
        "candidate_same_baseline_parent",
        *generated,
    }

    metadata = {
        "meter_number",
        "candidate_transformer",
        "division",
        "meterid",
        "superiorfloc",
        "funcloc",
        "subgroup_id",
        subgroup_col.lower(),
    }

    excluded = (
        drop
        | metadata
        | {
            column.lower()
            for column in exclude_features
        }
    )

    # ---------------------------------------------------------
    # Determine model feature columns
    # ---------------------------------------------------------

    features = sorted(
        {
            name
            for schema in schemas
            for name, dtype in schema.items()
            if (
                dtype.is_numeric()
                or dtype == pl.Boolean
            )
            and name.lower() not in excluded
            and not name.lower().endswith(
                (
                    "_id",
                    "_floc",
                    "_number",
                )
            )
            and not name.lower().startswith(
                (
                    "truth_",
                    "verified_",
                    "label_",
                    "validation_",
                )
            )
        }
    )

    # ---------------------------------------------------------
    # Load features from every division
    # ---------------------------------------------------------

    parts = []

    for (path, scan), schema in zip(
        scans,
        schemas,
    ):
        required = {
            "meter_number",
            "candidate_transformer",
            subgroup_col,
            "current_meter_floc",
            *features,
        }

        missing = required - set(
            schema.names()
        )

        if missing:
            raise ValueError(
                f"{path.name}: missing columns "
                f"{sorted(missing)}"
            )

        inconsistent_features = any(
            not (
                schema[column].is_numeric()
                or schema[column] in (
                    pl.Boolean,
                    pl.Null,
                )
            )
            for column in features
        )

        if inconsistent_features:
            raise TypeError(
                f"{path.name}: inconsistent model-feature "
                "types across divisions"
            )

        part = (
            scan.select(
                [
                    column
                    for column in schema.names()
                    if column.lower() not in drop
                ]
            )
            .with_columns(
                _id("meter_number").alias(
                    "meter_number"
                ),
                _id("candidate_transformer").alias(
                    "candidate_transformer"
                ),
                _id(subgroup_col).alias(
                    "subgroup_id"
                ),
                _id("current_meter_floc").alias(
                    "current_meter_floc"
                ),
                pl.lit(path.stem).alias(
                    "division"
                ),
            )
            .with_columns(
                pl.when(
                    pl.col("subgroup_id").is_not_null()
                )
                .then(
                    pl.col("current_meter_floc")
                    .drop_nulls()
                    .n_unique()
                    .over("subgroup_id")
                )
                .otherwise(pl.lit(1))
                .cast(pl.Int32)
                .alias(quality_col)
            )
            .join(
                eligible.select(
                    "meter_number"
                ).lazy(),
                on="meter_number",
                how="semi",
            )
            .collect(engine="streaming")
        )

        parts.append(part)

    # ---------------------------------------------------------
    # Combine candidate pairs
    # ---------------------------------------------------------

    pairs = (
        pl.concat(
            parts,
            how="diagonal_relaxed",
        )
        .join(
            eligible.select(
                "meter_number",
                "verified_transformer",
                "current_transformer",
            ),
            on="meter_number",
            how="left",
            validate="m:1",
        )
    )

    duplicate_pairs = pairs.select(
        pl.struct(
            "meter_number",
            "candidate_transformer",
        )
        .is_duplicated()
        .any()
    ).item()

    if (
        pairs["candidate_transformer"].null_count()
        or duplicate_pairs
    ):
        raise ValueError(
            "Null candidates or duplicate "
            "meter/candidate pairs"
        )

    # ---------------------------------------------------------
    # Generate labels and current-assignment features
    # ---------------------------------------------------------

    pairs = pairs.with_columns(
        (
            pl.col("candidate_transformer")
            == pl.col("verified_transformer")
        )
        .cast(pl.Int8)
        .alias("y"),
        (
            pl.col("candidate_transformer")
            == pl.col("current_transformer")
        )
        .fill_null(False)
        .cast(pl.Float32)
        .alias(generated[0]),
        (
            pl.col("candidate_transformer")
            .str.replace(r"-[^-]+$", "")
            == pl.col("current_transformer")
            .str.replace(r"-[^-]+$", "")
        )
        .fill_null(False)
        .cast(pl.Float32)
        .alias(generated[1]),
    )

    # ---------------------------------------------------------
    # Build meter-level metadata
    # ---------------------------------------------------------

    metadata_columns = [
        "division",
        "subgroup_id",
        quality_col,
    ]

    meters = (
        pairs.group_by("meter_number")
        .agg(
            pl.len().alias("candidate_count"),
            pl.col("y")
            .sum()
            .alias("positive_count"),
            pl.col("division").first(),
            pl.col("subgroup_id").first(),
            pl.col(quality_col)
            .first()
            .alias("subgroup_transformer_count"),
            *[
                pl.col(column)
                .n_unique()
                .alias(f"_n_{index}")
                for index, column in enumerate(
                    metadata_columns
                )
            ],
        )
    )

    inconsistent_metadata = meters.filter(
        pl.any_horizontal(
            pl.col("^_n_.*$") > 1
        )
    )

    if inconsistent_metadata.height:
        raise ValueError(
            "Inconsistent division/subgroup metadata "
            "within a meter"
        )

    inconsistent_subgroups = (
        meters.filter(
            pl.col("subgroup_id").is_not_null()
        )
        .group_by(
            "division",
            "subgroup_id",
        )
        .agg(
            pl.col("subgroup_transformer_count")
            .n_unique()
            .alias("n")
        )
        .filter(pl.col("n") > 1)
    )

    if inconsistent_subgroups.height:
        raise ValueError(
            "Subgroup transformer counts disagree "
            "between group members"
        )

    # ---------------------------------------------------------
    # Build meter audit
    # ---------------------------------------------------------

    audit = labels.join(
        meters.drop(
            "_n_0",
            "_n_1",
            "_n_2",
        ),
        on="meter_number",
        how="left",
    )

    invalid_quality = (
        (
            pl.col("subgroup_id").is_not_null()
            & pl.col(
                "subgroup_transformer_count"
            ).is_null()
        )
        | (
            pl.col("subgroup_transformer_count") <= 0
        )
    )

    audit = audit.with_columns(
        (
            pl.col("current_transformer")
            == pl.col("verified_transformer")
        ).alias("current_is_correct"),
        (
            pl.when(
                pl.col("truth_count").fill_null(0) == 0
            )
            .then(pl.lit("NO_APPROVED_TRUTH"))
            .when(
                pl.col("truth_count") > 1
            )
            .then(pl.lit("CONFLICTING_TRUTH"))
            .when(
                pl.col("ami_rows").is_null()
            )
            .then(pl.lit("NOT_IN_CURRENT_AMI"))
            .when(
                pl.col("assignment_count") > 1
            )
            .then(
                pl.lit("CONFLICTING_CURRENT_AMI")
            )
            .when(
                pl.col("previously_recommended")
                & pl.lit(
                    exclude_previous_recommendations
                )
            )
            .then(
                pl.lit("PREVIOUSLY_RECOMMENDED")
            )
            .when(
                pl.col("candidate_count").is_null()
            )
            .then(pl.lit("NO_FINAL_FEATURES"))
            .when(invalid_quality)
            .then(
                pl.lit("INVALID_SUBGROUP_QUALITY")
            )
            .when(
                pl.col(
                    "subgroup_transformer_count"
                )
                > max_subgroup_transformers
            )
            .then(
                pl.lit("SUBGROUP_OVER_LIMIT")
            )
            .when(
                pl.col("positive_count") == 0
            )
            .then(
                pl.lit("TRUTH_NOT_IN_CANDIDATES")
            )
            .when(
                pl.col("positive_count") != 1
            )
            .then(
                pl.lit("INVALID_POSITIVE_COUNT")
            )
            .when(
                pl.col("candidate_count") < 2
            )
            .then(pl.lit("SINGLE_CANDIDATE"))
            .otherwise(pl.lit("INCLUDED"))
            .alias("training_status")
        ),
    )

    included = audit.filter(
        pl.col("training_status") == "INCLUDED"
    )

    counts = dict(
        audit.group_by("training_status")
        .len()
        .iter_rows()
    )

    if included.is_empty():
        raise ValueError(
            "No usable training meters. "
            f"Status counts: {counts}"
        )

    # ---------------------------------------------------------
    # Create train, validation, and test splits
    # ---------------------------------------------------------

    splits = meter_splits(
        included,
        seed,
    )

    if splits["split"].n_unique() != 3:
        raise ValueError(
            "Fewer than three nonempty component splits; "
            "inspect subgroup connectivity"
        )

    audit = (
        audit.join(
            splits,
            on="meter_number",
            how="left",
        )
        .sort("meter_number")
    )

    # ---------------------------------------------------------
    # Build model dataset
    # ---------------------------------------------------------

    model_features = (
        features
        + generated
    )

    model = (
        pairs.join(
            splits,
            on="meter_number",
            how="inner",
            validate="m:1",
        )
        .drop("verified_transformer")
        .with_columns(
            [
                pl.when(
                    pl.col(column)
                    .cast(pl.Float32)
                    .is_finite()
                )
                .then(
                    pl.col(column).cast(
                        pl.Float32
                    )
                )
                .otherwise(None)
                .alias(column)
                for column in model_features
            ]
        )
    )

    # ---------------------------------------------------------
    # Build run summary
    # ---------------------------------------------------------

    summary = {
        "run_date": run_date,
        "assignment_asof": run_date,
        "feature_asof": run_date,
        "seed": seed,
        "polars_version": pl.__version__,
        "reviewed_meters": reviewed.height,
        "included_meters": included.height,
        "candidate_rows": model.height,
        "included_with_incorrect_current_floc": (
            included.filter(
                ~pl.col("current_is_correct")
            ).height
        ),
        "included_without_current_floc": (
            included[
                "current_transformer"
            ].null_count()
        ),
        "meters_by_status": counts,
        "model_features": model_features,
        "feature_count": len(model_features),
        "feature_null_counts": (
            model.select(model_features)
            .null_count()
            .row(0, named=True)
        ),
        "approved_label_sources": list(
            approved_label_sources
        ),
        "exclude_previous_recommendations": (
            exclude_previous_recommendations
        ),
        "max_subgroup_transformers": (
            max_subgroup_transformers
        ),
        "source_feature_files": [
            str(path)
            for path in files
        ],
        "splits": (
            model.group_by("split")
            .agg(
                pl.col("meter_number")
                .n_unique()
                .alias("meters"),
                pl.len().alias("rows"),
            )
            .sort("split")
            .to_dicts()
        ),
    }

    # ---------------------------------------------------------
    # Create history output folders
    # ---------------------------------------------------------

    history = (
        output
        / "history"
        / run_date
        / datetime.now().strftime(
            "%H%M%S_%f"
        )
    )

    sources = history / "sources"

    sources.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Save source snapshots
    # ---------------------------------------------------------

    labels_raw.write_parquet(
        sources / "adam_labels.parquet",
        compression="zstd",
    )

    ami.write_parquet(
        sources / "current_ami.parquet",
        compression="zstd",
    )

    if corrections.exists():
        shutil.copy2(
            corrections,
            sources
            / "historical_corrections.parquet",
        )

    # ---------------------------------------------------------
    # Save audit and model outputs
    # ---------------------------------------------------------

    audit.write_parquet(
        history / "meter_audit.parquet",
        compression="zstd",
    )

    for split in (
        "train",
        "validation",
        "test",
    ):
        (
            model.filter(
                pl.col("split") == split
            )
            .drop("split")
            .sort(
                "qid",
                "candidate_transformer",
            )
            .write_parquet(
                history / f"{split}.parquet",
                compression="zstd",
            )
        )

    (
        history / "feature_columns.json"
    ).write_text(
        json.dumps(
            model_features,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        history / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Copy latest outputs into the current folder
    # ---------------------------------------------------------

    current = output / "current"

    current.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in history.iterdir():
        if path.is_file():
            shutil.copy2(
                path,
                current / path.name,
            )

    print(
        f"Training: {included.height:,}/"
        f"{reviewed.height:,} reviewed meters, "
        f"{model.height:,} pairs, "
        f"{len(model_features)} features -> "
        f"{current}"
    )

    print(counts)

    return current