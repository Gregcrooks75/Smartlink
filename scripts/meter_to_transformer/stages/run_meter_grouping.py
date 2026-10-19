from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from mtx.grouping.group_assignments import (
    build_itron_group_assignments,
    build_smartlink_group_assignments,
    normalizemeter_population,
)
from mtx.run_config import RunConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]

GROUPING_CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "grouping"
)


def _atomic_write_parquet(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write, read-check, and then replace a current-state Parquet file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        f".{path.stem}.{uuid4().hex}.tmp.parquet"
    )

    try:
        frame.to_parquet(
            temporary,
            index=False,
        )

        check = pd.read_parquet(
            temporary,
            columns=["meter_number"],
        )

        if len(check) != len(frame):
            raise OSError(
                f"Temporary grouping file row count changed: {temporary}"
            )

        temporary.replace(path)

    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_partition(
    population: pd.DataFrame,
    groups: pd.DataFrame,
    division: str,
) -> None:
    """Confirm that grouping is a one-to-one view of the AMI population."""

    expected = set(population["meter_number"])
    actual = set(groups["meter_number"])

    if (
        groups["meter_number"].isna().any()
        or not groups["meter_number"].is_unique
    ):
        raise ValueError(
            f"[{division}] grouping meter_number must be non-null and unique."
        )

    if expected != actual or len(groups) != len(population):
        raise ValueError(
            f"[{division}] grouping population does not match AMI: "
            f"missing={len(expected - actual):,}, "
            f"unexpected={len(actual - expected):,}."
        )

    assigned = int(groups["tx_group_id"].notna().sum())
    unassigned = int(groups["tx_group_id"].isna().sum())

    if assigned + unassigned != len(groups):
        raise ValueError(
            f"[{division}] assigned and unassigned counts do not reconcile."
        )

from datetime import date


def run_meter_grouping_stage(
    ami_df: pd.DataFrame,
    itron_groups_df: pd.DataFrame | None = None,
    config: RunConfig | None = None,
    smartlink_cluster_root: Path | None = None,
) -> dict[str, Path]:
    """
    Build one grouping file per division each day.

    Reuse today's files when they pass validation.
    Rebuild invalid files or when overwrite is requested.

    Uses the existing grouping builders, validation helper,
    and atomic Parquet writer.
    """

    ami = ami_df.rename(
        columns={
            "MeterID": "meter_number",
            "Division": "division",
        }
    ).copy()

    required = {
        "meter_number",
        "division",
    }

    missing = required - set(ami.columns)

    if missing:
        raise ValueError(
            "AMI dataframe missing grouping columns: "
            f"{sorted(missing)}"
        )

    valid_population = (
        ami[["meter_number", "division"]]
        .dropna()
        .copy()
    )

    for column in ("meter_number", "division"):
        valid_population[column] = (
            valid_population[column]
            .astype("string")
            .str.strip()
        )

        if valid_population[column].eq("").any():
            raise ValueError(
                f"Filtered AMI contains blank {column} values."
            )

    if valid_population["meter_number"].duplicated().any():
        raise ValueError(
            "Filtered AMI must contain one row per meter_number."
        )

    population = normalizemeter_population(
        valid_population
    )

    divisions = sorted(
        population["division"].unique()
    )

    group_source = (
        config.group_source
        if config is not None
        else "itron"
    )

    if group_source not in {"itron", "smartlink"}:
        raise ValueError(
            "group_source must be 'itron' or 'smartlink'."
        )

    today = date.today().isoformat()

    output_dir = (
        GROUPING_CACHE_DIRECTORY
        / group_source
        / today
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    force_rebuild = (
        config is not None
        and config.overwrite_m2t_outputs
    )

    outputs: dict[str, Path] = {}
    saved_population = 0

    print(f"Grouping source: {group_source}")
    print(f"Grouping cache date: {today}")

    for division in divisions:
        division_population = population.loc[
            population["division"] == division,
            ["meter_number", "division"],
        ].copy()

        out_file = (
            output_dir
            / f"{division}_meter_groups.parquet"
        )

        # ----------------------------------------------------
        # REUSE TODAY'S CACHE ONLY IF VALID
        # ----------------------------------------------------

        if out_file.exists() and not force_rebuild:
            try:
                existing = pd.read_parquet(out_file)

                cached_meters = (
                    existing["meter_number"]
                    .astype("string")
                    .str.strip()
                )

                expected_meters = (
                    division_population["meter_number"]
                    .astype("string")
                    .str.strip()
                )

                if (
                    cached_meters.isna().any()
                    or cached_meters.eq("").any()
                    or cached_meters.duplicated().any()
                ):
                    raise ValueError(
                        "Cached meter IDs must be unique "
                        "and nonblank."
                    )

                if set(cached_meters) != set(expected_meters):
                    raise ValueError(
                        "Cached meters differ from the "
                        "current AMI population."
                    )

                cached_divisions = (
                    existing["division"]
                    .astype("string")
                    .str.strip()
                )

                if not cached_divisions.eq(
                    str(division).strip()
                ).all():
                    raise ValueError(
                        "Cached grouping contains another division."
                    )

                _validate_partition(
                    division_population,
                    existing,
                    division,
                )

            except (
                OSError,
                ValueError,
                AssertionError,
                KeyError,
            ) as exc:
                print(
                    f"[{division}] today's grouping cache "
                    f"is invalid; rebuilding. Reason: {exc}"
                )

            else:
                print(
                    f"[{division}] reusing validated "
                    f"grouping for {today}"
                )

                outputs[division] = out_file
                saved_population += len(existing)
                continue

        # ----------------------------------------------------
        # BUILD GROUPS
        # ----------------------------------------------------

        if group_source == "itron":
            if itron_groups_df is None:
                raise ValueError(
                    "group_source='itron' requires "
                    "itron_groups_df."
                )

            groups = build_itron_group_assignments(
                meter_population=division_population,
                itron_groups=itron_groups_df,
            )

        else:
            if smartlink_cluster_root is None:
                raise ValueError(
                    "group_source='smartlink' requires "
                    "smartlink_cluster_root."
                )

            clustered_file = (
                smartlink_cluster_root
                / f"{division}_meters_in_clusters.parquet"
            )

            standalone_file = (
                smartlink_cluster_root
                / f"{division}_standalone_meters.parquet"
            )

            if (
                not clustered_file.is_file()
                or not standalone_file.is_file()
            ):
                raise FileNotFoundError(
                    f"[{division}] SmartLink grouping "
                    "inputs are incomplete."
                )

            groups = build_smartlink_group_assignments(
                clustered=pd.read_parquet(clustered_file),
                standalone=pd.read_parquet(standalone_file),
                division=division,
            )

        # ----------------------------------------------------
        # VALIDATE AND SAVE
        # ----------------------------------------------------

        _validate_partition(
            division_population,
            groups,
            division,
        )

        _atomic_write_parquet(
            groups,
            out_file,
        )

        outputs[division] = out_file
        saved_population += len(groups)

        assigned = int(
            groups["tx_group_id"].notna().sum()
        )

        print(
            f"[{division}] saved {len(groups):,} meters "
            f"({assigned:,} assigned, "
            f"{len(groups) - assigned:,} unassigned)"
        )

    # --------------------------------------------------------
    # VALIDATE TOTAL POPULATION
    # --------------------------------------------------------

    if saved_population != len(population):
        raise ValueError(
            "Sum of division grouping populations does not "
            "equal the filtered AMI population: "
            f"grouping={saved_population:,}, "
            f"AMI={len(population):,}."
        )

    print(
        "Grouping population validated: "
        f"{saved_population:,} unique AMI meters across "
        f"{len(outputs):,} divisions."
    )

    return outputs