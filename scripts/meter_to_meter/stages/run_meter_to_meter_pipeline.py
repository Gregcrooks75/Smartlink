"""Run the complete SmartLink pipeline."""

from mtx.ingestion.ami_cache import (
    load_or_refresh_ny_ami,
)
from mtx.run_config import RunConfig
from deduplicate_meter_pairs import deduplicate_meter_pairs
from run_meter_correlations import (
    run_meter_correlations,
)
from run_tile_features import (
    run_tile_feature_stage,
)
from run_meter_pair_features import (
    run_meter_pair_feature_stage,
)
from run_weekly_voltages import (
    run_weekly_voltage_stage,
)
from validate_meter_pair_features import (
    validate_meter_pair_features,
)
from validate_meter_pairs import (
    validate_meter_pairs,
)


CONFIG = RunConfig(
    week_start="2026-02-02",
    week_end="2026-02-08",
    correlation_start="2026-02-02",
    correlation_end="2026-02-08",
    meter_correlation_radius=550,
    correlation_method="spearman",
    weekly_partition_size=25_000,
    seed_batch_size=5_000,
    force_ami_refresh=False,
)


def main() -> None:
    """Run and validate every SmartLink stage."""
    print("\nSmartLink run configuration")

    print(
        "Weekly dates: "
        f"{CONFIG.week_start} to {CONFIG.week_end}"
    )

    print(
        "Correlation dates: "
        f"{CONFIG.correlation_start} to "
        f"{CONFIG.correlation_end}"
    )

    print(
        "Radius: "
        f"{CONFIG.meter_correlation_radius:g} ft"
    )

    print(
        f"Method: {CONFIG.correlation_method}"
    )

    print(
        "\n0. Loading or refreshing AMI data..."
    )

    ami_df = load_or_refresh_ny_ami(
        force_refresh=CONFIG.force_ami_refresh,
    )

    print(
        "\n1. Running weekly-voltage stage..."
    )

    weekly_files = run_weekly_voltage_stage(
        config=CONFIG,
        ami_df=ami_df,
    )

    print("\n2. Running meter-correlation stage...")

    pair_files, all_pairs = run_meter_correlations(
        config=CONFIG,
        weekly_files=weekly_files,
        ami_df=ami_df,
    )


    print("\n2.5. Validating meter-pair outputs...")

    validate_meter_pairs(
        config=CONFIG,
        pair_files=pair_files,
    )

    print(
        "\n3. Running meter-pair feature stage..."
    )

    feature_files = run_meter_pair_feature_stage(
        config=CONFIG,
        pair_files=pair_files,
        ami_df=ami_df,
    )

    print(
        "\n3.5. Validating meter-pair features..."
    )

    validate_meter_pair_features(
        config=CONFIG,
        feature_files=feature_files,
    )

    # ==================================================
    # 4. TILE FEATURES
    # ==================================================

    print(
        "\n4. Running tile-feature stage..."
    )

    tile_feature_files = run_tile_feature_stage(
        feature_files=feature_files,
    )

    # ==================================================
    # COMPLETE
    # ==================================================

    print(
        "\nSmartLink completed successfully."
    )

    print(
        f"AMI records: {len(ami_df):,}"
    )

    print(
        f"Weekly divisions: {len(weekly_files)}"
    )

    print(
        f"Pair divisions: {len(pair_files)}"
    )

    print(
        f"Feature divisions: {len(feature_files)}"
    )

    print(
        "Tile-feature divisions: "
        f"{len(tile_feature_files)}"
    )


if __name__ == "__main__":
    main()