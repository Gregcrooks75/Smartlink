from __future__ import annotations

from pathlib import Path
from datetime import date

import geopandas as gpd
import pandas as pd
import polars as pl


# ======================================================
# INGESTION
# ======================================================

from mtx.ingestion.ami_cache import (
    load_or_refresh_ny_ami,
)

from mtx.ingestion.transformer_cache import (
    load_or_refresh_ny_transformers,
)

from mtx.ingestion.itron_group_cache import (
    load_or_refresh_itron_groups,
)

from mtx.ingestion.itron_groups import (
    build_itron_tx_subgroups,
)

from mtx.ingestion.adam_cache import (
    load_or_refresh_adam_data,
)


# ======================================================
# HISTORY
# ======================================================

from meter_to_transformer.stages.run_historical_corrections import (
    run_historical_corrections,
)

from mtx.history.adam_history import (
    track_adam_history,
    get_adam_training_changes,
)

from mtx.history.ami_history import (
    track_ami_history,
)

from mtx.history.itron_history import (
    track_grouping_history,
)

from mtx.history.transformer_history import (
    track_transformer_history,
)

from mtx.history.grouping_history import (
    track_meter_grouping_history,
)

from mtx.history.candidate_history import (
    track_candidate_history,
)

from mtx.history.candidate_feature_history import (
    track_candidate_feature_history,
)


# ======================================================
# CONFIG
# ======================================================

from mtx.run_config import RunConfig


# ======================================================
# DATA CLEANING
# ======================================================

from meter_to_transformer.data_cleaning.standardize_inputs import (
    standardize_all_inputs,
    save_standardized_cache,
)

from meter_to_transformer.data_cleaning.validate_coordinates import (
    validate_coordinates,
    save_rejection_log,
)


# ======================================================
# PIPELINE STAGES
# ======================================================

from meter_to_transformer.stages.run_meter_grouping import (
    run_meter_grouping_stage,
)

from meter_to_transformer.stages.run_transformer_candidates import (
    run_transformer_candidate_stage,
)

from meter_to_transformer.stages.run_candidate_features import (
    run_candidate_feature_stage,
)

from meter_to_transformer.stages.run_tile_features import (
    TileFeatureConfig,
    run_tile_feature_stage,
)

from meter_to_transformer.stages.run_final_features import (
    run_final_feature_stage,
)


# ======================================================
# DATABASE
# ======================================================

from mtx.db.db_AMI import (
    load_historical_ami,
    get_ny_ami_data,
)


# ======================================================
# TRAINING
# ======================================================

from meter_to_transformer.training.run_training_data import (
    run_training_data_stage,
)

from meter_to_transformer.training.run_model_training import (
    run_model_training_stage,
)


# ======================================================
# MODEL REVIEW
# ======================================================

from run_correction_review import (
    review_corrections,
)

from run_group_review import (
    run_group_review,
)
# ======================================================
# PREDICTIONS
# ======================================================


from meter_to_transformer.prediction.run_recommendations import (
run_recommendations_stage,
import_review_updates,
)


# ======================================================
# PROJECT PATHS
# ======================================================

PROJECT_ROOT = Path(
    r"C:\Users\u355394\OneDrive - IBERDROLA S.A"
    r"\Documents - Asset Data Strategy"
    r"\Asset Data Quality\SmartLink"
)

run_date = date.today()


# ======================================================
# CONFIGURATION
# ======================================================

CONFIG = RunConfig(
    week_start="2026-02-02",
    week_end="2026-02-08",
    correlation_start="2026-02-02",
    correlation_end="2026-02-08",
    meter_correlation_radius=550,
    transformer_candidate_radius_ft=550,
    correlation_method="spearman",
    group_source="itron",
    weekly_partition_size=25_000,
    seed_batch_size=5_000,
    force_ami_refresh=False,
    force_transformer_refresh=False,
    force_itron_refresh=False,
    transformer_division_col="Division",
    overwrite_m2t_outputs=False,
)


# ======================================================
# DIVISION FILES
# ======================================================

NYSEG_DIVISION_FILE = (
    PROJECT_ROOT
    / "data"
    / "data augmentation"
    / "NYSEG_Electric_Divisions.shp"
)

RGE_DIVISION_FILE = (
    PROJECT_ROOT
    / "data"
    / "data augmentation"
    / "RGE_Electric_Divisions.shp"
)


# ======================================================
# ITRON REQUIRED COLUMNS
# ======================================================

ITRON_REQUIRED_COLUMNS = (
    "meter_number",
    "Tx_Subgroup_ID",
    "tx_group_id",
    "tx_group_size",
    "group_source",
    "group_type",
)


# ======================================================
# MAIN
# ======================================================

def main():

    print("\n" + "=" * 80)
    print("SMARTLINK METER-TO-TRANSFORMER PIPELINE")
    print("=" * 80)

    # ==================================================
    # 1. LOAD INPUT DATA
    # ==================================================

    print("\n1. Loading Inputs...")

    ami_df = load_or_refresh_ny_ami(
        force_refresh=CONFIG.force_ami_refresh,
    )

    transformer_df = load_or_refresh_ny_transformers(
        force_refresh=CONFIG.force_transformer_refresh,
    )

    itron_groups_df = load_or_refresh_itron_groups(
        refresh_fn=build_itron_tx_subgroups,
        force_refresh=CONFIG.force_itron_refresh,
        max_age_days=1,
        required_columns=ITRON_REQUIRED_COLUMNS,
    )

    adam_df = load_or_refresh_adam_data(
        force_refresh=False,
    )

    
    full_ami_df = ami_df.copy()

    # ==================================================
    # 2. LOAD DIVISION POLYGONS
    # ==================================================

    print("\n2. Loading Division Polygons...")

    nyseg = gpd.read_file(
        NYSEG_DIVISION_FILE
    )

    rge = gpd.read_file(
        RGE_DIVISION_FILE
    )

    nyseg = nyseg.rename(
        columns={"DIVISION": "Division"}
    )

    rge = rge.rename(
        columns={"DISTRICT": "Division"}
    )

    division_gdf = gpd.GeoDataFrame(
        pd.concat(
            [
                nyseg[["Division", "geometry"]],
                rge[["Division", "geometry"]],
            ],
            ignore_index=True,
        ),
        geometry="geometry",
    )

    transformer_df = transformer_df[
        transformer_df["StructureType"] != "Vert"
    ].reset_index(
        drop=True
    )

    division_gdf = division_gdf.to_crs(
        epsg=4326
    )


    # ==================================================
    # 3. STANDARDIZE INPUT DATA
    # ==================================================

    print("\n3. Standardizing Inputs...")

    (
        ami_df,
        transformer_df,
        itron_groups_df,
        division_gdf,
        adam_df,
    ) = standardize_all_inputs(
        ami_df=ami_df,
        transformer_df=transformer_df,
        itron_groups_df=itron_groups_df,
        division_gdf=division_gdf,
        adam_df=adam_df,
    )

    # After standardize_all_inputs(), before coordinate validation:
    full_transformer_df = transformer_df.copy()
    
  
    
    # ==================================================
    # 4. RECONCILE HISTORICAL RECOMMENDATIONS
    # ==================================================

    print("\n4. Reconciling Historical Recommendations...")

    historical_corrections_df = run_historical_corrections(
        ami_df=ami_df
    )


    # ==================================================
    # 5. TRACK INPUT HISTORY
    # ==================================================

    print("\n5. Tracking History...")

    ami_changes = track_ami_history(
        current_df=ami_df,
    )

    print(
        f"AMI changes detected: "
        f"{len(ami_changes):,}"
    )

    adam_changes = track_adam_history(
        current_df=adam_df,
    )

    print(
        f"Adam changes detected: "
        f"{len(adam_changes):,}"
    )

    itron_changes = track_grouping_history(
        current_df=itron_groups_df,
    )

    print(
        f"ITRON changes detected: "
        f"{len(itron_changes):,}"
    )

    transformer_changes = track_transformer_history(
        current_df=transformer_df,
    )

    print(
        f"Transformer changes detected: "
        f"{len(transformer_changes):,}"
    )


    # ==================================================
    # 6. VALIDATE COORDINATES
    # ==================================================

    print("\n6. Validating Coordinates...")

    ami_df, invalid_meters = (
        validate_coordinates(
            df=ami_df,
            division_gdf=division_gdf,
        )
    )

    transformer_df, invalid_transformers = (
        validate_coordinates(
            df=transformer_df,
            division_gdf=division_gdf,
        )
    )

    print(
        f"Valid meters: "
        f"{len(ami_df):,}"
    )

    print(
        f"Rejected meters: "
        f"{len(invalid_meters):,}"
    )


    # ==================================================
    # 7. SAVE REJECTION LOG
    # ==================================================

    print(
        "\n7. Saving Rejection Log..."
    )

    save_rejection_log(
        invalid_meters=invalid_meters,
        invalid_transformers=invalid_transformers,
    )


    # ==================================================
    # 8. REMOVE REJECTED METERS AND TRANSFORMERS
    # ==================================================

    meters_to_omit = (
        invalid_meters
        .meter_number
        .tolist()
    )

    structures_to_omit = (
        invalid_transformers
        .transformer_floc
        .tolist()
    )

    print(
        fr"we currently have {len(meters_to_omit)} to omit"
    )

    print(
        fr"we currently have {len(meters_to_omit)} to omit"
    )

    ami_df = ami_df[
        ~ami_df["meter_number"].isin(meters_to_omit)
    ].reset_index(
        drop=True
    )

    transformer_df = transformer_df[
        ~transformer_df["transformer_floc"].isin(
            structures_to_omit
        )
    ].reset_index(
        drop=True
    )

    itron_groups_df = itron_groups_df[
        ~itron_groups_df["meter_number"].isin(
            meters_to_omit
        )
    ].reset_index(
        drop=True
    )

    adam_df = adam_df[
        ~adam_df["meter_number"].isin(
            meters_to_omit
        )
    ].reset_index(
        drop=True
    )


    # ==================================================
    # 9. BUILD METER GROUPS
    # ==================================================

    print("\n9. Building Meter Groups...")

    group_files = run_meter_grouping_stage(
        ami_df=ami_df,
        itron_groups_df=itron_groups_df,
    )

    grouping_changes = track_meter_grouping_history(
        group_files=group_files,
        ami_df=ami_df,
    )

    print(
        f"Grouping-result changes detected: "
        f"{len(grouping_changes):,}"
    )


    # ==================================================
    # 10. BUILD TRANSFORMER CANDIDATES
    # ==================================================

    print("\n10. Building Transformer Candidates...")

    candidate_outputs = run_transformer_candidate_stage(
        config=CONFIG,
        ami_df=ami_df,
        transformer_df=transformer_df,
    )

    print("Updating Candidate History")

    # track_candidate_history()


    # ==================================================
    # 11. BUILD CANDIDATE FEATURES
    # ==================================================

    print(
        "\n11. Building Candidate Features..."
    )

    from datetime import date

    run_date = date.today()

    candidate_feature_outputs = (
        run_candidate_feature_stage(
            config=CONFIG,
            candidate_files=candidate_outputs,
            group_files=group_files,
            ami_df=ami_df,
            transformer_df=transformer_df,
            snapshot_date=run_date,
        )
    )

    print(
        "Updating Candidate Feature History..."
    )

    candidate_feature_changes = (
        track_candidate_feature_history(
            snapshot_date=run_date,
        )
    )

    print(
        "Candidate feature changes detected: "
        f"{len(candidate_feature_changes):,}"
    )


    # ==================================================
    # 12. BUILD TILE FEATURES
    # ==================================================

    print(
        "\n12. Building Tile Features..."
    )

    TILE_CONFIG = TileFeatureConfig(
        grid_size_m=1000,
        aoi_buffer_m=50,
        max_rows_per_batch=500,
        checkpoint_every_batches=1,
        compression="zstd",
        max_building_hits_per_batch=50_000,
        delete_work_files_on_success=False,
    )

    SKIP_TILE_FEATURES = True  # False to run tile processing normally

    if SKIP_TILE_FEATURES:
        tile_feature_outputs = {
            division: (
                PROJECT_ROOT
                / "data"
                / "cache"
                / "tile_features"
                / "current"
                / f"{division}.parquet"
            )
            for division in candidate_feature_outputs
        }

        for path in tile_feature_outputs.values():
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing tile cache: {path}"
                )

        print(
            "Skipping tile processing; "
            "using existing tile files."
        )

    else:
        tile_feature_outputs = run_tile_feature_stage(
            config=CONFIG,
            candidate_feature_files=(
                candidate_feature_outputs
            ),
            candidate_feature_changes=(
                candidate_feature_changes
            ),
            tile_config=TILE_CONFIG,
        )


    # ==================================================
    # 13. ASSEMBLE FINAL FEATURE SETS
    # ==================================================

    print(
        "\n13. Assembling Final Feature Sets..."
    )
   
    final_feature_outputs = (
        run_final_feature_stage(
            candidate_feature_files=(
                candidate_feature_outputs
            ),
            tile_feature_files=(
                tile_feature_outputs
            ),
            snapshot_date=run_date,
        )
    )

  
   

    # ==================================================
    # 14. BUILD TRAINING DATA
    # ==================================================

    print("\n14. Building Training Data...")

    training_directory = run_training_data_stage(
        project_root=PROJECT_ROOT,
        run_date=run_date.isoformat(),
        adam_labels=pl.from_pandas(adam_df),
        current_ami=pl.DataFrame(ami_df[["meter_number", "current_meter_floc"]].to_dict(orient="list"), infer_schema_length=None,),
        ami_meter_col="meter_number",
        ami_transformer_col="current_meter_floc",
        subgroup_col="tx_group_id",
        quality_col="tx_current_transformer_count",
        exclude_previous_recommendations=False
    )

    print(
        f"Training data saved to: "
        f"{training_directory}"
    )


    # ==================================================
    # 15. TRAIN TRANSFORMER RANKER
    # ==================================================

    print("\n15. Training Transformer Ranker...")

    model_directory = run_model_training_stage(
        project_root=PROJECT_ROOT,
        training_directory=training_directory,
        exclude_features=(),
        evaluate_test=False,
    )


    # ==================================================
    # 16. REVIEW MODEL CORRECTIONS
    # ==================================================

    print("\n16. Reviewing Model Corrections...")

    review_corrections(
        model_directory,
        training_directory,
        target_precision=0.88,
        min_changes=100,
    )


    # ==================================================
    # 17. RUN GROUP REVIEW
    # ==================================================

    print("\n17. Run Group Review...")

    group_review_file = run_group_review(
        model_directory
    )

    # ==================================================
    # 18. RUN PREDICTIONS
    # ==================================================

    print("\n18. Run Predictions...")

    

    recommendation_directory = run_recommendations_stage(
        project_root=PROJECT_ROOT,
        model_date="2026-10-23",      # Choose your saved model
        min_score_gap=0.004216045,    # Bottom threshold in your screenshot
        ami_df=ami_df,
        full_ami_df=full_ami_df,
        transformer_df=full_transformer_df,
        final_feature_files=final_feature_outputs,
        include_previously_sent=True
    )

# ======================================================
# ENTRY POINT
# ======================================================

if __name__ == "__main__":
    main()