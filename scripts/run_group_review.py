from pathlib import Path
import pandas as pd


def add_group_agreement(predictions):
    """One row per meter; use predictions before confidence filtering."""

    out = predictions.copy()

    if out["meter_number"].duplicated().any():
        raise ValueError("Expected one prediction per meter.")

    if out["recommended_transformer"].isna().any():
        raise ValueError("Every meter must have a recommendation.")

    keys = ["division", "subgroup_id"]

    votes = (
        out.groupby(keys + ["recommended_transformer"])
        .size()
        .rename("votes")
        .reset_index()
    )

    votes["group_prediction_count"] = (
        votes.groupby(keys)["votes"].transform("sum")
    )

    votes["group_recommendation_agreement_pct"] = (
        100 * votes["votes"] / votes["group_prediction_count"]
    )

    leaders = (
        votes.sort_values("votes", ascending=False)
        .drop_duplicates(keys)
    )

    valid = (
        leaders["group_prediction_count"].gt(1)
        & leaders["group_recommendation_agreement_pct"].gt(50)
    )

    leaders["group_majority_recommendation"] = (
        leaders["recommended_transformer"].where(valid)
    )

    leaders["group_recommendation_agreement_pct"] = (
        leaders["group_recommendation_agreement_pct"].where(
            leaders["group_prediction_count"].gt(1)
        )
    )

    out = out.merge(
        leaders[
            keys
            + [
                "group_prediction_count",
                "group_recommendation_agreement_pct",
                "group_majority_recommendation",
            ]
        ],
        on=keys,
        how="left",
        validate="many_to_one",
    )

    out["recommendation_matches_group_majority"] = (
        out["recommended_transformer"]
        .eq(out["group_majority_recommendation"])
        .astype("boolean")
        .where(out["group_majority_recommendation"].notna())
    )

    return out


def run_group_review(model_directory, split="validation"):
    folder = Path(model_directory)

    predictions = pd.read_parquet(
        folder / f"{split}_predictions.parquet"
    )

    review = add_group_agreement(predictions)

    output = folder / f"{split}_group_review.parquet"

    review.to_parquet(
        output,
        index=False,
    )

    summary = (
        review.loc[review["proposed_change"]]
        .groupby(
            "recommendation_matches_group_majority",
            dropna=False,
        )["correct"]
        .agg(
            proposed_corrections="size",
            precision="mean",
        )
    )

    summary.to_csv(
        folder / f"{split}_group_review_summary.csv"
    )

    print(
        f"\nGroup review ({split}, before score-gap filtering):\n"
        f"{summary}"
    )

    print(f"Saved: {output}")

    return output