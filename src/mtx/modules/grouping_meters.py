# filename: grouping.py

from __future__ import annotations
from collections import defaultdict, Counter
from typing import Dict, Iterable, Optional, Tuple
import pandas as pd

def assign_groups_under_threshold(
    df: pd.DataFrame,
    *,
    meter_col: str = "Meter_Number",
    matched_col: str = "Matched_Meter_Number",
    distance_col: str = "Distance_Meter_Matched_Meter",
    threshold: float = 550.0,
    add_flag: bool = True,
) -> pd.DataFrame:
    """
    Build connected-component groups using ONLY pairs with distance < threshold.
    For any row whose pair distance is >= threshold, the group fields are set to NA.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least [meter_col, matched_col, distance_col].
    meter_col : str
        Column name for the primary meter ID.
    matched_col : str
        Column name for the matched meter ID.
    distance_col : str
        Column name for the distance between meter and matched meter (in feet).
    threshold : float
        Distance threshold in feet. Only pairs strictly below this are used to form groups.
    add_flag : bool
        If True, adds a boolean column 'ConformsThreshold' indicating distance < threshold.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with added columns:
        - 'Group' (component ID like G0001, G0002, …) or if non-conforming
        - 'GroupSize' (size of that component) or if non-conforming
        - 'GroupKey' (smallest meter ID in the component) or if non-conforming
        - 'ConformsThreshold' (optional boolean flag)

    Notes
    -----
    - Groups are formed *only* from rows where distance < threshold.
    - Any row with distance >= threshold will have Group/GroupSize/GroupKey blanked.
    - Meters that never appear in any < threshold pair will remain ungrouped (NA).
    """
    # Basic validation
    missing = {c for c in (meter_col, matched_col, distance_col) if c not in df.columns}
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Work on a shallow copy; we won't mutate the user's df in place
    out = df.copy()

    # --- Split allowed (< threshold) vs everything else -----------------------
    pairs_all = (
        out.loc[out[distance_col].notna(), [meter_col, matched_col, distance_col]]
        .dropna()
        .astype({meter_col: str, matched_col: str})
    )

    # Normalize whitespace
    pairs_all[meter_col] = pairs_all[meter_col].str.strip()
    pairs_all[matched_col] = pairs_all[matched_col].str.strip()

    allowed_pairs = pairs_all.loc[
        pairs_all[distance_col] < threshold, [meter_col, matched_col]
    ]

    # Build undirected unique edges, excluding self-edges
    allowed_edges = {
        tuple(sorted(t))
        for t in allowed_pairs.itertuples(index=False, name=None)
        if t[0] != t[1]
    }

    # --- Union-Find over allowed edges only -----------------------------------
    class UnionFind:
        __slots__ = ("parent", "rank")

        def __init__(self, items: Iterable[str]):
            self.parent = {x: x for x in items}
            self.rank = {x: 0 for x in items}

        def find(self, x: str) -> str:
            # Path compression
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, a: str, b: str) -> None:
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return
            # Union by rank
            if self.rank[ra] < self.rank[rb]:
                self.parent[ra] = rb
            elif self.rank[ra] > self.rank[rb]:
                self.parent[rb] = ra
            else:
                self.parent[rb] = ra
                self.rank[ra] += 1
    # Nodes are ONLY meters that appear in at least one < threshold relation
    all_nodes = set(allowed_pairs[meter_col]).union(set(allowed_pairs[matched_col]))

    # If no allowable edges, just return NA columns (and flag if requested)
    if not all_nodes:
        out["Group"] = pd.NA
        out["GroupSize"] = pd.NA
        out["GroupKey"] = pd.NA
        if add_flag:
            out["ConformsThreshold"] = out[distance_col].lt(threshold)
        return out

    uf = UnionFind(all_nodes)
    for a, b in allowed_edges:
        uf.union(a, b)

    # Representatives and group labels
    rep = {x: uf.find(x) for x in all_nodes}
    reps_sorted = sorted(set(rep.values()))
    rep_to_gid = {r: f"G{1+i:04d}" for i, r in enumerate(reps_sorted)}
    node_to_group = {node: rep_to_gid[rep[node]] for node in all_nodes}

    # Group size & canonical key (smallest meter ID per component)
    members = defaultdict(list)
    for node, r in rep.items():
        members[r].append(node)

    sizes_by_rep = {r: len(members[r]) for r in members}
    canon_by_rep = {r: min(members[r]) for r in members}

    group_size = {node: sizes_by_rep[rep[node]] for node in all_nodes}
    group_key = {node: canon_by_rep[rep[node]] for node in all_nodes}

    # --- Attach to original rows (meters not in any < threshold relation → NA) -
    out[meter_col] = out[meter_col].astype(str).str.strip()
    out["Group"] = out[meter_col].map(node_to_group)
    out["GroupSize"] = out[meter_col].map(group_size)
    out["GroupKey"] = out[meter_col].map(group_key)

    # Blank out group info for any non-conforming row (distance >= threshold)
    mask_bad = out[distance_col].ge(threshold)
    out.loc[mask_bad, ["Group", "GroupSize", "GroupKey"]] = pd.NA

    if add_flag:
        out["ConformsThreshold"] = out[distance_col].lt(threshold)

    return out