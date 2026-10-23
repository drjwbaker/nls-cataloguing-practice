#!/usr/bin/env python3
"""
build_stability_table_v1.py
============================
Membership-stability table for the HDBSCAN parameter sweep.

Reads the six ``practice-clusters_nls-merged-corpus_*_mcs<N>-ms<M>.tsv`` files in
``derived-data/`` (one per point in the {10,15,30} x {5,10} min_cluster_size /
min_samples grid; see ``parameter-sweep/README.md``), and asks the stronger
stability question: do the *same collections* group together across settings, not
just how many clusters/how much noise each setting produces (that count is already
reported in the README's results table).

For each cluster in each setting, a collection is a **substantial member** if it
contributes at least 10% of that cluster's records *and* at least 5 records (this
filters trace contamination, e.g. a lone ``eb:4`` tail riding along with a much
larger cluster). The cluster's **composition** is the frozenset of its substantial
member collection codes; a composition with 2+ members is cross-collection.

Because HDBSCAN relabels cluster ids run to run, stability is tracked by
composition (the *set* of collections), not by cluster id. The output is a
composition x setting pivot table: one row per composition, one column per
setting holding the matched cluster's size (blank if that composition did not
recur at that setting), plus an ``n_settings`` column counting how many of the six
settings the composition appears in at all.

Usage
-----
    python build_stability_table_v1.py

Writes ``derived-data/membership-stability_nls-merged-corpus_<timestamp>.tsv`` and
prints the table to stdout.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

CODE_DIR: Path = Path(__file__).resolve().parent
DERIVED_DIR: Path = CODE_DIR / "derived-data"

# A collection is a "substantial member" of a cluster iff it contributes at least
# this fraction of the cluster's records ...
MIN_MEMBER_FRACTION: float = 0.10
# ... AND at least this many records (filters trace contamination in large clusters).
MIN_MEMBER_COUNT: int = 5

# Matches the "mcs<N>-ms<M>" setting tag embedded in each sweep file's name, e.g.
# "practice-clusters_nls-merged-corpus_20260623-172717_mcs10-ms5.tsv" -> "mcs10-ms5".
_SETTING_RE = re.compile(r"(mcs\d+-ms\d+)")


def find_sweep_files(derived_dir: Path) -> list[Path]:
    """Return the six practice-clusters sweep TSVs in ``derived_dir``, sorted by setting.

    Sort order is by (min_cluster_size, min_samples) ascending, so columns in the
    final table read left-to-right as the grid gets coarser -- matching the README's
    results table order.
    """
    paths = sorted(derived_dir.glob("practice-clusters_nls-merged-corpus_*_mcs*-ms*.tsv"))
    assert paths, f"No practice-clusters_*_mcs*-ms*.tsv files found in {derived_dir}"

    def sort_key(p: Path) -> tuple[int, int]:
        m = _SETTING_RE.search(p.name)
        assert m, f"Could not parse mcs/ms setting from filename: {p.name}"
        mcs, ms = re.match(r"mcs(\d+)-ms(\d+)", m.group(1)).groups()
        return (int(mcs), int(ms))

    paths.sort(key=sort_key)
    return paths


def setting_label(path: Path) -> str:
    """Return the 'mcsN-msM' setting tag parsed from a sweep filename."""
    m = _SETTING_RE.search(path.name)
    assert m, f"Could not parse mcs/ms setting from filename: {path.name}"
    return m.group(1)


def parse_collections_field(field: str) -> dict[str, int]:
    """Parse a 'collections' cell (e.g. 'tb:91, sp:57, tp:18') into {code: count}.

    Raises no error on an empty/blank field (returns {}); the 'noise' row (which has
    no collection breakdown printed in this column format in some runs) is handled
    by the caller, which skips 'noise' clusters entirely before this is called.
    """
    counts: dict[str, int] = {}
    field = (field or "").strip()
    if not field:
        return counts
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        code, _, n = part.partition(":")
        counts[code.strip()] = int(n.strip())
    return counts


def substantial_members(collections: dict[str, int], cluster_size: int) -> frozenset[str]:
    """Return the frozenset of collection codes that are 'substantial members'.

    A collection is substantial iff it contributes >= MIN_MEMBER_FRACTION of
    cluster_size AND >= MIN_MEMBER_COUNT records. This filters trace contamination
    (e.g. a lone eb:4 tail in a 171-record cluster) so composition reflects the
    dominant practice grouping, not incidental overlap.
    """
    threshold = max(MIN_MEMBER_FRACTION * cluster_size, 0.0)
    return frozenset(
        code for code, n in collections.items()
        if n >= threshold and n >= MIN_MEMBER_COUNT
    )


def composition_label(composition: frozenset[str]) -> str:
    """Human-readable, order-independent label for a composition, e.g. 'eb+hd'."""
    return "+".join(sorted(composition))


def load_cluster_compositions(path: Path) -> dict[frozenset[str], int]:
    """Read one sweep TSV and return {composition: cluster_size} for its clusters.

    Skips the 'noise' row (unclustered records; not a practice cluster). Multiple
    *distinct* clusters in the same run can legitimately share a composition —
    this happens for single-collection ("no cross-collection") compositions, e.g.
    several different tp-only clusters at the same setting, each below the
    cross-collection bar. Composition is a set, not a cluster id, so these
    collapse to one row; the LARGEST such cluster's size is kept (a single
    representative figure for the table) and one consolidated note is printed per
    composition, not one per pair compared.
    """
    df = pd.read_csv(path, sep="\t")
    assert {"cluster", "size", "collections"}.issubset(df.columns), (
        f"{path.name} missing expected columns; found {list(df.columns)}"
    )
    sizes_by_composition: dict[frozenset[str], list[int]] = {}
    for _, row in df.iterrows():
        if str(row["cluster"]).strip().lower() == "noise":
            continue
        size = int(row["size"])
        collections = parse_collections_field(str(row["collections"]))
        composition = substantial_members(collections, size)
        if not composition:
            # No collection clears the substantial-membership bar (shouldn't happen
            # for a real cluster, since the dominant collection always qualifies,
            # but guard rather than silently drop).
            print(f"  WARNING: {path.name} cluster {row['cluster']!r} (size {size}) "
                  f"has no substantial member -- skipped.")
            continue
        sizes_by_composition.setdefault(composition, []).append(size)

    out: dict[frozenset[str], int] = {}
    for composition, sizes in sizes_by_composition.items():
        if len(sizes) > 1:
            print(f"  NOTE: {path.name} has {len(sizes)} distinct clusters sharing "
                  f"composition {composition_label(composition)} (sizes {sorted(sizes, reverse=True)}); "
                  f"recording the largest.")
        out[composition] = max(sizes)
    return out


def build_stability_table(files: list[Path]) -> pd.DataFrame:
    """Build the composition x setting pivot table across all sweep files.

    Rows are compositions (frozenset of substantial-member collection codes,
    rendered as a '+'-joined label); columns are one per setting holding the
    matched cluster size (NaN/blank if that composition did not recur at that
    setting); plus an 'n_settings' column counting recurrence across the grid, and
    a 'cross_collection' column flagging compositions with 2+ members.
    """
    settings = [setting_label(p) for p in files]
    per_setting: dict[str, dict[frozenset[str], int]] = {
        setting_label(p): load_cluster_compositions(p) for p in files
    }

    all_compositions: set[frozenset[str]] = set()
    for comp_sizes in per_setting.values():
        all_compositions.update(comp_sizes.keys())

    rows = []
    for comp in all_compositions:
        row = {"composition": composition_label(comp),
               "cross_collection": "YES" if len(comp) >= 2 else "no"}
        n_settings = 0
        for setting in settings:
            size = per_setting[setting].get(comp)
            row[setting] = size if size is not None else pd.NA
            if size is not None:
                n_settings += 1
        row["n_settings"] = n_settings
        rows.append(row)

    table = pd.DataFrame(rows)
    ordered_cols = ["composition", "cross_collection"] + settings + ["n_settings"]
    table = table[ordered_cols]
    # Most-recurrent, most cross-collection groupings first; alphabetical tie-break.
    table = table.sort_values(
        by=["n_settings", "cross_collection", "composition"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return table


def main() -> None:
    """Build and write the membership-stability table; print it to stdout."""
    files = find_sweep_files(DERIVED_DIR)
    print(f"Found {len(files)} sweep files:")
    for p in files:
        print(f"  {p.name}  (setting {setting_label(p)})")

    table = build_stability_table(files)
    assert not table.empty, "No cluster compositions found across the sweep -- check input files."

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = DERIVED_DIR / f"membership-stability_nls-merged-corpus_{ts}.tsv"
    table.to_csv(out_path, sep="\t", index=False)

    print(f"\nWrote {out_path} | {len(table)} distinct compositions across "
          f"{len(files)} settings\n")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
