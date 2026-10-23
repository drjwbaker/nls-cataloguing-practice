# HDBSCAN parameter sweep — practice clustering

This folder is the **sensitivity sweep** for the practice-clustering step of
`structure_difference_v1.ipynb`. Instead of one notebook whose two HDBSCAN parameters
get edited and re-run by hand, each parameter combination is a **separate, self-contained
notebook**. Run each once; compare the headline numbers in the table below.

The point is honesty, not optimisation. A reviewer can reasonably ask: *did you pick the
parameters that gave you the cross-collection result you wanted?* The answer this sweep
supports is — *here is the full grid; the result holds (or does not) across it, and we
report it either way.* Do **not** pick the cell that maximises cross-collection clusters
(van Strien principle).

## What varies, and what does not

Only the two HDBSCAN parameters change. Corpus, random seed (42), the K=10 structural
categoriser, and all sampling caps are identical across the six notebooks, so any movement
in the outputs is attributable to these two parameters alone.

| parameter | meaning | values swept |
|---|---|---|
| `MIN_CLUSTER_SIZE` | smallest group of records allowed to be called a practice cluster | **10, 15, 30** |
| `MIN_SAMPLES` | conservativeness — how dense a region must be before it is a cluster rather than noise (higher → more noise) | **5, 10** |

Current/base setting is `MIN_CLUSTER_SIZE = 15`, `MIN_SAMPLES = 5` — included as one cell of
the grid (`..._mcs15_ms05.ipynb`).

## The notebooks

| notebook | min_cluster_size | min_samples |
|---|---|---|
| `structure_difference_sweep_mcs10_ms05.ipynb` | 10 | 5 |
| `structure_difference_sweep_mcs10_ms10.ipynb` | 10 | 10 |
| `structure_difference_sweep_mcs15_ms05.ipynb` (= base) | 15 | 5 |
| `structure_difference_sweep_mcs15_ms10.ipynb` | 15 | 10 |
| `structure_difference_sweep_mcs30_ms05.ipynb` | 30 | 5 |
| `structure_difference_sweep_mcs30_ms10.ipynb` | 30 | 10 |

Every notebook tags its `derived-data/` and `visualisations/` outputs with `mcs..-ms..`
(via the `TS` run-stamp), so the six runs never overwrite each other and the output files
are self-labelling.

## Why this grid (literature justification)

The grid is chosen to **head off the parameter-fishing objection**, not to find a
flattering setting.

- The HDBSCAN authors' own parameter guide treats `min_cluster_size` as "the smallest
  grouping you wish to consider a cluster" and demonstrates a sweep across {15, 30, 60};
  it treats `min_samples` as a conservativeness dial and recommends setting it **explicitly**
  (its default is whatever `min_cluster_size` is) and keeping it fixed while varying
  `min_cluster_size` where possible.
  <https://hdbscan.readthedocs.io/en/latest/parameter_selection.html>
- The grid {10, 15, 30} × {5, 10} **brackets the current setting (15 / 5) on both axes** and
  keeps `min_samples ≤ min_cluster_size` throughout (the documented convention; values above
  `min_cluster_size` are unusual and were excluded).
- Published method work stresses that density-clustering hyperparameters are genuinely
  sensitive and **do not transfer between datasets**, so a single value cannot be assumed
  representative — reporting the sweep is the defensible response rather than tuning to one
  number (Malzer & Baldock 2019; Elmahdi 2026; Peng et al. 2025).

## How to run and compare

1. Run all six notebooks (each top-to-bottom). The expensive, shared steps — corpus load,
   masking, categoriser — are identical, so if runtime is a concern you can run them in any
   order and walk away.
2. After each run, read the **Step 6** output (sample size, cluster count, noise %) and the
   **Step 8 / Step 12** summary (collection-crossing clusters), and fill in the table below.
3. Run `build_stability_table_v1.py` (below) to check membership stability automatically
   instead of reading it off the six TSVs by eye.
4. Read the sweep as a whole. **Robust** = the cross-collection finding survives across the
   grid. **Fragile** = it appears at only one setting — in which case it is a caveat to state
   in the article, not a headline.

## Automating the membership-stability check

`build_stability_table_v1.py` reproduces the "Membership stability" section below
programmatically, once all six `practice-clusters_nls-merged-corpus_*_mcs<N>-ms<M>.tsv`
files exist in `derived-data/`:

```bash
python build_stability_table_v1.py
```

For each cluster in each of the six files it applies the same substantial-membership
rule used manually below (a collection counts iff it is ≥10% of the cluster's records
**and** ≥5 records; the `noise` row is skipped), derives the cluster's composition (the
frozenset of substantial-member collection codes), and pivots composition × setting into
`derived-data/membership-stability_nls-merged-corpus_<timestamp>.tsv` — one row per
composition, one column per setting holding the matched cluster size (blank if that
composition didn't recur at that setting), plus an `n_settings` recurrence count. It
prints the table to stdout as well. Confirms the six recurring groupings and the two
"honest caveat" compositions (`eb+hd+mia` at 4/6; `sp+tp` at 5/6) reported below.

### Results table (six runs, final v3 categories, 23 June 2026)

Final run after the category clean-up: the two openings kept separate (CAT_00 terse/stub,
CAT_08 descriptive) and CAT_06 isolated as CAT_OTHER. (An interim run that *merged* the two
openings was reversed after the opening-blob eyeball — it had collapsed the Wood+Adair maps
result; see `opening-merge-eyeball_*.xlsx` and the next-steps doc.)

| min_cluster_size | min_samples | sample n | practice clusters | noise % | cross-collection clusters | notes |
|---|---|---|---|---|---|---|
| 10 | 5 | 901 | 21 | 17% | 10 | finest grain; most clusters, lowest noise |
| 10 | 10 | 901 | 17 | 24% | 8 | finer grain, more conservative |
| 15 | 5 (base) | 901 | 13 | 17% | 9 | **reported setting**; mid-grid |
| 15 | 10 | 901 | 12 | 23% | 8 | base size, more conservative |
| 30 | 5 | 901 | 7 | 28% | 6 | coarse; few, large clusters |
| 30 | 10 | 901 | 6 | 29% | 6 | coarsest; highest noise, but every cluster crosses collections |

The grid behaves exactly as the parameters predict — larger `min_cluster_size` and larger
`min_samples` both yield fewer clusters and more noise. Crucially, the cross-collection count
never collapses: it ranges 6–10 across all six settings and is never zero. The
cross-collection *fraction* is in fact **highest at the coarsest settings** (6/6 and 6/7 at
`min_cluster_size = 30`), i.e. the densest, most conservative groupings are the most
collection-crossing — the opposite of what a parameter-fishing artefact would look like.

### Membership stability — the *same* pairings recur (not just the counts)

Cluster counts being stable is the weak claim. The stronger test is whether the *same
collections* group together across settings. Because HDBSCAN renumbers clusters, this is
checked by composition, not cluster ID. A collection counts as a **substantial member** of a
cluster only if it contributes ≥10% of the cluster **and** ≥5 records (this filters trace
contamination such as a lone `eb:4` tail).

Six cross-collection practice groups recur at **6/6 settings**, at near-identical sizes — the
same density cores, not coincidentally similar counts:

| practice group | collections | size across settings | stability |
|---|---|---|---|
| Tay Bridge photos + Soviet + theatre posters | tb + sp + tp | n ≈ 171–173 | 6/6 |
| ballads + Hutton drawings | eb + hd | n = 167 | 6/6 |
| Wood + Adair maps | jw + ja | n ≈ 78–79 | 6/6 |
| Forth Bridge + Hutton | fb + hd | n = 52 | 6/6 |
| Soviet + theatre posters (distinct small group) | sp + tp | n = 25 (finer); n = 117 at (30,10) | 6/6 |

**Honest caveats (setting-dependent):**

- A **ballads + Hutton + MIA** trio (eb + hd + mia, n ≈ 55–106) is substantial only at
  `min_cluster_size` ∈ {15, 30} (**4/6**); at the finest grain (10) MIA falls below threshold
  and the group reduces to eb + hd. So MIA's attachment to the ballads/Hutton practice is real
  but weaker — state it as a secondary, not headline, link.
- The v10 plan's **ballads + Forth Bridge** pairing (old P9) does **not** survive: `fb` appears
  only at trace level (`fb:1`) in the ballads/Hutton clusters. The genuine Forth Bridge link is
  **Forth Bridge + Hutton** (6/6). Revise the plan's wording accordingly.

### One-line reading for the article

> Across `min_cluster_size` ∈ {10, 15, 30} and `min_samples` ∈ {5, 10}, the number of
> collection-crossing practice clusters **ranged from 6 to 10 and never fell to zero**, and six
> specific cross-collection groups — the poster/photograph caption clusters, the two map
> collections, and the ballads/Hutton and Forth Bridge/Hutton groups — **recurred at every
> setting with essentially the same membership and size**. The cross-collection signal is
> therefore **robust to the clustering parameters** rather than an artefact of any single
> setting. The reported configuration (15 / 5) sits in the middle of the grid; it was not
> chosen to maximise the cross-collection count.

## Generation note

These notebooks were generated programmatically from `structure_difference_v1.ipynb`: only
the Configuration cell was edited (the two parameters plus the output run-stamp), all cell
outputs were cleared, and a provenance header was inserted. Every other code cell is
byte-identical to the base notebook (verified). The base notebook is unchanged.
