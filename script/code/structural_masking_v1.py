#!/usr/bin/env python3
"""
structural_masking_v1.py
========================
Data-driven structural masking for the cross-catalogue cataloguing-practice
pipeline. Replaces the superseded, collection-keyed ``masking_v1.py`` (now in
``attic/``).

Why this exists (see research-notes/2026-06-18_NLS-merged-corpus_analysis-notes.md,
Renewal 2026-06-22)
-------------------------------------------------------------------------------
The first masked attempt classified each clause with a HAND-BUILT vocabulary
(FIRST_LINE, SHOTLIST_WITH_TC, PUBLICATION_NOTE, ...) whose regular expressions
were each tuned to one NLS collection's house style. A mask whose categories are
collection-diagnostic still sorts records by collection — it measures collection
membership under another name, and it cannot transfer to a catalogue it has never
seen.

This module instead derives the structural vocabulary FROM THE DATA:

  1. strip boilerplate (collection-agnostic; identical NLS Data Foundry strings
     only manufacture within-collection near-duplicates);
  2. segment each record into clauses;
  3. describe each clause by CONTENT-AGNOSTIC STRUCTURAL FEATURES only — length,
     digit/timecode presence, punctuation and capitalisation profile, function-
     word ratio, position — never by any specific vocabulary or collection cue;
  4. cluster the clause feature-vectors into K emergent functional categories
     (KMeans, seeded), with K chosen by an inspectable criterion;
  5. mask each record to (a) the ordered SEQUENCE of emergent category labels and
     (b) the PROFILE (count distribution) over categories.

Because the features carry no vocabulary, the same fitted categoriser runs on an
unseen catalogue. The category labels are neutral (CAT_00, CAT_01, ...); a
human-readable descriptor is derived post hoc from each cluster's dominant feature
profile, and exemplar clauses are surfaced for the researcher to label and
validate (see ``evaluation`` helpers). Following van Strien:
"correct" is defined on a labelled sample BEFORE the categoriser is trusted at
scale — this module produces the material for that judgement; it does not
fabricate accuracy figures.

All feature functions are pure and deterministic. Clustering is seeded.

Design choices for reproducible research
---------------------------------------
- Transparent, inspectable features (no opaque embedding); fully reproducible.
- KMeans over a documented K range with silhouette + seed-stability (ARI)
  reported, rather than a single magic K.
- A collection-independence diagnostic (adjusted mutual information between
  category and collection) is reported so the researcher can SEE whether the
  emergent categories are, or are not, collection proxies — the exact failure of
  the previous approach.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# ── 1. Boilerplate (collection-agnostic preprocessing) ─────────────────────────
# Carried over from the superseded module: these strings are identical across
# records and only create within-collection near-duplicates. Stripping them is
# preprocessing, not masking.
BOILERPLATE_FRAGMENTS: list[str] = [
    "Metadata context: metadata created as part of normal Library activities.",
    "Metadata created specifically for digitisation project.",
    "Metadata was prepared from a finding aid.",
    "Metadata created specifically for digitisation.",
]


def strip_boilerplate(text: str) -> str:
    """Remove all known NLS boilerplate fragments and collapse whitespace.

    Pure and deterministic. Operates on raw description text from either the
    ``dc:`` (digitised collections) or ``dcterms:`` (MIA) namespace — it is text
    replacement and is namespace-agnostic.
    """
    for frag in BOILERPLATE_FRAGMENTS:
        text = text.replace(frag, " ")
    return " ".join(text.split())


# ── 2. Segmentation ────────────────────────────────────────────────────────────
# Split a description field into clauses. We split on sentence-like boundaries but
# keep the splitter deliberately simple and language-light so it transfers across
# catalogues. A field with no internal boundary yields a single clause.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9\[])")


def segment_record(descriptions: Iterable[str]) -> list[str]:
    """Return the ordered list of non-empty clauses across all description fields.

    Boilerplate is stripped first. Handles both DC namespaces (the caller passes
    the raw text of every dc:description / dcterms:description field).
    """
    clauses: list[str] = []
    for field_text in descriptions:
        text = strip_boilerplate(field_text or "")
        if not text:
            continue
        for clause in _CLAUSE_SPLIT_RE.split(text):
            clause = clause.strip()
            if clause:
                clauses.append(clause)
    return clauses


# ── 3. Content-agnostic structural features ────────────────────────────────────
# A small, closed-class English function-word set. Its ONLY role is to gauge how
# "prose-like" vs "list-like" a clause is (high ratio => running prose; low ratio
# => terse/structured). It carries no topical content. For a non-English catalogue
# this set would be swapped for the relevant closed-class list — flagged in the
# methods section as the single language-dependent knob.
_FUNCTION_WORDS: frozenset[str] = frozenset("""
a an the and or but nor of to in on at by for with from into onto over under
is are was were be been being has have had do does did this that these those
it its as not no he she they we you i him her them his their our your my me
which who whom whose where when while if then than so such about after before
""".split())

_TIMECODE_RE = re.compile(r"\(\s*\d{0,2}[:.]\d{2}\s*\)")
_DIGIT_RUN_RE = re.compile(r"\d+")
_TOKEN_RE = re.compile(r"\b\w[\w'-]*\b", re.UNICODE)

# The ordered feature names define the columns of the feature matrix. Keeping this
# explicit makes the matrix self-documenting and the analysis reproducible.
FEATURE_NAMES: list[str] = [
    "log_n_tokens",          # length (log1p of token count)
    "mean_token_len",        # average token length in chars
    "digit_char_ratio",      # fraction of chars that are digits
    "has_timecode",          # 0/1 contains an (mm:ss)-style timecode
    "n_timecodes_log",       # log1p count of timecodes
    "comma_per_token",       # punctuation density features (structure, not content)
    "semicolon_per_token",
    "colon_per_token",
    "period_per_token",
    "bracket_per_token",     # ( ) [ ]
    "dash_per_token",
    "upper_char_ratio",      # fraction of alpha chars that are uppercase
    "titlecase_token_ratio", # fraction of tokens that are Title-Case (proper-noun/list proxy)
    "function_word_ratio",   # prose-vs-list proxy
    "ends_with_colon",       # 0/1 clause ends with ':' (introduces a list/field)
    "starts_with_digit",     # 0/1
    "alpha_token_ratio",     # fraction of tokens that are purely alphabetic
]


def clause_features(clause: str, position: float = 0.0) -> np.ndarray:
    """Return the content-agnostic structural feature vector for one clause.

    Parameters
    ----------
    clause : str
        A single clause (already boilerplate-stripped, from ``segment_record``).
    position : float
        Reserved for an optional positional feature; not used in the default
        feature set (kept out so that a clause's category does not depend on the
        length of the record it sits in). Accepted for interface stability.

    Returns
    -------
    np.ndarray  shape (len(FEATURE_NAMES),), dtype float64.

    The vector contains NO vocabulary — only counts, ratios and shape. Two clauses
    with identical structure but different subject matter map to the same vector.
    """
    tokens = _TOKEN_RE.findall(clause)
    n_tok = len(tokens)
    n_chars = max(len(clause), 1)
    if n_tok == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    alpha_chars = [c for c in clause if c.isalpha()]
    n_alpha = max(len(alpha_chars), 1)
    timecodes = _TIMECODE_RE.findall(clause)
    titlecase = sum(1 for t in tokens if t[:1].isupper() and t[1:].islower())
    func = sum(1 for t in tokens if t.lower() in _FUNCTION_WORDS)
    alpha_tokens = sum(1 for t in tokens if t.isalpha())

    feats = {
        "log_n_tokens": np.log1p(n_tok),
        "mean_token_len": np.mean([len(t) for t in tokens]),
        "digit_char_ratio": sum(c.isdigit() for c in clause) / n_chars,
        "has_timecode": 1.0 if timecodes else 0.0,
        "n_timecodes_log": np.log1p(len(timecodes)),
        "comma_per_token": clause.count(",") / n_tok,
        "semicolon_per_token": clause.count(";") / n_tok,
        "colon_per_token": clause.count(":") / n_tok,
        "period_per_token": clause.count(".") / n_tok,
        "bracket_per_token": (clause.count("(") + clause.count("[")) / n_tok,
        "dash_per_token": (clause.count("-") + clause.count("–")) / n_tok,
        "upper_char_ratio": sum(c.isupper() for c in alpha_chars) / n_alpha,
        "titlecase_token_ratio": titlecase / n_tok,
        "function_word_ratio": func / n_tok,
        "ends_with_colon": 1.0 if clause.rstrip().endswith(":") else 0.0,
        "starts_with_digit": 1.0 if clause[:1].isdigit() else 0.0,
        "alpha_token_ratio": alpha_tokens / n_tok,
    }
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)


# ── 4. The fitted categoriser ───────────────────────────────────────────────────
@dataclass
class StructuralCategoriser:
    """A fitted, data-driven map from clause -> emergent structural category.

    Fit once on all clauses in the corpus (``fit``), then apply to any record
    (``mask_record``) or any unseen catalogue. Holds the StandardScaler and KMeans
    model so the same transformation is applied everywhere — the single source of
    truth both analysis notebooks import.

    Attributes are populated by ``fit``; ``k`` is the chosen number of categories
    and ``random_seed`` makes the clustering reproducible.
    """

    k: int = 12
    random_seed: int = 42
    scaler: object = None          # sklearn StandardScaler
    model: object = None           # sklearn KMeans
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    labels_: list[str] = field(default_factory=list)  # CAT_00 ... CAT_{k-1}

    def label_for(self, cluster_id: int) -> str:
        """Neutral, order-stable category label (CAT_00, CAT_01, ...)."""
        return f"CAT_{cluster_id:02d}"

    def fit(self, clauses: Sequence[str]) -> "StructuralCategoriser":
        """Fit scaler + KMeans on the feature matrix of all corpus clauses."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = np.vstack([clause_features(c) for c in clauses])
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.model = KMeans(n_clusters=self.k, random_state=self.random_seed,
                            n_init=10).fit(Xs)
        self.labels_ = [self.label_for(i) for i in range(self.k)]
        return self

    def predict_clause(self, clause: str) -> str:
        """Return the emergent category label for a single clause."""
        if self.model is None:
            raise RuntimeError("Categoriser is not fitted. Call fit() first.")
        x = self.scaler.transform(clause_features(clause).reshape(1, -1))
        return self.label_for(int(self.model.predict(x)[0]))

    def predict_clauses(self, clauses: Sequence[str]) -> np.ndarray:
        """Vectorised: return the integer cluster id for each clause in one pass.

        Use this (not repeated ``predict_clause``) when masking the whole corpus —
        on the WSL2/Dropbox setup a per-clause ``transform`` call over 100k+ clauses
        is needlessly slow. One feature matrix, one transform, one predict.
        """
        if self.model is None:
            raise RuntimeError("Categoriser is not fitted. Call fit() first.")
        if not clauses:
            return np.empty(0, dtype=int)
        X = self.scaler.transform(np.vstack([clause_features(c) for c in clauses]))
        return self.model.predict(X)

    def mask_record(self, descriptions: Iterable[str],
                    collapse_repeats: bool = True) -> list[str]:
        """Mask a record to its ordered SEQUENCE of emergent category labels.

        Consecutive identical labels are collapsed by default (a long multi-clause
        narrative becomes one run, mirroring the old field-level behaviour) so that
        sequence similarity reflects the *order of cataloguing moves*, not the
        verbosity of any one move. An empty list means the record had no
        non-boilerplate description content.
        """
        seq: list[str] = []
        for clause in segment_record(descriptions):
            lab = self.predict_clause(clause)
            if collapse_repeats and seq and seq[-1] == lab:
                continue
            seq.append(lab)
        return seq

    def profile(self, descriptions: Iterable[str]) -> np.ndarray:
        """Return the record's category PROFILE: a length-k count vector.

        This is the representation the difference measures (Jensen-Shannon
        divergence) operate on: how a record's cataloguing effort is *distributed*
        across structural categories, independent of order. Not normalised here —
        callers normalise to a probability distribution as needed.
        """
        counts = np.zeros(self.k, dtype=np.float64)
        for clause in segment_record(descriptions):
            x = self.scaler.transform(clause_features(clause).reshape(1, -1))
            counts[int(self.model.predict(x)[0])] += 1.0
        return counts

    def cluster_feature_means(self) -> np.ndarray:
        """Return (k, n_features) array of mean (original-scale) features per
        category, for deriving human-readable descriptors."""
        centres_scaled = self.model.cluster_centers_
        return self.scaler.inverse_transform(centres_scaled)

    def describe_categories(self, top_n: int = 4) -> dict[str, list[str]]:
        """Auto-derive a human-readable descriptor per category from the features
        that most distinguish it from the corpus mean. For interpretation only —
        the authoritative labels are the neutral CAT_xx."""
        means = self.cluster_feature_means()
        grand = means.mean(axis=0)
        std = means.std(axis=0) + 1e-9
        z = (means - grand) / std
        out: dict[str, list[str]] = {}
        for i in range(self.k):
            order = np.argsort(-np.abs(z[i]))[:top_n]
            descr = [f"{self.feature_names[j]}{'+' if z[i, j] > 0 else '-'}"
                     for j in order]
            out[self.label_for(i)] = descr
        return out


# ── 5. Subject handling (complementary, optional) ──────────────────────────────
def subject_token(n_subjects: int) -> str:
    """Coarse ordinal token for subject-indexing depth (no subject content).

    Optional complementary structural signal: HOW HEAVILY a record is subject-
    indexed, never WHICH subjects. Buckets are deliberately coarse and medium-
    agnostic.
    """
    if n_subjects == 0:
        return "SUBJ_NONE"
    if n_subjects <= 3:
        return "SUBJ_FEW"
    if n_subjects <= 8:
        return "SUBJ_MANY"
    return "SUBJ_RICH"


# ── PI-validated category codebook (from the eyeball workbook) ─────────────────
# Human labels and verdicts recorded by the PI in
# practice-eyeball_nls-merged-corpus_*.xlsx (Categories tab), against the seeded
# run K=10 / RANDOM_SEED=42 on nls-merged-corpus-v2.xml. These labels are
# RUN-SCOPED: valid only while the categoriser is fitted with the same K, seed and
# corpus (clustering is deterministic, so CAT_xx ids are stable). Re-fitting with a
# different K/seed/corpus changes the ids and requires a fresh eyeball.
#
# Fields: label, captures, coherent (Yes/Partly/No), greedy (Yes/No).
VALIDATED_LABELS: dict[str, dict[str, str]] = {
    "CAT_00": {"label": "opening", "captures": "the start of a catalogue entry",
               "coherent": "Partly", "greedy": "No"},
    "CAT_01": {"label": "shotlist", "captures": "a part of a shotlist",
               "coherent": "Yes", "greedy": "No"},
    "CAT_02": {"label": "direction-to-reader", "captures": "direction to the reader to focus on one part of the object",
               "coherent": "Yes", "greedy": "No"},
    "CAT_03": {"label": "genre", "captures": "genres, separated by semi-colons",
               "coherent": "Yes", "greedy": "No"},
    "CAT_04": {"label": "location", "captures": "mostly places",
               "coherent": "Partly", "greedy": "No"},
    "CAT_05": {"label": "descriptive shotlist", "captures": "a shotlist part that includes interpretation and context",
               "coherent": "Partly", "greedy": "No"},
    "CAT_06": {"label": "people and genre (unclear)", "captures": "not clearly defined — PI unsure",
               "coherent": "No", "greedy": "No"},
    "CAT_07": {"label": "date", "captures": "date in yyyy format",
               "coherent": "Yes", "greedy": "No"},
    "CAT_08": {"label": "opening (broad)", "captures": "what seems to be the start of a catalogue entry",
               "coherent": "Yes", "greedy": "Yes"},
    "CAT_09": {"label": "closing", "captures": "what seems to be the end of a catalogue entry",
               "coherent": "Partly", "greedy": "No"},
}

# Expert agenda for the NEXT iteration — not applied automatically, because the
# PI's validation (categories + cluster exemplars) was done against the CURRENT
# categories/clusters; re-clustering now would invalidate that work.
CATEGORY_REVIEW_NOTES: list[str] = [
    "CAT_00 ~ CAT_08: both labelled 'opening' by the PI -> merge candidate (redundant split).",
    "CAT_08: flagged greedy (absorbs distinct moves).",
    "CAT_06: judged incoherent ('not sure') -> low-confidence; assignments unreliable.",
    # Run 2 (2026-06-23): K=14 re-fit TESTED AND REJECTED. Raising K did not split the "
    # greedy opening into clean moves; it raised noise (17%->29%), created new duplicate "
    # labels (shotlist x2, detail x2, name-parts x2, quotes x2), shifted greediness to two "
    # 'name parts' categories, and produced a bogus cluster (0/5 exemplar fit). K=10 RETAINED. "
    # Global K is not the fix for the greedy 'opening'; the remaining targeted option is an "
    # expert-driven MERGE of the two K=10 openings, with the greedy opening documented as a "
    # known limitation. Bonus: cross-collection practice groupings replicated at K=14 -> the "
    # cross-collection finding is robust to category granularity.",
    "RUN 2 (K=14) tested and rejected — see comment above; K=10 retained; cross-collection finding robust across K.",
    # Run 3 (2026-06-23): the CAT_00+CAT_08 MERGE was tried and then REVERSED. Merging
    # collapsed ~1/3 of the sample into one opening blob and lost the Wood+Adair maps result.
    # A targeted PI eyeball of that blob (opening-merge-eyeball_*.xlsx) showed CAT_00 and
    # CAT_08 are NOT redundant: CAT_00 = terse publication-date/stub opening (Wood plans 100%),
    # CAT_08 = full descriptive opening (Hutton 93%). So the openings are KEPT SEPARATE and
    # relabelled; only the incoherent CAT_06 is consolidated -> CAT_OTHER. See CATEGORY_REMAP.
    "RUN 3 (2026-06-23): opening merge REVERSED after eyeball — CAT_00 (stub) vs CAT_08 (descriptive) are a real distinction; only CAT_06 -> CAT_OTHER.",
]


# ── v2 expert consolidation (post-hoc; does NOT re-fit the categoriser) ────────
# Applied AFTER clustering. The KMeans fit and every clause->cluster assignment are
# UNCHANGED, so the PI's clause-level category eyeball stays valid; this layer only
# renames category LABELS.
#
# REVISION (2026-06-23, after the opening-blob eyeball): the two "opening" categories
# are NOT merged. A targeted PI eyeball of the over-merged opening cluster
# (opening-merge-eyeball_nls-merged-corpus_2026-06-23.xlsx) showed CAT_00 and CAT_08
# encode a REAL distinction, not a redundant split:
#   CAT_00 -> Wood town plans 100%  = terse "publication-date / stub" opening
#   CAT_08 -> Hutton drawings  93%  = full discursive "description" opening
# Merging them collapsed the Wood+Adair maps result (one third of the sample fell into
# a single opening blob). So the merge is REVERSED; only the genuinely-incoherent CAT_06
# is consolidated:
#   CAT_06 -> CAT_OTHER  (PI judged incoherent; isolated, not interpreted)
CATEGORY_REMAP: dict[str, str] = {
    "CAT_06": "CAT_OTHER",
}

# Fixed display/column order for the consolidated (length-10) profile vector.
CONSOLIDATED_ORDER: list[str] = [
    "CAT_00", "CAT_01", "CAT_02", "CAT_03", "CAT_04",
    "CAT_05", "CAT_07", "CAT_08", "CAT_09", "CAT_OTHER",
]

# v3 codebook: FUNCTIONAL labels from the PI's two-round category-label eyeball
# (category-label-eyeball_..._20260708 + round2-30). Object-names (opening/location/
# shotlist/closing) replaced by structural/functional names. Four categories form one
# "information/statement" family, separated by structure (plain / comma-list / prose / colon).
# Labels are cosmetic; the clustering and all numbers are unchanged.
VALIDATED_LABELS_V2: dict[str, dict[str, str]] = {
    "CAT_00":    {"label": "information (terse / stub)", "captures": "terse standalone information/stub, few commas or colons; e.g. 'Originally Published 1822', 'Actress in dancing costume'",
                  "coherent": "Yes", "greedy": "No"},
    "CAT_01":    {"label": "reference or list fragment", "captures": "reference/accession numbers, run-lengths, prices, list fragments; e.g. 'SRO No. series 119/144-148', '14 [1885], six nights', '6d. or 2s.'",
                  "coherent": "Partly", "greedy": "No"},
    "CAT_02":    {"label": "part(s) of given name(s)", "captures": "terse name-part fragments, plus some reader-directions; e.g. 'Fitzgerald.', 'Miss Wallis..', 'In two columns.'",
                  "coherent": "Partly", "greedy": "No"},
    "CAT_03":    {"label": "genre", "captures": "semicolon-separated genre/form tags; e.g. 'non-fiction; television arts'",
                  "coherent": "Yes", "greedy": "No"},
    "CAT_04":    {"label": "information (terse / stub), comma-separated", "captures": "comma-separated terse information: imprints, credits, responsibility; e.g. 'Producer, STV', 'Printed by Stafford & Co., Netherfield'",
                  "coherent": "Yes", "greedy": "No"},
    "CAT_05":    {"label": "sequential / timecoded description", "captures": "shotlist body: sequential and/or timecoded description; e.g. 'Pulls out through dock gates (2.30)'",
                  "coherent": "Yes", "greedy": "No"},
    "CAT_07":    {"label": "numeric value (date or reference)", "captures": "a bare number: a date or a reference/accession number; e.g. '27th June 1913.', '119/122'",
                  "coherent": "Partly", "greedy": "No"},
    "CAT_08":    {"label": "information (descriptive / prose)", "captures": "longer, prose-like information/description; e.g. 'The drawing is undated and unsigned.', poster/photograph descriptions",
                  "coherent": "Yes", "greedy": "Partly (broadest; longest of the information family)"},
    "CAT_09":    {"label": "information (terse / stub) after colon", "captures": "field-label 'X: Y' statement; e.g. 'Tune: Boyne water.', 'Date of event depicted: 1879', 'Caption: ...'",
                  "coherent": "Yes", "greedy": "No"},
    "CAT_OTHER": {"label": "other / unreliable", "captures": "isolated incoherent cluster (was CAT_06); predominantly moving-image bare 'non-fiction'; kept isolated, not interpreted",
                  "coherent": "No", "greedy": "No"},
}

_RAW_ORDER = [f"CAT_{i:02d}" for i in range(10)]   # raw cluster ids 0..9


def consolidate(cat_id: str) -> str:
    """Map a raw CAT_xx id to its consolidated v2 label (identity if unmapped)."""
    return CATEGORY_REMAP.get(cat_id, cat_id)


def consolidate_sequence(seq: list[str]) -> list[str]:
    """Remap a raw label sequence to v2 and re-collapse consecutive repeats."""
    out: list[str] = []
    for tok in seq:
        lab = consolidate(tok)
        if out and out[-1] == lab:
            continue
        out.append(lab)
    return out


def consolidate_profile(raw_counts) -> np.ndarray:
    """Aggregate a raw length-10 count vector (CAT_00..CAT_09 order) into the
    length-9 consolidated v2 profile (in CONSOLIDATED_ORDER)."""
    agg = {c: 0.0 for c in CONSOLIDATED_ORDER}
    for cid, n in zip(_RAW_ORDER, raw_counts):
        agg[consolidate(cid)] += float(n)
    return np.array([agg[c] for c in CONSOLIDATED_ORDER], dtype=float)


def label_v2(cat_id: str) -> str:
    """Human-readable label for a consolidated v2 category id; falls back to id."""
    return VALIDATED_LABELS_V2.get(cat_id, {}).get("label", cat_id)


def label_of(cat_id: str) -> str:
    """Human-readable label for a CAT_xx id; falls back to the id."""
    return VALIDATED_LABELS.get(cat_id, {}).get("label", cat_id)


def labelled(cat_id: str) -> str:
    """'CAT_03 (genre)' style string for display in tables/figures."""
    lab = VALIDATED_LABELS.get(cat_id, {}).get("label")
    return f"{cat_id} ({lab})" if lab else cat_id


# ── 6. Token-set helpers for sequence similarity ───────────────────────────────
def sequence_string(masked_seq: Sequence[str]) -> str:
    """Join a masked label sequence into a space-separated string for SM."""
    return " ".join(masked_seq)


def token_ngrams(masked_seq: Sequence[str], n: int = 2) -> set[str]:
    """Token n-grams of a masked sequence (default bigrams) for Jaccard/LSH."""
    toks = list(masked_seq)
    if n <= 1:
        return set(toks)
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


if __name__ == "__main__":
    # Tiny self-demonstration: identical-structure / different-content clauses
    # should land in the same category; different-structure clauses should not.
    demo = [
        "Photograph of wrecked piers after collapse of part of the first Tay Bridge.",
        "Photograph of the new Forth Bridge under construction near Queensferry.",
        "SHOTLIST: No credits. Shots of the exterior (.23); tulip beds (.29); crowd (1.02).",
        "First line reads: As I walk'd out one morning across the verdant plains.",
        "In one column with a woodcut above the title.",
        "Originally published 1682.",
    ]
    cat = StructuralCategoriser(k=4, random_seed=42).fit(demo)
    for c in demo:
        print(f"{cat.predict_clause(c):8s} | {c[:60]}")
    print("\nCategory descriptors:")
    for lab, d in cat.describe_categories().items():
        print(f"  {lab}: {', '.join(d)}")
