# Datasheet — NLS merged corpus v2 (NLSDFCat)

A brief datasheet following the approach in Alkemade, Henk, Gustavo Candela, Steven Claeyssens, et al. Datasheets for Digital Cultural Heritage Datasets. 7 July 2025. https://doi.org/10.5281/zenodo.15828222

## Motivation

The corpus (hereafter 'NLSDFCat') was assembled to support pilot work on reading cataloguing practice across heterogeneous catalogue data — that is, to study *how* catalogue records are written rather than *what* they describe. It was created for a pilot study of cataloguing practice across the collections of the National Library of Scotland, written up as an article for the *Journal of Digital History* (in preparation). It is not a representative sample of any institutional holdings, rather it is a handpicked collection of catalogue data chosen for the disparate subject and media it represents.

## Composition

26,957 catalogue records merged from ten National Library of Scotland Data Foundry collections, each record tagged with its source collection and a record identifier:

| prefix | collection | records |
|---|---|---|
| `mia` | Moving Image Archive (2025) | 23,491 |
| `eb` | English broadside ballads | 2,438 |
| `hd` | Hutton drawings | 537 |
| `tp` | Theatre posters | 190 |
| `tb` | Tay Bridge enquiry (photographs) | 91 |
| `sp` | Soviet posters | 72 |
| `jw` | John Wood town plans | 62 |
| `fb` | Forth Bridge illustrations (1886–1887) | 40 |
| `jk` | John Kirk survey | 23 |
| `ja` | John Adair county maps | 13 |

Each record is a Dublin Core description as published by the Data Foundry — typically title, one or more description fields, and where present creator, date, subject, language, publisher, format, extent, identifier and rights statements — plus the two fields added by the merge (`mc:collection`, `mc:record_id`). The article's analysis reads only the descriptive content; the other fields are retained untouched. The moving-image archive dominates by count; analyses that sample per collection should be read with that imbalance in mind. The collections are content and medium disparate by design (moving images, ballads, posters, maps, drawings, bridge photographs and illustrations).

The material described spans the late seventeenth century (county maps, early ballads) to the 2010s (broadcast television deposits). Description languages are English and Scottish Gaelic, including bilingual records; note that one analysis feature (`function_word_ratio`) is English-dependent, as the article states.

## Collection process

The ten collections were selected and downloaded by the author from the [NLS Data Foundry Metadata Collection](https://data.nls.uk/data/metadata-collections/), where each collection is published as open data. `build_corpus_v1.py` then performs a discovery-based merge of every collection present in the data directory into a single namespace-aware XML file, tagging each record by source collection and identifier. Build metadata is in `nls-merged-corpus-v2_manifest.txt`.

## Preprocessing / cleaning

Records were de-duplicated (final count 26,957). Data Foundry boilerplate — standard rights and provenance text repeated across records — is retained in the corpus file and stripped during analysis by the pipeline (`strip_boilerplate` in `structural_masking_v1.py`), because it is the one piece of surface text shared across collections and manufactures spurious near-duplicate matches. Users of the corpus for text-similarity work should apply the same or an equivalent strip. Structural features (see the article) are computed by `structural_masking_v1.py` and are not stored in the corpus itself.

## Personal and sensitive data

Records name individuals — creators, performers, and people depicted in or associated with collection items. All text is National Library of Scotland catalogue description published under the Public Domain Mark; the corpus adds no personal data beyond what the Library has published, and none of it is sensitive in the datasheet sense.

## Uses

Suitable for:

- studying the structure and form of catalogue description
- method development for content-agnostic / structural text analysis
- reproducing the analysis in the accompanying article

Not suitable for:

- treating as a representative or complete record of NLS holdings
- drawing conclusions about object contents from description alone
- treating the emergent categories as canonical structures of catalogue records in general: all ten collections come from one institution, and the categories described may not transfer to catalogues with different conventions

## Distribution and licence

Released under **CC0 1.0**. The underlying catalogue records are NLS Data Foundry material
published under the **Public Domain Mark**; this repository adds no further restriction. See
`LICENSE-DATA.md`.

## Maintenance

The corpus is a frozen snapshot, built in June 2026 for the analysis, and will not track subsequent Data Foundry updates. Corrections and queries via the repository's issue tracker.

---

## Derived-data note (brief)

The `derived-data/` and `parameter-sweep/derived-data/` folders hold the outputs the article reports, all generated from the corpus by the code here and released CC0:

- **Stage-1 full-text baseline** — `communities_*`, `tfidf-communities_*`, `centrality_*`, `bridge-scan-subthreshold_*`, `gateway-nodes_*`, `no-description-records_*`, `disagreements-sm-vs-tfidf_*`: the similarity network and the within-collection communities that motivate the structural mask.
- **Stage-2 structural** — `clause-features_*` (one row per clause with its seventeen features and category), `structure-features_*` (per-record sequences and category profiles), `structure-codebook_*` (the ten categories with their functional labels and coherence judgements), `practice-clusters_*` and `practice-sample_*` (the collection-blind HDBSCAN practice clusters and a read sample). The run reported in the article is `20260708-152108`.
- **Sweep** — `parameter-sweep/derived-data/`: the same Stage-2 outputs at six HDBSCAN settings plus `membership-stability_*`, supporting the robustness check.

Field-level column meanings for the codebook and cluster tables are described in the article and in `parameter-sweep/README.md`.
