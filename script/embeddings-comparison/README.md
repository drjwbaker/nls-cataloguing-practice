# Comparison with an off-the-shelf language model

`embeddings_baseline_v1.ipynb` groups the same catalogue records with a sentence-embedding model (`sentence-transformers/all-mpnet-base-v2`, seed 42) and compares the groups with those of the masked method, using the masked method's own record sample (901 records) and HDBSCAN settings (minimum group size 15, minimum samples 5).

Inputs (paths relative to this folder): `../data/nls-merged-corpus-v2.xml`; `../code/structural_masking_v1.py` (for `strip_boilerplate`, so the text is prepared as in the masked method); `../derived-data/practice-sample_nls-merged-corpus_20260708-152108.tsv` and `../derived-data/structure-features_nls-merged-corpus_20260708-152108.tsv`.

Saved results of the run of 20 Oct 2026 (`derived-data/`): `embeddings-summary_…json` (every figure reported), `embeddings-vs-practice_…tsv` (each of the 901 records with its embedding group and practice group), `embeddings-leiden_…tsv` (whole-corpus grouping), and the record order of the embedding matrix (`…ids.tsv`). The matrix itself (78.5 MB) is not included; Step 4 recreates it, in a few minutes with a GPU.

Headline result: of each record's ten closest records, 97% share its collection by embedding, against 66% for the masked method (13% by chance); the embeddings form one group spanning collections, the masked method nine.

Python packages: `requirements.txt` in this folder (minimum versions used for the run).
