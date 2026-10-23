# Cataloguing practice across the collections of the National Library of Scotland: data and code

*Draft README (23 Oct 2026).*

Data and code for an article in preparation for the *Journal of Digital History*, which asks what the catalogue of the National Library of Scotland (NLS) shows about how cataloguing practice passed between collections and media, and how far a computational method can recover it.

The method splits each catalogue description into clauses, describes each clause by seventeen measures of its shape that carry no vocabulary, sorts the clauses into ten types, and groups records by their mix of clause types without knowing which collection each came from. A comparison with an off-the-shelf language model (all-mpnet-base-v2), which reads what records are about, shows that the language model groups records almost entirely by collection.

## Layout

The layout follows the *Journal of Digital History* template: the article notebook will sit in the root, images in `media/`, and code and data in `script/`.

```
nls-cataloguing-practice/
├── README.md                this file (abstract and keywords to be added)
├── requirements.txt         Python packages (to be pinned and tested on MyBinder)
├── LICENSE                  MIT (code)
├── LICENSE-DATA.md          CC0 1.0 (data and derived data), with a note on source rights
├── DATASHEET.md             datasheet for the merged corpus
├── CITATION.cff             how to cite
├── media/                   figures for the article (empty for now)
└── script/
    ├── code/                corpus building, text matching (stage 1), clause features and clustering (stage 2)
    ├── data/                the merged corpus (26,957 records, ten collections) and its build manifest
    ├── derived-data/        clause and record tables, clusters, codebook, and stage-1 outputs
    ├── parameter-sweep/     HDBSCAN sensitivity sweep at six settings, with the membership-stability table
    └── embeddings-comparison/
        ├── embeddings_baseline_v1.ipynb   comparison with all-mpnet-base-v2 on the same records
        └── derived-data/                  saved results of the run of 20 Oct 2026
```

The embeddings themselves (a 78.5 MB matrix for 26,799 records) are not included; the notebook recreates them in a few minutes on a GPU, and the saved results above are what the article reports.

## Source data and licences

The catalogue records come from the NLS Data Foundry metadata collections (https://data.nls.uk/data/metadata-collections/), published under the Public Domain Mark. The merged corpus and all derived data are dedicated to the public domain under CC0 1.0 (`LICENSE-DATA.md`). Code is under the MIT Licence (`LICENSE`).

