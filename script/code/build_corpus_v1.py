#!/usr/bin/env python3
"""
build_corpus_v1.py
==================
Rebuild the NLS merged corpus from *all* collection directories currently present
under ``data/`` (excluding ``nls-merged-corpus/`` itself).

This is the v1 corpus builder for the masked cross-collection analysis pipeline. It
supersedes ``split_nls_datasets.py`` (which hard-coded seven collections) by
*discovering* collections dynamically and assigning short prefixes systematically.

Design decisions
----------------
1. **Discovery, not hard-coding.** Every ``nls-*`` directory under ``data/`` is
   inspected for ``*-DC.xml`` files (Dublin Core). Adding a new collection to
   ``data/`` is therefore sufficient for it to enter the corpus on the next run —
   no code change required.

2. **Stable prefixes.** Prefixes for the seven original collections are carried
   over verbatim (eb, hd, jk, jw, mia, tb, tp) so that record IDs remain
   comparable with the v2 (unmasked) pipeline outputs. New collections receive a
   deterministic, human-readable prefix derived from their directory name, with a
   fixed mapping for the four named new collections.

3. **Content-level de-duplication.** Some directories in ``data/`` contain
   *identical* DC exports under different directory names (a known data-staging
   issue — see the manifest output). To avoid double-counting the same records
   under two prefixes, each DC file's content is hashed; the first directory to
   claim a given content hash wins, and later duplicates are skipped and reported.

4. **Two Dublin Core namespaces.** MIA records use ``dcterms:`` (DC Terms);
   the digitised collections use ``dc:`` (DC Elements 1.1). Both are handled.

5. **Non-destructive.** Existing files in ``data/nls-merged-corpus/`` are never
   overwritten. The new merged XML and manifest are written with ``-v2`` names;
   if a target already exists the script aborts with a clear message unless
   ``--force`` is given (which writes a timestamped sibling instead of
   overwriting).

Outputs
-------
- ``data/nls-merged-corpus/nls-merged-corpus-v2.xml``           — merged DC XML
- ``data/nls-merged-corpus/nls-merged-corpus-v2_manifest.txt``  — collection → prefix → count

Each record in the merged XML is an ``<rdf:Description>`` element carrying an added
``mc:collection`` and ``mc:record_id`` attribute-free child element pair (namespace
``mc`` = "merged corpus") so that the masked pairwise notebook can read collection
identity and a stable record id directly from the XML without relying on filenames.

Usage
-----
    python3 build_corpus_v1.py
    python3 build_corpus_v1.py --data-dir /path/to/data
    python3 build_corpus_v1.py --force        # write timestamped output if target exists

Author: James Baker / pipeline rebuild, 2026.
"""

import argparse
import datetime
import hashlib
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict

# ── namespaces ────────────────────────────────────────────────────────────────
RDF_NS     = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DC_NS      = "http://purl.org/dc/elements/1.1/"
DCTERMS_NS = "http://purl.org/dc/terms/"
MC_NS      = "http://purl.org/nls-merged-corpus/1.0/"   # local provenance namespace

NS = {"rdf": RDF_NS, "dc": DC_NS, "dcterms": DCTERMS_NS, "mc": MC_NS}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


# ── prefix assignment ─────────────────────────────────────────────────────────
# Carry over the seven original prefixes verbatim, keyed by directory name.
# New collections get an explicit, stable prefix. Any further collection added in
# future falls back to a deterministic prefix derived from the directory name.
KNOWN_PREFIXES = {
    "nls-moving-image-archive-2025":            "mia",
    "nls-english-ballads":                      "eb",
    "nls-hutton-drawings":                      "hd",
    "nls-john-kirk-survey":                     "jk",
    "nls-john-wood-town-plans":                 "jw",
    "nls-tay-bridge-enquiry":                   "tb",
    "nls-theatre-posters":                      "tp",
    # ── new collections (this rebuild) ──
    "nls-forth-bridge-illustrations-1886-1887": "fb",
    "nls-jacobite-prints-and-broadsides":       "jp",
    "nls-john-adair-county-maps":               "ja",
    "nls-soviet-posters":                       "sp",
}


# Directories to exclude from the build.
# The content-hash deduplication logic (see build() below) will detect and skip
# any DC file whose content is identical to a file already loaded from another
# collection, logging it in the manifest. No hard exclusions are currently set.
EXCLUDE_DIRS = set()


def derive_prefix(dirname, taken):
    """Deterministic fallback prefix from a directory name (e.g. for unforeseen
    collections). Uses the initials of the hyphen-separated, nls-stripped tokens,
    then disambiguates against already-taken prefixes."""
    tokens = [t for t in dirname.replace("nls-", "").split("-") if t and not t.isdigit()]
    base = "".join(t[0] for t in tokens[:2]).lower() or "xx"
    cand = base
    i = 1
    while cand in taken:
        i += 1
        cand = f"{base}{i}"
    return cand


# ── helpers ───────────────────────────────────────────────────────────────────
def find_dc_files(coll_dir):
    """Return DC XML files directly inside a collection directory (case-insensitive
    match on '-DC.xml'). MARC files and nested attic/ directories are ignored."""
    out = []
    for fname in sorted(os.listdir(coll_dir)):
        fp = os.path.join(coll_dir, fname)
        if not os.path.isfile(fp):
            continue
        if re.search(r"-dc\.xml$", fname, flags=re.IGNORECASE):
            out.append(fp)
    return out


def first_numeric_identifier(desc):
    """First numeric dc:/dcterms: identifier value, else None."""
    for prefix in ("dc", "dcterms"):
        for el in desc.findall(f"{prefix}:identifier", NS):
            text = (el.text or "").strip()
            if text.isdigit():
                return text
    return None


def first_identifier_any(desc):
    """Any first identifier (numeric preferred, else filmRef-style), for MIA."""
    num = first_numeric_identifier(desc)
    if num:
        return num
    for prefix in ("dcterms", "dc"):
        for el in desc.findall(f"{prefix}:identifier", NS):
            text = (el.text or "").strip()
            if text:
                # e.g. "(filmRef)0001" -> "0001"
                m = re.search(r"([A-Za-z0-9]+)$", text)
                if m:
                    return m.group(1)
    return None


def content_hash(filepath):
    """SHA-1 of the *parsed record content* (identifiers + descriptions) so that
    two files that differ only in whitespace/encoding but contain the same records
    are recognised as duplicates."""
    try:
        root = ET.parse(filepath).getroot()
    except ET.ParseError:
        # fall back to raw byte hash
        with open(filepath, "rb") as fh:
            return hashlib.sha1(fh.read()).hexdigest()
    h = hashlib.sha1()
    for desc in root.findall("rdf:Description", NS):
        ident = first_identifier_any(desc) or ""
        descs = []
        for prefix in ("dc", "dcterms"):
            descs.extend((el.text or "").strip() for el in desc.findall(f"{prefix}:description", NS))
        h.update(ident.encode("utf-8"))
        h.update("".join(descs).encode("utf-8"))
    return h.hexdigest()


# ── MIA: records come from the per-record txt files already prepared ──────────
def load_mia_descriptions(data_dir):
    """MIA DC is distributed as four XML files; load every rdf:Description from
    them. Returns a list of (record_id, ET.Element)."""
    mia_dir = os.path.join(data_dir, "nls-moving-image-archive-2025")
    xml_files = sorted(
        os.path.join(mia_dir, f)
        for f in os.listdir(mia_dir)
        if re.match(r"2025_mia_dc_\d+\.xml$", f)
    )
    out = []
    seen = set()
    for xf in xml_files:
        root = ET.parse(xf).getroot()
        for desc in root.findall("rdf:Description", NS):
            rid = first_identifier_any(desc)
            if rid is None:
                continue
            if rid in seen:
                continue
            seen.add(rid)
            out.append((rid, desc))
    return out, xml_files


# ── main build ────────────────────────────────────────────────────────────────
def build(data_dir, out_xml, out_manifest):
    merged_dir = os.path.join(data_dir, "nls-merged-corpus")

    # Discover collection directories (exclude the merged-corpus dir itself)
    coll_dirs = sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("nls-")
        and d != "nls-merged-corpus"
        and d not in EXCLUDE_DIRS
        and os.path.isdir(os.path.join(data_dir, d))
    )

    excluded_present = sorted(
        d for d in EXCLUDE_DIRS
        if os.path.isdir(os.path.join(data_dir, d))
    )
    for d in excluded_present:
        print(f"  [---]  {d:46s}  EXCLUDED (see EXCLUDE_DIRS note)")

    taken_prefixes = set(KNOWN_PREFIXES.values())
    seen_hashes = {}          # content_hash -> (dirname, filepath)  (first wins)
    manifest_rows = []        # (dirname, prefix, n_records, note)

    # Root of the merged document
    rdf_root = ET.Element(f"{{{RDF_NS}}}RDF")

    total_records = 0

    for dirname in coll_dirs:
        coll_path = os.path.join(data_dir, dirname)
        prefix = KNOWN_PREFIXES.get(dirname) or derive_prefix(dirname, taken_prefixes)
        taken_prefixes.add(prefix)

        # ── MIA handled specially (multi-file DC export) ──
        if dirname == "nls-moving-image-archive-2025":
            records, mia_files = load_mia_descriptions(data_dir)
            n = 0
            for rid, desc in records:
                _append_record(rdf_root, desc, prefix, rid)
                n += 1
            manifest_rows.append((dirname, prefix, n, f"{len(mia_files)} DC files"))
            total_records += n
            print(f"  [{prefix:4s}] {dirname:46s} {n:>6,} records")
            continue

        # ── digitised collections (one or more -DC.xml files) ──
        dc_files = find_dc_files(coll_path)
        if not dc_files:
            manifest_rows.append((dirname, prefix, 0, "NO DC FILE FOUND — skipped"))
            print(f"  [{prefix:4s}] {dirname:46s} {'0':>6} records   (no DC file)")
            continue

        n_dir = 0
        notes = []
        for dc_file in dc_files:
            chash = content_hash(dc_file)
            if chash in seen_hashes:
                dup_dir, _ = seen_hashes[chash]
                notes.append(
                    f"DUPLICATE of '{dup_dir}' ({os.path.basename(dc_file)}) — skipped"
                )
                print(f"  [{prefix:4s}] {dirname:46s} skipped {os.path.basename(dc_file)} "
                      f"(content-identical to '{dup_dir}')")
                continue
            seen_hashes[chash] = (dirname, dc_file)

            root = ET.parse(dc_file).getroot()
            for desc in root.findall("rdf:Description", NS):
                rid = first_numeric_identifier(desc) or first_identifier_any(desc)
                if rid is None:
                    continue
                _append_record(rdf_root, desc, prefix, rid)
                n_dir += 1

        note = "; ".join(notes) if notes else ""
        if n_dir > 0:
            manifest_rows.append((dirname, prefix, n_dir, note))
            print(f"  [{prefix:4s}] {dirname:46s} {n_dir:>6,} records"
                  + (f"   ({note})" if note else ""))
        else:
            manifest_rows.append((dirname, prefix, 0, note or "all files duplicate/empty"))
        total_records += n_dir

    # ── write merged XML ──────────────────────────────────────────────────────
    tree = ET.ElementTree(rdf_root)
    ET.indent(tree, space="  ")
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)

    # ── write manifest ────────────────────────────────────────────────────────
    with open(out_manifest, "w", encoding="utf-8") as fh:
        fh.write("# NLS merged corpus v2 — build manifest\n")
        fh.write(f"# Built: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"# Builder: build_corpus_v1.py\n")
        fh.write(f"# Data dir: {os.path.abspath(data_dir)}\n")
        fh.write("#\n")
        fh.write("# collection_directory\tprefix\trecord_count\tnote\n")
        for dirname, prefix, n, note in manifest_rows:
            fh.write(f"{dirname}\t{prefix}\t{n}\t{note}\n")
        for d in excluded_present:
            fh.write(f"{d}\t-\t0\tEXCLUDED (see EXCLUDE_DIRS)\n")
        fh.write("#\n")
        fh.write(f"# TOTAL\t-\t{total_records}\t-\n")

    print("-" * 72)
    print(f"Total records written : {total_records:,}")
    print(f"Merged XML            : {os.path.abspath(out_xml)}")
    print(f"Manifest              : {os.path.abspath(out_manifest)}")
    return total_records


def _append_record(rdf_root, desc, prefix, rid):
    """Clone a source rdf:Description into the merged root, adding provenance
    children mc:collection and mc:record_id."""
    # Deep-copy the element so we do not mutate the source tree.
    new = ET.fromstring(ET.tostring(desc, encoding="unicode"))
    coll_el = ET.SubElement(new, f"{{{MC_NS}}}collection")
    coll_el.text = prefix
    rid_el = ET.SubElement(new, f"{{{MC_NS}}}record_id")
    rid_el.text = f"{prefix}_{rid}"
    rdf_root.append(new)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "data"))

    ap = argparse.ArgumentParser(description="Build the NLS merged corpus v2 (all collections).")
    ap.add_argument("--data-dir", default=default_data_dir,
                    help=f"Root data directory (default: {default_data_dir})")
    ap.add_argument("--force", action="store_true",
                    help="If the v2 output already exists, write a timestamped sibling "
                         "instead of aborting (never overwrites).")
    args = ap.parse_args()

    merged_dir = os.path.join(args.data_dir, "nls-merged-corpus")
    if not os.path.isdir(merged_dir):
        os.makedirs(merged_dir, exist_ok=True)

    out_xml      = os.path.join(merged_dir, "nls-merged-corpus-v2.xml")
    out_manifest = os.path.join(merged_dir, "nls-merged-corpus-v2_manifest.txt")

    # Non-destructive guard
    if os.path.exists(out_xml) or os.path.exists(out_manifest):
        if not args.force:
            print("Target output already exists — refusing to overwrite:")
            if os.path.exists(out_xml):
                print(f"  {out_xml}")
            if os.path.exists(out_manifest):
                print(f"  {out_manifest}")
            print("Re-run with --force to write a timestamped sibling instead.")
            sys.exit(1)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_xml      = os.path.join(merged_dir, f"nls-merged-corpus-v2_{stamp}.xml")
        out_manifest = os.path.join(merged_dir, f"nls-merged-corpus-v2_manifest_{stamp}.txt")
        print(f"--force: writing timestamped output ({stamp}).")

    print(f"Data directory : {os.path.abspath(args.data_dir)}")
    print(f"Discovering collections under data/ ...")
    print("-" * 72)
    build(args.data_dir, out_xml, out_manifest)


if __name__ == "__main__":
    main()
