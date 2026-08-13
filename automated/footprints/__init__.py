"""Automated footprint building from point locations and patch embeddings.

The CLI is ../build_footprints.py; this package holds the stages it drives.

    backends    streaming embedding access (parquet or DuckDB)
    geometry    reprojection, grid inference, patch squares
    labels      positives from points, sampled negatives
    modeling    classifier, out-of-fold probabilities, threshold, curves
    inference   full-AOI inference, patches to merged polygons
    pipeline    one train-infer-filter pass, mining and admission loops
    assessment  reference-free checks that a run went wrong
    reporting   the config and stats files a run writes
    util        progress logging and the run-wide warning list
"""
