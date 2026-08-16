"""Representations module exports for ARES V2 Phase 3."""

from ares.representations.collector import (
    RepresentationCollector,
    RepresentationRecord,
    create_collector_from_config,
)
from ares.representations.storage import (
    save_representations,
    load_representations,
    save_representations_hdf5,
    load_representations_hdf5,
    save_representations_parquet,
    load_representations_parquet,
    save_collection_metadata,
    load_collection_metadata,
)
from ares.representations.analysis import (
    LayerStatistics,
    SeparabilityResult,
    ClusteringResult,
    CorrectnessCorrelationResult,
    compute_layer_statistics,
    compute_domain_separability,
    run_pca,
    run_tsne,
    cluster_representations,
    analyze_correctness_correlation,
    compute_all_separability,
    compute_all_clustering,
    compute_all_correctness_correlations,
    generate_analysis_report,
)

__all__ = [
    # Collector
    "RepresentationCollector",
    "RepresentationRecord",
    "create_collector_from_config",
    # Storage
    "save_representations",
    "load_representations",
    "save_representations_hdf5",
    "load_representations_hdf5",
    "save_representations_parquet",
    "load_representations_parquet",
    "save_collection_metadata",
    "load_collection_metadata",
    # Analysis
    "LayerStatistics",
    "SeparabilityResult",
    "ClusteringResult",
    "CorrectnessCorrelationResult",
    "compute_layer_statistics",
    "compute_domain_separability",
    "run_pca",
    "run_tsne",
    "cluster_representations",
    "analyze_correctness_correlation",
    "compute_all_separability",
    "compute_all_clustering",
    "compute_all_correctness_correlations",
    "generate_analysis_report",
]