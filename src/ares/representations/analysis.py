"""Analysis utilities for representation separability and quality assessment.

Provides PCA, t-SNE, LDA, clustering, and correlation analysis for collected representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import warnings

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from ares.representations.collector import RepresentationRecord


@dataclass
class LayerStatistics:
    """Statistics for a single layer's representations."""
    layer: int
    n_samples: int
    hidden_dim: int
    mean_norm: float
    std_norm: float
    mean_representation: np.ndarray
    std_representation: np.ndarray


@dataclass
class SeparabilityResult:
    """Results of domain separability analysis for a layer."""
    layer: int
    lda_accuracy: float
    silhouette_score: float
    domain_centroids: Dict[str, np.ndarray]
    domain_counts: Dict[str, int]


@dataclass
class ClusteringResult:
    """Results of K-means clustering on representations."""
    layer: int
    n_clusters: int
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    inertia: float
    domain_purity: Dict[int, Dict[str, float]]  # cluster -> domain -> purity


@dataclass
class CorrectnessCorrelationResult:
    """Correlation between representation features and correctness."""
    layer: int
    norm_correlation: float
    norm_pvalue: float
    entropy_correlation: float
    entropy_pvalue: float
    margin_correlation: float
    margin_pvalue: float
    confidence_correlation: float
    confidence_pvalue: float


def _group_records_by_layer(records: List[RepresentationRecord]) -> Dict[int, List[RepresentationRecord]]:
    """Group records by layer index."""
    grouped = {}
    for r in records:
        grouped.setdefault(r.layer, []).append(r)
    return grouped


def _group_records_by_domain(records: List[RepresentationRecord]) -> Dict[str, List[RepresentationRecord]]:
    """Group records by domain."""
    grouped = {}
    for r in records:
        grouped.setdefault(r.domain, []).append(r)
    return grouped


def compute_layer_statistics(records: List[RepresentationRecord]) -> Dict[int, LayerStatistics]:
    """Compute basic statistics for representations at each layer.

    Args:
        records: List of RepresentationRecord objects

    Returns:
        Dict mapping layer -> LayerStatistics
    """
    grouped = _group_records_by_layer(records)
    stats = {}

    for layer, layer_records in grouped.items():
        reps = np.stack([r.representation for r in layer_records], axis=0)
        norms = np.linalg.norm(reps, axis=1)

        stats[layer] = LayerStatistics(
            layer=layer,
            n_samples=len(layer_records),
            hidden_dim=reps.shape[1],
            mean_norm=float(np.mean(norms)),
            std_norm=float(np.std(norms)),
            mean_representation=np.mean(reps, axis=0),
            std_representation=np.std(reps, axis=0),
        )

    return stats


def compute_domain_separability(
    records: List[RepresentationRecord],
    layer: int,
    n_components: Optional[int] = None,
) -> SeparabilityResult:
    """Compute domain separability metrics for a specific layer.

    Uses Linear Discriminant Analysis (LDA) for classification accuracy
    and silhouette score for cluster separation quality.

    Args:
        records: List of RepresentationRecord objects
        layer: Target layer index
        n_components: Number of LDA components (default: n_domains - 1)

    Returns:
        SeparabilityResult with accuracy and silhouette scores
    """
    layer_records = [r for r in records if r.layer == layer]
    if not layer_records:
        raise ValueError(f"No records found for layer {layer}")

    # Prepare data
    X = np.stack([r.representation for r in layer_records], axis=0)
    domains = np.array([r.domain for r in layer_records])
    unique_domains = np.unique(domains)

    if len(unique_domains) < 2:
        raise ValueError("Need at least 2 domains for separability analysis")

    # Domain centroids
    domain_centroids = {}
    domain_counts = {}
    for domain in unique_domains:
        mask = domains == domain
        domain_centroids[domain] = np.mean(X[mask], axis=0)
        domain_counts[domain] = int(np.sum(mask))

    # LDA classification (leave-one-out or train/test split)
    # Use simple train/test split for speed
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, domains, test_size=0.3, random_state=42, stratify=domains
    )

    lda = LinearDiscriminantAnalysis(
        n_components=min(n_components or len(unique_domains) - 1, len(unique_domains) - 1)
    )
    lda.fit(X_train, y_train)
    lda_accuracy = float(lda.score(X_test, y_test))

    # Silhouette score
    sil_score = float(silhouette_score(X, domains, metric="cosine"))

    return SeparabilityResult(
        layer=layer,
        lda_accuracy=lda_accuracy,
        silhouette_score=sil_score,
        domain_centroids=domain_centroids,
        domain_counts=domain_counts,
    )


def run_pca(
    records: List[RepresentationRecord],
    layer: int,
    n_components: int = 50,
    whiten: bool = False,
) -> Tuple[np.ndarray, PCA]:
    """Run PCA on representations for a specific layer.

    Args:
        records: List of RepresentationRecord objects
        layer: Target layer index
        n_components: Number of principal components
        whiten: Whether to whiten the data

    Returns:
        Tuple of (transformed_data, fitted_PCA_object)
    """
    layer_records = [r for r in records if r.layer == layer]
    if not layer_records:
        raise ValueError(f"No records found for layer {layer}")

    X = np.stack([r.representation for r in layer_records], axis=0)

    actual_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=actual_components, whiten=whiten, random_state=42)
    X_transformed = pca.fit_transform(X)

    return X_transformed, pca


def run_tsne(
    records: List[RepresentationRecord],
    layer: int,
    n_components: int = 2,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Run t-SNE on representations for visualization.

    Args:
        records: List of RepresentationRecord objects
        layer: Target layer index
        n_components: Output dimensions (2 or 3)
        perplexity: t-SNE perplexity
        max_iter: Maximum iterations
        random_state: Random seed

    Returns:
        Transformed data of shape (n_samples, n_components)
    """
    layer_records = [r for r in records if r.layer == layer]
    if not layer_records:
        raise ValueError(f"No records found for layer {layer}")

    X = np.stack([r.representation for r in layer_records], axis=0)

    # Reduce dimensionality first if too high
    if X.shape[1] > 50:
        X, _ = run_pca(records, layer, n_components=50)

    n_samples = X.shape[0]
    actual_perplexity = min(perplexity, max(1, n_samples - 1))

    tsne = TSNE(
        n_components=n_components,
        perplexity=actual_perplexity,
        max_iter=max_iter,
        random_state=random_state,
        init="pca",
    )
    X_tsne = tsne.fit_transform(X)

    return X_tsne


def cluster_representations(
    records: List[RepresentationRecord],
    layer: int,
    n_clusters: int = 4,
    random_state: int = 42,
) -> ClusteringResult:
    """Run K-means clustering on representations and compute domain purity.

    Args:
        records: List of RepresentationRecord objects
        layer: Target layer index
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        ClusteringResult with labels, centers, and domain purity per cluster
    """
    layer_records = [r for r in records if r.layer == layer]
    if not layer_records:
        raise ValueError(f"No records found for layer {layer}")

    X = np.stack([r.representation for r in layer_records], axis=0)
    domains = np.array([r.domain for r in layer_records])
    unique_domains = np.unique(domains)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    # Compute domain purity per cluster
    domain_purity = {}
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if np.sum(mask) == 0:
            domain_purity[cluster_id] = {}
            continue

        cluster_domains = domains[mask]
        unique, counts = np.unique(cluster_domains, return_counts=True)
        total = np.sum(counts)
        domain_purity[cluster_id] = {
            domain: count / total for domain, count in zip(unique, counts)
        }

    return ClusteringResult(
        layer=layer,
        n_clusters=n_clusters,
        cluster_labels=labels,
        cluster_centers=kmeans.cluster_centers_,
        inertia=float(kmeans.inertia_),
        domain_purity=domain_purity,
    )


def analyze_correctness_correlation(
    records: List[RepresentationRecord],
    layer: int,
) -> Optional[CorrectnessCorrelationResult]:
    """Analyze correlation between representation features and correctness.

    Args:
        records: List of RepresentationRecord objects
        layer: Target layer index

    Returns:
        CorrectnessCorrelationResult or None if no correctness labels
    """
    layer_records = [r for r in records if r.layer == layer and r.correctness is not None]
    if len(layer_records) < 10:
        warnings.warn(f"Insufficient correctness-labeled samples for layer {layer}: {len(layer_records)}")
        return None

    X = np.stack([r.representation for r in layer_records], axis=0)
    norms = np.linalg.norm(X, axis=1)
    entropy = np.array([r.entropy for r in layer_records])
    margin = np.array([r.margin for r in layer_records])
    confidence = np.array([r.confidence for r in layer_records])
    correctness = np.array([float(r.correctness) for r in layer_records])

    results = {}
    for name, feature in [("norm", norms), ("entropy", entropy),
                          ("margin", margin), ("confidence", confidence)]:
        # Point-biserial correlation (continuous vs binary)
        corr, pval = stats.pointbiserialr(correctness, feature)
        results[f"{name}_correlation"] = float(corr) if not np.isnan(corr) else 0.0
        results[f"{name}_pvalue"] = float(pval) if not np.isnan(pval) else 1.0

    return CorrectnessCorrelationResult(
        layer=layer,
        norm_correlation=results["norm_correlation"],
        norm_pvalue=results["norm_pvalue"],
        entropy_correlation=results["entropy_correlation"],
        entropy_pvalue=results["entropy_pvalue"],
        margin_correlation=results["margin_correlation"],
        margin_pvalue=results["margin_pvalue"],
        confidence_correlation=results["confidence_correlation"],
        confidence_pvalue=results["confidence_pvalue"],
    )


def compute_all_separability(
    records: List[RepresentationRecord],
    layers: Optional[List[int]] = None,
) -> Dict[int, SeparabilityResult]:
    """Compute separability for all layers."""
    if layers is None:
        layers = sorted(set(r.layer for r in records))

    results = {}
    for layer in layers:
        try:
            results[layer] = compute_domain_separability(records, layer)
        except Exception as e:
            warnings.warn(f"Separability analysis failed for layer {layer}: {e}")
    return results


def compute_all_clustering(
    records: List[RepresentationRecord],
    layers: Optional[List[int]] = None,
    n_clusters: int = 4,
) -> Dict[int, ClusteringResult]:
    """Compute clustering for all layers."""
    if layers is None:
        layers = sorted(set(r.layer for r in records))

    results = {}
    for layer in layers:
        try:
            results[layer] = cluster_representations(records, layer, n_clusters)
        except Exception as e:
            warnings.warn(f"Clustering failed for layer {layer}: {e}")
    return results


def compute_all_correctness_correlations(
    records: List[RepresentationRecord],
    layers: Optional[List[int]] = None,
) -> Dict[int, CorrectnessCorrelationResult]:
    """Compute correctness correlations for all layers."""
    if layers is None:
        layers = sorted(set(r.layer for r in records))

    results = {}
    for layer in layers:
        result = analyze_correctness_correlation(records, layer)
        if result is not None:
            results[layer] = result
    return results


def generate_analysis_report(
    records: List[RepresentationRecord],
    output_path: Union[str, Path],
    layers: Optional[List[int]] = None,
    n_clusters: int = 4,
    include_pca: bool = True,
    include_tsne: bool = False,
) -> Path:
    """Generate comprehensive analysis report as Markdown with embedded JSON.

    Args:
        records: List of RepresentationRecord objects
        output_path: Path to save report
        layers: Layers to analyze (default: all)
        n_clusters: Number of clusters for K-means
        include_pca: Whether to run PCA
        include_tsne: Whether to run t-SNE (slow)

    Returns:
        Path to saved report
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if layers is None:
        layers = sorted(set(r.layer for r in records))

    # Compute all analyses
    layer_stats = compute_layer_statistics(records)
    separability = compute_all_separability(records, layers)
    clustering = compute_all_clustering(records, layers, n_clusters)
    correctness_corr = compute_all_correctness_correlations(records, layers)

    # PCA for first layer if requested
    pca_results = {}
    if include_pca and layers:
        try:
            pca_data, pca_model = run_pca(records, layers[0])
            pca_results = {
                "layer": layers[0],
                "explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
                "cumulative_variance": np.cumsum(pca_model.explained_variance_ratio_).tolist(),
                "n_components": pca_model.n_components_,
            }
        except Exception as e:
            warnings.warn(f"PCA failed: {e}")

    # t-SNE for first layer if requested
    tsne_results = {}
    if include_tsne and layers:
        try:
            tsne_data = run_tsne(records, layers[0])
            tsne_results = {
                "layer": layers[0],
                "data": tsne_data.tolist(),
                "domains": [r.domain for r in records if r.layer == layers[0]],
            }
        except Exception as e:
            warnings.warn(f"t-SNE failed: {e}")

    # Build report
    report = {
        "summary": {
            "total_records": len(records),
            "unique_layers": len(layers),
            "layers": layers,
            "domains": sorted(set(r.domain for r in records)),
            "samples_per_domain": {
                domain: sum(1 for r in records if r.domain == domain)
                for domain in sorted(set(r.domain for r in records))
            },
        },
        "layer_statistics": {
            str(layer): {
                "n_samples": stats.n_samples,
                "hidden_dim": stats.hidden_dim,
                "mean_norm": stats.mean_norm,
                "std_norm": stats.std_norm,
            }
            for layer, stats in layer_stats.items()
        },
        "separability": {
            str(layer): {
                "lda_accuracy": res.lda_accuracy,
                "silhouette_score": res.silhouette_score,
                "domain_counts": res.domain_counts,
            }
            for layer, res in separability.items()
        },
        "clustering": {
            str(layer): {
                "n_clusters": res.n_clusters,
                "inertia": res.inertia,
                "domain_purity": res.domain_purity,
            }
            for layer, res in clustering.items()
        },
        "correctness_correlation": {
            str(layer): {
                "norm_correlation": res.norm_correlation,
                "norm_pvalue": res.norm_pvalue,
                "entropy_correlation": res.entropy_correlation,
                "entropy_pvalue": res.entropy_pvalue,
                "margin_correlation": res.margin_correlation,
                "margin_pvalue": res.margin_pvalue,
                "confidence_correlation": res.confidence_correlation,
                "confidence_pvalue": res.confidence_pvalue,
            }
            for layer, res in correctness_corr.items()
        },
        "pca": pca_results,
        "tsne": tsne_results,
    }

    # Save as JSON for programmatic access
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Generate Markdown report
    md_lines = [
        "# Representation Analysis Report",
        "",
        f"**Total Records:** {report['summary']['total_records']}",
        f"**Layers Analyzed:** {report['summary']['unique_layers']} ({', '.join(map(str, layers))})",
        f"**Domains:** {', '.join(report['summary']['domains'])}",
        "",
        "## Layer Statistics",
        "",
        "| Layer | Samples | Hidden Dim | Mean Norm | Std Norm |",
        "|-------|---------|------------|-----------|----------|",
    ]

    for layer, stats in layer_stats.items():
        md_lines.append(
            f"| {layer} | {stats.n_samples} | {stats.hidden_dim} | "
            f"{stats.mean_norm:.4f} | {stats.std_norm:.4f} |"
        )

    md_lines.extend([
        "",
        "## Domain Separability",
        "",
        "| Layer | LDA Accuracy | Silhouette Score |",
        "|-------|--------------|------------------|",
    ])

    for layer, res in separability.items():
        md_lines.append(
            f"| {layer} | {res.lda_accuracy:.4f} | {res.silhouette_score:.4f} |"
        )

    md_lines.extend([
        "",
        "## K-Means Clustering (Domain Purity)",
        "",
    ])

    for layer, res in clustering.items():
        md_lines.append(f"### Layer {layer} (k={res.n_clusters})")
        md_lines.append("")
        md_lines.append("| Cluster | " + " | ".join(sorted(report['summary']['domains'])) + " |")
        md_lines.append("|---------|" + "|".join(["------"] * len(report['summary']['domains'])) + "|")

        for cluster_id in range(res.n_clusters):
            purity = res.domain_purity.get(cluster_id, {})
            row = f"| {cluster_id} |"
            for domain in sorted(report['summary']['domains']):
                row += f" {purity.get(domain, 0):.3f} |"
            md_lines.append(row)
        md_lines.append("")

    if correctness_corr:
        md_lines.extend([
            "## Correctness Correlation",
            "",
            "| Layer | Norm Corr (p) | Entropy Corr (p) | Margin Corr (p) | Conf Corr (p) |",
            "|-------|---------------|------------------|-----------------|---------------|",
        ])
        for layer, res in correctness_corr.items():
            md_lines.append(
                f"| {layer} | {res.norm_correlation:.4f} ({res.norm_pvalue:.4f}) | "
                f"{res.entropy_correlation:.4f} ({res.entropy_pvalue:.4f}) | "
                f"{res.margin_correlation:.4f} ({res.margin_pvalue:.4f}) | "
                f"{res.confidence_correlation:.4f} ({res.confidence_pvalue:.4f}) |"
            )

    if pca_results:
        md_lines.extend([
            "",
            "## PCA (First Layer)",
            "",
            f"**Layer:** {pca_results['layer']}",
            f"**Components:** {pca_results['n_components']}",
            f"**Cumulative Variance (top 10):** {pca_results['cumulative_variance'][:10]}",
        ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return out_path