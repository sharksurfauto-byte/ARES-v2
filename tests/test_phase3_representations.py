"""Unit tests for Phase 3 representation collection, storage, and analysis."""

import sys
import os
import tempfile
import warnings

import torch
import torch.nn as nn

# Check for optional dependencies
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ares.representations.analysis import (
    ClusteringResult,  # noqa: E402 - imported for test access
)

import pytest
import torch
import numpy as np

from ares.representations.collector import (
    RepresentationRecord,
    RepresentationCollector,
    create_collector_from_config,
)
from ares.representations.storage import (
    save_representations,
    load_representations,
    save_collection_metadata,
    load_collection_metadata,
)
from ares.representations.analysis import (
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


class DummyEmbeddings:
    """Mock embeddings module."""
    weight = torch.nn.Parameter(torch.randn(100, 64))


class DummyModel(nn.Module):
    """Dummy model for hook testing, following Qwen-like structure."""
    def __init__(self):
        super().__init__()
        self.config = type('Config', (), {'num_hidden_layers': 12, 'hidden_size': 64, 'vocab_size': 100, '_name_or_path': 'dummy'})()
        self.embed = DummyEmbeddings()
        # Create 12 dummy layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
            ) for _ in range(12)
        ])
        self.lm_head = nn.Linear(64, 100)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None and len(kwargs) > 0:
            input_ids = next(iter(kwargs.values()))
        if input_ids is None:
            input_ids = torch.zeros((1, 4), dtype=torch.long)
        x = input_ids
        x = self.embed.weight[x.shape[0] % 100:x.shape[0] % 100 + x.shape[0]]
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def get_input_embeddings(self):
        class DummyEmbeddingsMod:
            weight = self.embed.weight
        return DummyEmbeddingsMod()


class DummyTokenizer:
    """Mock tokenizer."""
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, max_length=16, padding="max_length", truncation=True, return_tensors="pt"):
        tokens = [2, 3, 4, 5] + [self.pad_token_id] * (max_length - 4)
        input_ids = torch.tensor([tokens[:max_length]])
        attention_mask = torch.tensor([[1 if t != self.pad_token_id else 0 for t in tokens[:max_length]]])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens=True):
        return f"token_{token_ids[0]}"


@pytest.fixture
def dummy_model_and_collector():
    """Create dummy model and collector for testing."""
    model = DummyModel()
    tokenizer = DummyTokenizer()

    target_layers = [-1, -6, -12]

    collector = RepresentationCollector(
        model=model,
        tokenizer=tokenizer,
        target_layers=target_layers,
        pooling_method="last_token",
        compute_entropy=True,
        compute_margin=True,
    )

    return model, tokenizer, collector, target_layers


class TestRepresentationRecord:
    """Test RepresentationRecord dataclass."""

    def test_record_creation(self):
        """Test basic record creation."""
        record = RepresentationRecord(
            sample_id="test_1",
            domain="general",
            task="factual",
            layer=-1,
            representation=np.zeros(64, dtype=np.float32),
            logits=np.zeros(100, dtype=np.float32),
            prediction="token_5",
            confidence=0.85,
            entropy=1.2,
            margin=0.5,
        )

        assert record.sample_id == "test_1"
        assert record.layer == -1
        assert record.confidence == 0.85
        assert record.entropy == 1.2
        assert record.margin == 0.5

    def test_record_to_from_dict(self):
        """Test round-trip through dict."""
        record = RepresentationRecord(
            sample_id="test_2",
            domain="math",
            task="algebra",
            layer=-6,
            representation=np.ones(64, dtype=np.float32) * 0.5,
            logits=np.ones(100, dtype=np.float32),
            prediction="token_10",
            confidence=0.7,
            entropy=0.8,
        )

        d = record.to_dict()
        record2 = RepresentationRecord.from_dict(d)

        assert record2.sample_id == "test_2"
        assert record2.layer == -6
        assert record2.domain == "math"
        assert record2.confidence == 0.7


class TestRepresentationCollector:
    """Test RepresentationCollector."""

    def test_collector_initialization(self, dummy_model_and_collector):
        """Test collector registers hooks correctly."""
        model, tokenizer, collector, target_layers = dummy_model_and_collector

        # Should have registered hooks
        assert collector.hook_manager is not None
        assert len(collector.hook_manager.handles) > 0

    def test_collect_sample_basic(self, dummy_model_and_collector):
        """Test basic sample collection."""
        model, tokenizer, collector, target_layers = dummy_model_and_collector

        sample = type('obj', (object,), {
            'sample_id': 'test_sample',
            'domain': 'general',
            'text': 'Test sentence for representation collection.',
        })()

        records = collector.collect_sample(sample)

        # Should produce one record per target layer
        assert len(records) == len(target_layers), \
            f"Expected {len(target_layers)} records, got {len(records)}"

        for record in records:
            assert hasattr(record, 'representation')
            assert hasattr(record, 'logits')
            assert hasattr(record, 'prediction')
            assert hasattr(record, 'confidence')
            assert hasattr(record, 'entropy')
            assert hasattr(record, 'margin')
            # Check representation shape
            assert record.representation.shape[0] == 64, \
                f"Expected hidden dim 64, got {record.representation.shape[0]}"
            # Check values are finite
            assert np.isfinite(record.representation).all()
            assert np.isfinite(record.logits).all()

    def test_collect_batch(self, dummy_model_and_collector):
        """Test batch collection."""
        model, tokenizer, collector, target_layers = dummy_model_and_collector

        samples = [
            type('obj', (object,), {
                'sample_id': f'sample_{i}',
                'domain': 'general' if i % 2 == 0 else 'math',
                'text': f'Test sentence {i} for representation collection.',
            })()
            for i in range(4)
        ]

        records = collector.collect_batch(samples)

        # Should produce 4 samples * len(target_layers) records
        expected = 4 * len(target_layers)
        assert len(records) == expected, \
            f"Expected {expected} records, got {len(records)}"

    def test_cleanup(self, dummy_model_and_collector):
        """Test hook cleanup."""
        model, tokenizer, collector, target_layers = dummy_model_and_collector

        # Collect first
        sample = type('obj', (object,), {
            'sample_id': 'cleanup_test',
            'domain': 'general',
            'text': 'Cleanup test sentence.',
        })()
        _ = collector.collect_sample(sample)

        # Cleanup
        collector.cleanup()

        # Handles should be removed
        assert len(collector.hook_manager.handles) == 0

    def test_context_manager(self, dummy_model_and_collector):
        """Test context manager usage."""
        model, tokenizer, collector, target_layers = dummy_model_and_collector

        with RepresentationCollector(
            model=model,
            tokenizer=tokenizer,
            target_layers=target_layers,
            pooling_method="last_token",
            compute_entropy=True,
            compute_margin=True,
        ) as collector_ctx:
            sample = type('obj', (object,), {
                'sample_id': 'ctx_test',
                'domain': 'general',
                'text': 'Context manager test.',
            })()
            records = collector_ctx.collect_sample(sample)
            assert len(records) == len(target_layers)

        # After exit, hooks should be removed
        assert len(collector_ctx.hook_manager.handles) == 0


class TestCollectorFactory:
    """Test create_collector_from_config."""

    def test_create_from_config(self):
        """Test collector creation from config dict."""
        config = {
            "representation_collection": {
                "target_layers": [-1, -6, -12],
                "pooling_method": "last_token",
                "compute_entropy": True,
                "compute_margin": True,
            }
        }

        # Mock minimal model and tokenizer with named_modules support
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(64, 64) for _ in range(12)
                ])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = MockModel()
        model.get_input_embeddings = lambda: type('obj', (object,), {'weight': torch.nn.Parameter(torch.randn(100, 64))})()

        tokenizer = type('obj', (object,), {
            'pad_token_id': 0,
            'eos_token_id': 1,
            'decode': lambda token_ids, skip_special_tokens: f"token_{token_ids[0]}",
        })()

        collector = create_collector_from_config(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

        assert collector is not None
        assert len(collector.target_layers) == 3


class TestStorage:
    """Test storage functions."""

    @pytest.fixture
    def sample_records(self):
        """Create sample records for storage testing."""
        records = []
        for i in range(10):
            record = RepresentationRecord(
                sample_id=f"s{i}",
                domain="general",
                task="factual",
                layer=-1,
                representation=np.random.randn(64).astype(np.float32),
                logits=np.random.randn(100).astype(np.float32),
                prediction=f"token_{i}",
                confidence=0.5 + i * 0.05,
                entropy=1.0 + i * 0.1,
                margin=0.2 + i * 0.02,
            )
            records.append(record)
        return records

    def test_hdf5_save_load_roundtrip(self, sample_records, tmp_path):
        """Test HDF5 save and load round-trip."""
        import h5py

        # Skip if h5py not available
        if not HAS_H5PY:
            pytest.skip("h5py not available")

        output_dir = tmp_path / "hdf5_test"
        # Save
        h5_path = save_representations(sample_records, output_dir, format="hdf5")
        assert h5_path.exists()

        # Load
        loaded = load_representations(h5_path, format="hdf5")
        assert len(loaded) == len(sample_records)

        # Verify structure
        for orig, loaded_rec in zip(sample_records, loaded):
            assert orig.sample_id == loaded_rec.sample_id
            assert orig.layer == loaded_rec.layer
            assert orig.domain == loaded_rec.domain
            assert np.allclose(orig.representation, loaded_rec.representation)

    @pytest.fixture
    def h5py_available(self):
        """Check if h5py is available."""
        global HAS_H5PY
        try:
            import h5py
            return True
        except ImportError:
            return False

    def test_parquet_save_load(self, sample_records, tmp_path, h5py_available):
        """Test Parquet save and load round-trip."""
        output_dir = tmp_path / "parquet_test"
        # Save
        pq_path = save_representations(sample_records, output_dir, format="parquet")
        assert pq_path.exists()

    def test_save_load_metadata(self, sample_records, tmp_path):
        """Test metadata save and load."""
        metadata_path = save_collection_metadata(sample_records, tmp_path, {
            "config": {"test": "value"},
            "stats": {"n_records": len(sample_records)},
        })
        assert metadata_path.exists()

        loaded_meta = load_collection_metadata(metadata_path.parent)
        assert loaded_meta["config"]["test"] == "value"
        assert loaded_meta["stats"]["n_records"] == len(sample_records)


class TestAnalysis:
    """Test analysis functions."""

    @pytest.fixture
    def sample_records_analysis(self):
        """Create records suitable for analysis."""
        records = []
        for i in range(50):
            record = RepresentationRecord(
                sample_id=f"s{i}",
                domain=["general", "math", "code", "science"][i % 4],
                task=["factual", "arithmetic", "function", "definition"][i % 4],
                layer=[-1, -6, -12][i % 3],
                representation=np.random.randn(32).astype(np.float32),
                logits=np.random.randn(100).astype(np.float32),
                prediction=f"token_{i % 10}",
                correctness=(i % 5 != 0),  # 80% correct
                confidence=0.5 + (i % 5) * 0.1,
                entropy=1.0 + (i % 5) * 0.2,
                margin=0.2 + (i % 5) * 0.05,
            )
            records.append(record)
        return records

    def test_compute_layer_statistics(self, sample_records_analysis):
        """Test layer statistics computation."""
        stats = compute_layer_statistics(sample_records_analysis)

        # Should have stats for all 3 layers
        assert len(stats) == 3  # 3 unique layers
        for layer in [-1, -6, -12]:
            assert layer in stats
            layer_stats = stats[layer]
            assert layer_stats.n_samples > 0
            assert layer_stats.hidden_dim > 0
            assert np.isfinite(layer_stats.mean_norm)
            assert np.isfinite(layer_stats.std_norm)

    def test_compute_domain_separability(self, sample_records_analysis):
        """Test domain separability analysis."""
        result = compute_domain_separability(sample_records_analysis, layer=-1)

        assert result.layer == -1
        assert 0.0 <= result.lda_accuracy <= 1.0
        # Silhouette score can be negative; just verify it's a valid float
        assert isinstance(result.silhouette_score, float)
        assert len(result.domain_centroids) > 0
        assert len(result.domain_counts) > 0

    def test_run_pca(self, sample_records_analysis):
        """Test PCA computation."""
        layer_records_count = len([r for r in sample_records_analysis if r.layer == -1])
        transformed, pca_model = run_pca(sample_records_analysis, layer=-1, n_components=10)

        assert transformed.shape[0] == layer_records_count
        assert transformed.shape[1] == 10  # n_components
        assert pca_model.explained_variance_ratio_.sum() <= 1.0

    def test_run_tsne(self, sample_records_analysis):
        """Test t-SNE computation."""
        layer_records_count = len([r for r in sample_records_analysis if r.layer == -1])
        # sklearn t-SNE requires max_iter >= 250
        transformed = run_tsne(sample_records_analysis, layer=-1, n_components=2, perplexity=5, max_iter=250)

        assert transformed.shape[0] == layer_records_count
        assert transformed.shape[1] == 2

    def test_cluster_representations(self, sample_records_analysis):
        """Test K-means clustering."""
        result = cluster_representations(sample_records_analysis, layer=-1, n_clusters=4)

        assert result.layer == -1
        assert result.n_clusters == 4
        # sample_records_analysis has 50 records split across 3 layers (~17 per layer)
        assert len(result.cluster_labels) <= 50
        assert len(result.cluster_centers) == 4
        assert result.inertia > 0
        assert len(result.domain_purity) == 4

    def test_analyze_correctness_correlation_with_labels(self, sample_records_analysis):
        """Test correctness correlation with labels."""
        result = analyze_correctness_correlation(sample_records_analysis, layer=-1)

        # Should have results for all feature types
        assert result is not None
        assert hasattr(result, 'norm_correlation')
        assert hasattr(result, 'entropy_correlation')
        assert hasattr(result, 'margin_correlation')
        assert hasattr(result, 'confidence_correlation')

    def test_analyze_correctness_correlation_no_labels(self):
        """Test correctness correlation without labels returns None."""
        # Create records without correctness
        records = []
        for i in range(20):
            record = RepresentationRecord(
                sample_id=f"s{i}",
                domain="general",
                task="factual",
                layer=-1,
                representation=np.random.randn(32).astype(np.float32),
                logits=np.random.randn(100).astype(np.float32),
                prediction=f"token_{i}",
                correctness=None,  # No correctness labels
                confidence=0.5,
                entropy=1.0,
                margin=0.2,
            )
            records.append(record)

        result = analyze_correctness_correlation(records, layer=-1)
        assert result is None

    def test_compute_all_separability(self, sample_records_analysis):
        """Test all-separability computation."""
        results = compute_all_separability(sample_records_analysis, layers=[-1, -6])

        assert -1 in results
        assert -6 in results
        for layer, res in results.items():
            assert isinstance(res, type(results[-1]))

    def test_compute_all_clustering(self, sample_records_analysis):
        """Test all-clustering computation."""
        results = compute_all_clustering(sample_records_analysis, layers=[-1, -6], n_clusters=3)

        assert -1 in results
        assert -6 in results
        for layer, res in results.items():
            assert isinstance(res, ClusteringResult)

    def test_compute_all_correctness_correlations(self, sample_records_analysis):
        """Test all-correctness correlations."""
        results = compute_all_correctness_correlations(sample_records_analysis, layers=[-1])

        assert -1 in results
        result = results[-1]
        assert hasattr(result, 'norm_correlation')

    def test_generate_analysis_report(self, sample_records_analysis, tmp_path):
        """Test analysis report generation."""
        # Add correctness to records for correlation analysis
        for r in sample_records_analysis:
            if r.correctness is None:
                r.correctness = True

        report_path = tmp_path / "analysis_report.md"
        report_file = generate_analysis_report(
            sample_records_analysis,
            report_path,
            layers=[-1, -6],
            n_clusters=3,
            include_pca=True,
            include_tsne=False,
        )

        assert report_file.exists()
        assert report_file.suffix == ".md"

        # Check JSON report also exists
        json_path = report_path.parent / "analysis_report.json"
        assert json_path.exists()

        # Read markdown and verify key sections
        with open(report_file, "r") as f:
            content = f.read()

        assert "Representation Analysis Report" in content
        assert "Layer Statistics" in content
        assert "Domain Separability" in content