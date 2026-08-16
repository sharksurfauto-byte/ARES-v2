"""Efficient storage for representation records (HDF5/Parquet) with metadata."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

from ares.representations.collector import RepresentationRecord


def _records_to_arrays(records: List[RepresentationRecord]) -> Dict[str, Any]:
    """Convert list of records to columnar arrays for storage."""
    if not records:
        return {}

    # Group by layer
    layers = sorted(set(r.layer for r in records))
    layer_to_records = {layer: [r for r in records if r.layer == layer] for layer in layers}

    # Metadata arrays (one per record)
    sample_ids = []
    domains = []
    tasks = []
    record_layers = []
    correctness = []
    confidence = []
    entropy = []
    margin = []
    predictions = []

    # Representation arrays per layer
    representations = {}
    logits = {}

    for layer in layers:
        layer_records = layer_to_records[layer]
        n = len(layer_records)

        reps = np.stack([r.representation for r in layer_records], axis=0)
        lgts = np.stack([r.logits for r in layer_records], axis=0)

        representations[layer] = reps
        logits[layer] = lgts

        for r in layer_records:
            sample_ids.append(r.sample_id)
            domains.append(r.domain)
            tasks.append(r.task)
            record_layers.append(r.layer)
            correctness.append(r.correctness if r.correctness is not None else -1)
            confidence.append(r.confidence)
            entropy.append(r.entropy)
            margin.append(r.margin)
            predictions.append(r.prediction)

    return {
        "sample_ids": np.array(sample_ids, dtype=object),
        "domains": np.array(domains, dtype=object),
        "tasks": np.array(tasks, dtype=object),
        "layers": np.array(record_layers, dtype=np.int32),
        "correctness": np.array(correctness, dtype=np.int8),
        "confidence": np.array(confidence, dtype=np.float32),
        "entropy": np.array(entropy, dtype=np.float32),
        "margin": np.array(margin, dtype=np.float32),
        "predictions": np.array(predictions, dtype=object),
        "representations": representations,
        "logits": logits,
        "unique_layers": np.array(layers, dtype=np.int32),
    }


def _decode_str(val: Any) -> str:
    """Safely decode string or bytes values."""
    if isinstance(val, bytes):
        return val.decode("utf-8")
    elif isinstance(val, str):
        return val
    elif hasattr(val, "item"):
        v = val.item()
        if isinstance(v, bytes):
            return v.decode("utf-8")
        return str(v)
    return str(val)


def _arrays_to_records(arrays: Dict[str, Any]) -> List[RepresentationRecord]:
    """Convert stored arrays back to RepresentationRecord objects."""
    n_records = len(arrays["sample_ids"])
    layers = arrays.get("unique_layers", np.unique(arrays["layers"]))

    # Build layer -> record indices mapping
    layer_indices = {}
    for i, layer in enumerate(arrays["layers"]):
        layer_indices.setdefault(layer, []).append(i)

    records = []
    for layer in layers:
        indices = layer_indices[layer]
        reps = arrays["representations"][layer]
        lgts = arrays["logits"][layer]

        for j, idx in enumerate(indices):
            record = RepresentationRecord(
                sample_id=_decode_str(arrays["sample_ids"][idx]),
                domain=_decode_str(arrays["domains"][idx]),
                task=_decode_str(arrays["tasks"][idx]),
                layer=int(arrays["layers"][idx]),
                representation=reps[j],
                logits=lgts[j],
                prediction=_decode_str(arrays["predictions"][idx]),
                correctness=bool(arrays["correctness"][idx]) if arrays["correctness"][idx] >= 0 else None,
                confidence=float(arrays["confidence"][idx]),
                entropy=float(arrays["entropy"][idx]),
                margin=float(arrays["margin"][idx]),
            )
            records.append(record)

    return records


def save_representations_hdf5(
    records: List[RepresentationRecord],
    output_dir: Union[str, Path],
    chunk_size: int = 1000,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> Path:
    """Save representations to HDF5 format with hierarchical structure.

    Structure:
    /representations/
      /layer_{idx}/  (dataset: [N, hidden_dim])
    /logits/
      /layer_{idx}/  (dataset: [N, vocab_size])
    /metadata/
      sample_ids, domains, tasks, layers, correctness, confidence, entropy, margin, predictions
    """
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 storage. Install with: pip install h5py")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "representations.h5"

    arrays = _records_to_arrays(records)

    with h5py.File(h5_path, "w") as f:
        # Representations group
        rep_group = f.create_group("representations")
        for layer, reps in arrays["representations"].items():
            layer_name = f"layer_{layer}"
            rep_group.create_dataset(
                layer_name,
                data=reps,
                chunks=(min(chunk_size, reps.shape[0]), reps.shape[1]),
                compression=compression,
                compression_opts=compression_opts,
            )

        # Logits group
        logits_group = f.create_group("logits")
        for layer, lgts in arrays["logits"].items():
            layer_name = f"layer_{layer}"
            logits_group.create_dataset(
                layer_name,
                data=lgts,
                chunks=(min(chunk_size, lgts.shape[0]), lgts.shape[1]),
                compression=compression,
                compression_opts=compression_opts,
            )

        # Metadata group
        meta_group = f.create_group("metadata")
        for key in ["sample_ids", "domains", "tasks", "layers", "correctness",
                    "confidence", "entropy", "margin", "predictions"]:
            data = arrays[key]
            if data.dtype == object:
                # Variable-length strings
                dt = h5py.string_dtype(encoding="utf-8")
                meta_group.create_dataset(key, data=data, dtype=dt)
            else:
                meta_group.create_dataset(key, data=data)

        # Store unique layers
        meta_group.create_dataset("unique_layers", data=arrays["unique_layers"])

    return h5_path


def load_representations_hdf5(input_path: Union[str, Path]) -> List[RepresentationRecord]:
    """Load representations from HDF5 file."""
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 storage. Install with: pip install h5py")

    arrays = {"representations": {}, "logits": {}}

    with h5py.File(input_path, "r") as f:
        # Load representations
        rep_group = f["representations"]
        for layer_name in rep_group.keys():
            layer = int(layer_name.replace("layer_", ""))
            arrays["representations"][layer] = rep_group[layer_name][:]

        # Load logits
        logits_group = f["logits"]
        for layer_name in logits_group.keys():
            layer = int(layer_name.replace("layer_", ""))
            arrays["logits"][layer] = logits_group[layer_name][:]

        # Load metadata
        meta_group = f["metadata"]
        for key in meta_group.keys():
            if key != "unique_layers":
                arrays[key] = meta_group[key][:]

    return _arrays_to_records(arrays)


def save_representations_parquet(
    records: List[RepresentationRecord],
    output_dir: Union[str, Path],
    chunk_size: int = 1000,
) -> Path:
    """Save representations to Parquet format (columnar, good for analytics)."""
    if not PARQUET_AVAILABLE:
        raise ImportError("pyarrow is required for Parquet storage. Install with: pip install pyarrow")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "representations.parquet"

    arrays = _records_to_arrays(records)

    # Build a flat table (one row per record)
    # Note: representations and logits are stored as lists of arrays
    data = {
        "sample_id": arrays["sample_ids"],
        "domain": arrays["domains"],
        "task": arrays["tasks"],
        "layer": arrays["layers"],
        "correctness": arrays["correctness"],
        "confidence": arrays["confidence"],
        "entropy": arrays["entropy"],
        "margin": arrays["margin"],
        "prediction": arrays["predictions"],
        "representation": [arr for arr in arrays["representations"].values() for _ in range(len(arr))],
        "logits": [arr for arr in arrays["logits"].values() for _ in range(len(arr))],
    }

    # This approach doesn't work well for nested arrays in Parquet
    # Better: store metadata in one file, representations per-layer in separate files
    meta_data = {k: v for k, v in data.items() if k not in ("representation", "logits")}
    table = pa.table(meta_data)
    pq.write_table(table, out_dir / "metadata.parquet")

    # Store representations per layer
    for layer, reps in arrays["representations"].items():
        layer_table = pa.table({
            "sample_id": arrays["sample_ids"][arrays["layers"] == layer],
            "representation": list(reps),
        })
        pq.write_table(layer_table, out_dir / f"representations_layer_{layer}.parquet")

    for layer, lgts in arrays["logits"].items():
        layer_table = pa.table({
            "sample_id": arrays["sample_ids"][arrays["layers"] == layer],
            "logits": list(lgts),
        })
        pq.write_table(layer_table, out_dir / f"logits_layer_{layer}.parquet")

    meta_path = out_dir / "metadata.parquet"
    return meta_path


def load_representations_parquet(input_dir: Union[str, Path]) -> List[RepresentationRecord]:
    """Load representations from Parquet files."""
    if not PARQUET_AVAILABLE:
        raise ImportError("pyarrow is required for Parquet storage. Install with: pip install pyarrow")

    in_dir = Path(input_dir)

    # Load metadata
    meta_table = pq.read_table(in_dir / "metadata.parquet")
    meta = meta_table.to_pydict()

    # Load representations and logits per layer
    arrays = {
        "sample_ids": np.array(meta["sample_id"]),
        "domains": np.array(meta["domain"]),
        "tasks": np.array(meta["task"]),
        "layers": np.array(meta["layer"]),
        "correctness": np.array(meta["correctness"]),
        "confidence": np.array(meta["confidence"]),
        "entropy": np.array(meta["entropy"]),
        "margin": np.array(meta["margin"]),
        "predictions": np.array(meta["prediction"]),
        "representations": {},
        "logits": {},
    }

    for parquet_file in in_dir.glob("representations_layer_*.parquet"):
        layer = int(parquet_file.stem.replace("representations_layer_", ""))
        table = pq.read_table(parquet_file)
        data = table.to_pydict()
        arrays["representations"][layer] = np.stack(data["representation"])

    for parquet_file in in_dir.glob("logits_layer_*.parquet"):
        layer = int(parquet_file.stem.replace("logits_layer_", ""))
        table = pq.read_table(parquet_file)
        data = table.to_pydict()
        arrays["logits"][layer] = np.stack(data["logits"])

    arrays["unique_layers"] = np.unique(arrays["layers"])
    return _arrays_to_records(arrays)


def save_representations(
    records: List[RepresentationRecord],
    output_dir: Union[str, Path],
    format: str = "hdf5",
    chunk_size: int = 1000,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> Path:
    """Save representations in specified format.

    Args:
        records: List of RepresentationRecord objects
        output_dir: Output directory
        format: "hdf5" or "parquet"
        chunk_size: Chunk size for HDF5
        compression: HDF5 compression algorithm
        compression_opts: HDF5 compression options

    Returns:
        Path to saved file/directory
    """
    if format == "hdf5":
        return save_representations_hdf5(
            records, output_dir, chunk_size, compression, compression_opts
        )
    elif format == "parquet":
        return save_representations_parquet(records, output_dir, chunk_size)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'hdf5' or 'parquet'")


def load_representations(input_path: Union[str, Path], format: str = "hdf5") -> List[RepresentationRecord]:
    """Load representations from file/directory."""
    p = Path(input_path)
    if p.is_dir() and (p / "metadata.parquet").exists():
        return load_representations_parquet(p)
    if p.is_file() and p.name == "metadata.parquet":
        return load_representations_parquet(p.parent)

    if format == "hdf5":
        return load_representations_hdf5(input_path)
    elif format == "parquet":
        dir_path = p if p.is_dir() else p.parent
        return load_representations_parquet(dir_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'hdf5' or 'parquet'")


def save_collection_metadata(
    output_dir: Union[str, Path, List[Any]],
    config: Union[Dict[str, Any], str, Path] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save collection metadata (config + statistics) as JSON sidecar."""
    if isinstance(output_dir, (list, tuple)):
        # Handle signature: save_collection_metadata(sample_records, output_dir, metadata_dict)
        records = output_dir
        out_dir = Path(config)
        meta_dict = stats if isinstance(stats, dict) else {}
        cfg = meta_dict.get("config", {})
        st = meta_dict.get("stats", {"n_records": len(records)})
    else:
        out_dir = Path(output_dir)
        cfg = config if isinstance(config, dict) else {}
        st = stats if isinstance(stats, dict) else {}

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "collection_meta.json"

    metadata = {
        "config": cfg,
        "stats": st,
        "statistics": st,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return meta_path


def load_collection_metadata(input_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load collection metadata from JSON sidecar."""
    meta_path = Path(input_dir) / "collection_meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)