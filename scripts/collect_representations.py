"""CLI script for Phase 3 representation collection and analysis."""

import argparse
import sys
from pathlib import Path

# Mask preinstalled incompatible torchao version on Kaggle
sys.modules["torchao"] = None

import torch
from ares.adaptation import (
    DomainSample,
    MultiDomainTextDataset,
    create_multi_domain_dataloader,
)
from ares.backbone.qwen_loader import load_qwen_model, load_qwen_tokenizer
from ares.representations import (
    RepresentationCollector,
    save_representations,
    generate_analysis_report,
    save_collection_metadata,
)
from ares.utils.config import ModelConfig, load_config
from ares.utils.environment import is_main_process, set_seed, setup_ddp, cleanup_ddp, is_ddp_initialized


def generate_synthetic_multi_domain_samples(
    max_per_domain: int = 100,
    seed: int = 42,
) -> list[DomainSample]:
    """Generate synthetic multi-domain samples for representation collection."""
    import random
    random.seed(seed)

    templates = {
        "general": [
            ("General Knowledge: The capital of {country} is {capital}.", "factual"),
            ("General Knowledge: {animal} is a {type} known for {trait}.", "factual"),
            ("General Knowledge: {event} happened in {year}.", "historical"),
        ],
        "math": [
            ("Solve the math problem step-by-step: {problem}", "arithmetic"),
            ("Solve the math problem step-by-step: If {a} {op} {b} = {c}, what is {x}?", "algebra"),
            ("Solve the math problem step-by-step: Calculate {expr}.", "calculation"),
        ],
        "code": [
            ("Write Python code for the following task: {task}", "function"),
            ("Write Python code for the following task: {task} using {lib}", "library"),
            ("Write Python code for the following task: Fix the bug in {code}", "debugging"),
        ],
        "science": [
            ("Science Question: What is {concept}?", "definition"),
            ("Science Question: Explain {phenomenon}.", "explanation"),
            ("Science Question: What causes {effect}?", "causality"),
        ],
    }

    # Sample data pools
    countries = [("France", "Paris"), ("Japan", "Tokyo"), ("Germany", "Berlin"), ("Brazil", "Brasília")]
    animals = [("dog", "mammal", "loyalty"), ("eagle", "bird", "keen eyesight"), ("shark", "fish", "sharp teeth")]
    events = [("Moon landing", "1969"), ("WWII end", "1945"), ("Declaration of Independence", "1776")]
    math_problems = ["5 + 3", "12 * 4", "20 - 8", "9 * 7", "100 / 4"]
    code_tasks = ["return the sum of a list", "check if a string is palindrome", "find max in array", "reverse a string"]
    concepts = ["photosynthesis", "gravity", "DNA", "electron", "black hole"]
    phenomena = ["rainbow formation", "volcanic eruption", "magnetism", "evaporation"]

    samples = []
    sample_id = 0

    for domain, domain_templates in templates.items():
        for i in range(max_per_domain):
            template, task_type = random.choice(domain_templates)

            if domain == "general":
                if "{country}" in template:
                    country, capital = random.choice(countries)
                    text = template.format(country=country, capital=capital)
                elif "{animal}" in template:
                    animal, atype, trait = random.choice(animals)
                    text = template.format(animal=animal, type=atype, trait=trait)
                else:
                    event, year = random.choice(events)
                    text = template.format(event=event, year=year)

            elif domain == "math":
                if "{problem}" in template:
                    text = template.format(problem=random.choice(math_problems))
                elif "{expr}" in template:
                    text = template.format(expr=random.choice(math_problems))
                else:
                    a, b = random.randint(1, 20), random.randint(1, 20)
                    ops = {"+": a+b, "-": a-b, "*": a*b, "/": a//b if b != 0 else 1}
                    op, c = random.choice(list(ops.items()))
                    x_val = a if op in ["+", "-"] else b
                    text = template.format(a=a, op=op, b=b, c=c, x=x_val)

            elif domain == "code":
                task = random.choice(code_tasks)
                if "{lib}" in template:
                    text = template.format(task=task, lib=random.choice(["standard lib", "numpy", "pandas"]))
                elif "{code}" in template:
                    text = template.format(code=random.choice(["def foo(): pass", "x = [1, 2, 3]"]))
                else:
                    text = template.format(task=task)

            elif domain == "science":
                if "{concept}" in template:
                    text = template.format(concept=random.choice(concepts))
                elif "{phenomenon}" in template:
                    text = template.format(phenomenon=random.choice(phenomena))
                else:
                    text = template.format(effect=random.choice(["rain", "earthquake", "tides"]))

            samples.append(DomainSample(
                sample_id=f"{domain}_{sample_id}",
                domain=domain,
                text=text,
                target=None,  # No target for representation collection
            ))
            sample_id += 1

    return samples


def load_real_datasets(
    domains: list[str],
    max_per_domain: int,
    seed: int = 42,
) -> list[DomainSample]:
    """Load real HF datasets for the specified domains."""
    from datasets import load_dataset
    import random
    random.seed(seed)

    samples = []
    sample_id = 0

    for domain in domains:
        if domain == "general":
            try:
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                texts = [ex["text"] for ex in ds if len(ex["text"].strip()) > 50]
                for text in random.sample(texts, min(max_per_domain, len(texts))):
                    samples.append(DomainSample(
                        sample_id=f"general_{sample_id}",
                        domain="general",
                        text=f"General Knowledge: {text[:500]}",
                    ))
                    sample_id += 1
            except Exception as e:
                print(f"Warning: Failed to load wikitext: {e}")

        elif domain == "math":
            try:
                ds = load_dataset("gsm8k", "main", split="train")
                for ex in ds.select(range(min(max_per_domain, len(ds)))):
                    samples.append(DomainSample(
                        sample_id=f"math_{sample_id}",
                        domain="math",
                        text=f"Solve the math problem step-by-step: {ex['question']}",
                        target=ex["answer"],
                    ))
                    sample_id += 1
            except Exception as e:
                print(f"Warning: Failed to load gsm8k: {e}")

        elif domain == "code":
            try:
                ds = load_dataset("mbpp", "sanitized", split="train")
                for ex in ds.select(range(min(max_per_domain, len(ds)))):
                    samples.append(DomainSample(
                        sample_id=f"code_{sample_id}",
                        domain="code",
                        text=f"Write Python code for the following task: {ex['text']}",
                        target=ex["code"],
                    ))
                    sample_id += 1
            except Exception as e:
                print(f"Warning: Failed to load mbpp: {e}")

        elif domain == "science":
            try:
                ds = load_dataset("ai2_arc", "ARC-Easy", split="train")
                for ex in ds.select(range(min(max_per_domain, len(ds)))):
                    question = ex["question"]
                    choices = ex["choices"]["text"]
                    text = f"Science Question: {question}\nChoices: {', '.join(choices)}"
                    samples.append(DomainSample(
                        sample_id=f"science_{sample_id}",
                        domain="science",
                        text=text,
                        target=ex["answerKey"],
                    ))
                    sample_id += 1
            except Exception as e:
                print(f"Warning: Failed to load ai2_arc: {e}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="ARES V2 - Phase 3 Representation Collection")
    parser.add_argument("--config", type=str, default="configs/reliability/representation_collection.yaml",
                        help="Path to representation collection config YAML")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Hugging Face model ID (overrides config)")
    parser.add_argument("--torch_dtype", type=str, default=None,
                        help="Precision format ('float16', 'bfloat16', 'float32')")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA foundation adapter checkpoint")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per domain (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for representations (overrides config)")
    parser.add_argument("--use_real_data", action="store_true",
                        help="Use real HF datasets instead of synthetic")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit NF4 quantization")
    parser.add_argument("--analyze", action="store_true",
                        help="Run analysis after collection")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--ddp", action="store_true",
                        help="Enable DDP (set automatically by torchrun)")
    args = parser.parse_args()

    # Setup DDP if requested/needed
    if args.ddp or is_ddp_initialized():
        local_rank = setup_ddp()
    else:
        local_rank = 0

    set_seed(args.seed, deterministic=True)

    # Load config
    cfg = load_config(args.config)
    rc_cfg = cfg.__dict__.get("representation_collection", {})
    # The config loader expects ExperimentConfig, but our YAML is just representation_collection
    # Let's load raw YAML for this specific config
    import yaml
    with open(args.config, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}
    rc_cfg = raw_cfg.get("representation_collection", raw_cfg)

    # Override from CLI
    if args.model_name:
        rc_cfg["model"] = rc_cfg.get("model", {})
        rc_cfg["model"]["name_or_path"] = args.model_name
    if args.torch_dtype:
        rc_cfg["model"] = rc_cfg.get("model", {})
        rc_cfg["model"]["torch_dtype"] = args.torch_dtype
    if args.max_samples:
        rc_cfg["max_samples_per_domain"] = args.max_samples
    if args.output_dir:
        rc_cfg["storage"] = rc_cfg.get("storage", {})
        rc_cfg["storage"]["output_dir"] = args.output_dir

    model_cfg = ModelConfig(
        name_or_path=rc_cfg.get("model", {}).get("name_or_path", "Qwen/Qwen2.5-7B"),
        revision=rc_cfg.get("model", {}).get("revision", "main"),
        torch_dtype=rc_cfg.get("model", {}).get("torch_dtype", "bfloat16"),
        device_map=rc_cfg.get("model", {}).get("device_map", "auto"),
        trust_remote_code=rc_cfg.get("model", {}).get("trust_remote_code", False),
        use_cache=rc_cfg.get("model", {}).get("use_cache", True),
        load_in_4bit=args.load_in_4bit or rc_cfg.get("model", {}).get("load_in_4bit", False),
    )

    if is_main_process():
        print(f"=== ARES V2 Phase 3 Representation Collection ===")
        print(f"Model: {model_cfg.name_or_path}")
        print(f"Layers: {rc_cfg.get('target_layers')}")
        print(f"Pooling: {rc_cfg.get('pooling_method')}")
        print(f"Domains: {rc_cfg.get('domains')}")
        print(f"Max samples/domain: {rc_cfg.get('max_samples_per_domain')}")
        print(f"DDP: {is_ddp_initialized()} (rank {local_rank})")

    # Load model & tokenizer
    tokenizer = load_qwen_tokenizer(model_cfg)
    base_model = load_qwen_model(model_cfg)

    # Load LoRA adapter if provided
    if args.checkpoint:
        if is_main_process():
            print(f"Loading LoRA adapter from: {args.checkpoint}")
        from peft import PeftModel
        base_model = PeftModel.from_pretrained(base_model, args.checkpoint)
        base_model.eval()

    # Generate/load samples
    if args.use_real_data:
        if is_main_process():
            print("Loading real HF datasets...")
        samples = load_real_datasets(
            rc_cfg.get("domains", ["general", "math", "code", "science"]),
            rc_cfg.get("max_samples_per_domain", 500),
            args.seed,
        )
    else:
        if is_main_process():
            print("Generating synthetic multi-domain samples...")
        samples = generate_synthetic_multi_domain_samples(
            rc_cfg.get("max_samples_per_domain", 500),
            args.seed,
        )

    if is_main_process():
        print(f"Total samples: {len(samples)}")
        domain_counts = {}
        for s in samples:
            domain_counts[s.domain] = domain_counts.get(s.domain, 0) + 1
        for d, c in domain_counts.items():
            print(f"  {d}: {c}")

    # Distribute samples across DDP ranks
    if is_ddp_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        samples = samples[rank::world_size]
        if is_main_process():
            print(f"Rank {rank} processing {len(samples)} samples")

    # Create dataset and dataloader
    dataset = MultiDomainTextDataset(
        samples,
        tokenizer,
        max_seq_length=model_cfg.max_position_embeddings,
    )
    dataloader, sampler = create_multi_domain_dataloader(
        dataset,
        batch_size=rc_cfg.get("batch_size", 4),
        shuffle=False,  # No need to shuffle for representation collection
    )

    # Create collector
    collector = RepresentationCollector(
        model=base_model,
        tokenizer=tokenizer,
        target_layers=rc_cfg.get("target_layers", [-1, -6, -12]),
        pooling_method=rc_cfg.get("pooling_method", "last_token"),
        compute_entropy=rc_cfg.get("compute_entropy", True),
        compute_margin=rc_cfg.get("compute_margin", True),
        config=model_cfg,
    )

    # Collect representations
    all_records = []
    if is_main_process():
        print("Collecting representations...")

    for batch_idx, batch in enumerate(dataloader):
        # Process batch
        batch_records = []
        for i in range(len(batch["input_ids"])):
            sample_idx = batch_idx * rc_cfg.get("batch_size", 4) + i
            if sample_idx < len(samples):
                sample = samples[sample_idx]
                records = collector.collect_sample(sample)
                batch_records.extend(records)

        all_records.extend(batch_records)

        if is_main_process() and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: {len(all_records)} records collected")

    # Only main process saves
    if is_main_process():
        output_dir = Path(rc_cfg.get("storage", {}).get("output_dir", "representations"))
        format = rc_cfg.get("storage", {}).get("format", "hdf5")
        chunk_size = rc_cfg.get("storage", {}).get("chunk_size", 1000)
        compression = rc_cfg.get("storage", {}).get("compression", "gzip")
        compression_opts = rc_cfg.get("storage", {}).get("compression_opts", 4)

        print(f"\nSaving {len(all_records)} records to {output_dir} ({format})...")
        save_path = save_representations(
            all_records,
            output_dir,
            format=format,
            chunk_size=chunk_size,
            compression=compression,
            compression_opts=compression_opts,
        )
        print(f"Saved to: {save_path}")

        # Save metadata
        stats = {
            "total_records": len(all_records),
            "total_samples": len(samples),
            "layers": rc_cfg.get("target_layers"),
            "domains": rc_cfg.get("domains"),
            "pooling_method": rc_cfg.get("pooling_method"),
            "model": model_cfg.name_or_path,
        }
        save_collection_metadata(output_dir, rc_cfg, stats)

        # Run analysis if requested
        if args.analyze:
            print("\nRunning analysis...")
            report_path = output_dir / "analysis_report.md"
            generate_analysis_report(
                all_records,
                report_path,
                layers=rc_cfg.get("target_layers"),
                n_clusters=len(rc_cfg.get("domains", 4)),
                include_pca=True,
                include_tsne=False,  # Slow, disable by default
            )
            print(f"Analysis report saved to: {report_path}")

    # Cleanup
    collector.cleanup()
    if is_ddp_initialized():
        cleanup_ddp()

    if is_main_process():
        print("\n=== Phase 3 Representation Collection Complete ===")


if __name__ == "__main__":
    main()