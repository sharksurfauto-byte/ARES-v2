# AGENTS.md — AI Engineering & Implementation Contract

This document forms the binding engineering rules and execution contract for all AI coding agents working on **ARES V2** (Adaptive Reliable Expert System).

---

## 1. Core Mandates & Principles

1. **Phase-by-Phase Execution**:
   - Development must proceed sequentially through explicit phases (Phase 0 -> Phase 1 -> Phase 2 -> ...).
   - A phase must be fully verified and tested before advancing to the next phase.

2. **File-by-File Modular Implementation**:
   - Code must be built file-by-file with clear architectural responsibility and inline documentation.
   - Relying on authoritative, standard open-source libraries (e.g., PyTorch, Hugging Face `transformers`, `accelerate`, `peft`, `safetensors`) is expected and mandated over reinventing complex components from scratch.

3. **Multi-GPU & DDP Training Compatibility**:
   - All training pipelines and heavy compute modules MUST natively support Multi-GPU execution via PyTorch DistributedDataParallel (DDP) or Hugging Face `Accelerate`.
   - Code must be runnable both locally (single-GPU / CPU debugging) and on Kaggle (dual-T4 GPUs with DDP).

4. **Backbone-First & Strict Checkpoint Integrity**:
   - The pretrained Qwen model backbone is frozen/active and integrated via official library interfaces (`AutoModelForCausalLM`, `AutoTokenizer`).
   - **NO SILENT FALLBACKS**: Checkpoint loading must NEVER use unvalidated `strict=False` or swallow errors with silent `try/except` blocks. Missing/unexpected tensor keys or shape mismatches must fail loudly.

5. **Configuration-Driven Reproducibility**:
   - All experimental parameters (model IDs, hyperparameters, thresholds, random seeds) must reside in YAML configuration files under `configs/`. No hard-coded magic constants in code.

---

## 2. Explicit Prohibitions

AI agents working on this repository are strictly forbidden from:

- ❌ **Silently catching checkpoint errors** or loading uninitialized random weights when a checkpoint fails.
- ❌ **Using `strict=False`** on weight loads without explicit, documented adapter validation.
- ❌ **Deleting or overwriting working code** without explicit user instruction.
- ❌ **Bypassing or mocking unit tests** to claim success without empirical runtime verification.
- ❌ **Modifying architecture or model topologies** without documenting and updating configuration specs.
- ❌ **Changing dependency versions unnecessarily** or introducing unverified external packages.
- ❌ **Hardcoding single-GPU specific calls** (`.cuda()`, `.to('cuda:0')`) without DDP / rank / device abstraction.

---

## 3. Mandatory Development Workflow

For every file modification or component addition, agents must strictly adhere to:

```text
Inspect ➔ Plan ➔ Modify ➔ Test ➔ Report ➔ Commit
```

1. **Inspect**: Read relevant code, PRD sections, and current test suites.
2. **Plan**: Present clear rationale and file content explanations to the user.
3. **Modify**: Write clean, modular, typed, and docstring-annotated code.
4. **Test**: Run unit tests and runtime verification scripts.
5. **Report**: Synthesize results, log outputs, and verification metrics clearly.
6. **Commit**: Ensure state is clean before moving to the next task.
