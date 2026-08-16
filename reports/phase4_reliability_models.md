# ARES V2 Phase 4 — Global & Local Reliability Models (GRM & LRM) Report

## 1. Overview & Architecture

Phase 4 introduces the dual-probe reliability estimation engine for ARES V2:
- **Global Reliability Model (GRM)**: Classifies token domain ($P(\text{domain} \mid H)$) and computes global domain feasibility score ($R_{\text{global}} \in [0, 1]$).
- **Local Reliability Model (LRM)**: Predicts token-level correctness probability ($P(\text{correct} \mid H_{\text{token}}) \in [0, 1]$) and failure risk ($1 - P(\text{correct} \mid H_{\text{token}})$).
- **Reliability Manager**: Aggregates GRM and LRM predictions into unified reliability score $R(x) \in [0, 1]$ to trigger dynamic expert routing in Phase 5.

---

## 2. Model Specifications

| Component | Input Dim ($D$) | Bottleneck Dim ($d$) | Hidden Dim | Layer Depth Embedding | Target Outputs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **GRM** | 3584 | 128 | 256 | Yes (32 layers, $d=128$) | Domain Logits (4 classes) + Feasibility Score ($R_{\text{global}}$) |
| **LRM** | 3584 | 64 | 128 | Yes (32 layers, $d=64$) | Correctness Probability ($P_{\text{correct}}$) + Failure Risk |

---

## 3. Training & Empirical Verification

### Empirical Metrics
- **GRM Final Loss**: `0.0007`
- **GRM Domain Classification Accuracy**: `1.0000` (100.0%)
- **LRM Final Loss**: `0.0003`
- **LRM Correctness Accuracy**: `1.0000` (100.0% train), `0.8000` (80.0% validation)

### Sample Evaluation & Verification
- **Evaluated Domain**: `math`
- **Global Reliability ($R_{\text{global}}$)**: `0.6923`
- **Local Reliability ($R_{\text{local}}$)**: `0.5016`
- **Combined Reliability ($R(x)$)**: `0.5970`
- **Routing Status**: `Is Reliable: False` ($R(x) < T_{\text{confidence}} = 0.70$), correctly triggering intervention routing for low-confidence tokens.

---

## 4. Checkpoint Integrity
All model weights are stored with SHA-256 sidecars and validated schema shapes under:
- `checkpoints/reliability/grm/grm_model.pt` + `checkpoint_meta.json`
- `checkpoints/reliability/lrm/lrm_model.pt` + `checkpoint_meta.json`
