"""Token-by-token streaming generation with ARES adaptive routing.

Implements the core generation loop with correct two-pass semantics:
- Base forward pass for routing decision
- Expert forward pass (if intervention) for actual token generation
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Tuple
import time
import torch
from transformers import PreTrainedTokenizer

from ares.inference.engine import ARESInferenceEngine
from ares.inference.events import InferenceEvent, GenerationConfig, RoutingPolicyConfig
from ares.routing.router import RoutingDecision


def generate_stream(
    engine: ARESInferenceEngine,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    do_sample: bool = True,
    policy: str = "adaptive",
    layer_idx: Optional[int] = None,
) -> Generator[InferenceEvent, None, None]:
    """
    Generate tokens with ARES adaptive routing, yielding InferenceEvent per token.

    Core generation loop with correct two-pass semantics for interventions:

    For each generation step:
    1. BASE forward pass → extract hidden state → evaluate reliability → route
    2. IF intervention required:
         a. EXPERT forward pass → logits
         b. expert_latency_ms measured
       ELSE:
         a. BASE logits already available
         b. expert_latency_ms = 0
    3. Sample next token
    4. Forward with ACTIVE expert to get NEXT hidden state
    5. Yield InferenceEvent with full latency breakdown

    Args:
        engine: Initialized ARESInferenceEngine
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to sample (True) or greedy (False)
        policy: Routing policy ('adaptive', 'always_base', 'always_expert', 'random_expert')
        layer_idx: Layer for hidden state extraction (None = engine.target_layer)

    Yields:
        InferenceEvent for each generated token (and prompt tokens if tracked)
    """
    if layer_idx is None:
        layer_idx = engine.target_layer

    # Tokenize prompt
    inputs = engine.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(engine.input_device)
    attention_mask = inputs["attention_mask"].to(engine.input_device)

    # Initial prompt length
    prompt_length = input_ids.shape[1]

    # Reset expert state for new generation
    engine.set_active_expert(None)
    engine.expert_activation_count = 0

    # ──────────────────────────────────────────────────────────────
    # INITIAL BASE FORWARD to get first hidden state for routing
    # ──────────────────────────────────────────────────────────────
    start_routing = time.perf_counter()
    base_logits = engine.forward_base(input_ids, attention_mask)
    routing_latency_ms = (time.perf_counter() - start_routing) * 1000

    # Extract hidden state from BASE forward
    hidden_state = engine.get_hidden_state(input_ids, attention_mask)

    # Evaluate reliability on BASE hidden state
    rel_result = engine.evaluate_reliability(hidden_state, layer_idx=layer_idx)

    # Route to decide first expert
    routing_decision = engine.route(hidden_state, layer_idx=layer_idx, policy=policy)

    # Apply routing decision (set active expert if intervention)
    if routing_decision.requires_intervention:
        engine.set_active_expert(routing_decision.selected_expert)
        engine.expert_activation_count += 1
    else:
        engine.set_active_expert(None)

    # Sample first token from BASE logits (routing was on BASE)
    first_token_id, first_token_str = engine.sample_next_token(
        base_logits, temperature=temperature, do_sample=do_sample
    )

    # Yield first token event
    first_event = InferenceEvent(
        token=first_token_str,
        token_id=first_token_id,
        token_index=0,
        sequence_position=prompt_length,
        is_prompt_token=False,
        global_reliability=rel_result.global_reliability,
        local_reliability=rel_result.local_reliability,
        combined_reliability=rel_result.combined_reliability,
        failure_risk=rel_result.failure_risk,
        is_reliable=rel_result.is_reliable,
        predicted_domain=rel_result.predicted_domain,
        domain_probabilities=rel_result.domain_probabilities,
        domain_confidence=routing_decision.domain_confidence,
        is_domain_certain=routing_decision.is_domain_certain,
        policy=routing_decision.policy,
        selected_expert=routing_decision.selected_expert,
        requires_intervention=routing_decision.requires_intervention,
        routing_reason=routing_decision.reason,
        routing_latency_ms=routing_latency_ms,
        expert_latency_ms=0.0,  # First token uses BASE logits
        total_latency_ms=routing_latency_ms,
        base_logits_used=True,
        expert_logits_used=False,
        expert_activation_count=engine.expert_activation_count,
        timestamp=time.time(),
        layer_idx=layer_idx,
    )
    yield first_event

    # ──────────────────────────────────────────────────────────────
    # MAIN GENERATION LOOP
    # ──────────────────────────────────────────────────────────────
    generated_tokens = 1
    current_input_ids = torch.cat([input_ids, torch.tensor([[first_token_id]], device=engine.input_device)], dim=1)
    current_attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=engine.input_device, dtype=attention_mask.dtype)], dim=1)

    for token_idx in range(1, max_new_tokens):
        # ──────────────────────────────────────────────────────
        # ROUTING PASS: Base forward → hidden state → GRM/LRM → Router
        # ──────────────────────────────────────────────────────
        routing_start = time.perf_counter()

        # Base forward for routing
        base_logits = engine.forward_base(current_input_ids, current_attention_mask)

        # Extract hidden state from BASE
        hidden_state = engine.get_hidden_state(current_input_ids, current_attention_mask)

        # Evaluate reliability
        rel_result = engine.evaluate_reliability(hidden_state, layer_idx=layer_idx)

        # Route
        routing_decision = engine.route(hidden_state, layer_idx=layer_idx, policy=policy)

        routing_latency_ms = (time.perf_counter() - routing_start) * 1000

        # ──────────────────────────────────────────────────────
        # GENERATION PASS: Expert forward (if intervention) or use base logits
        # ──────────────────────────────────────────────────────
        expert_latency_ms = 0.0
        base_logits_used = False
        expert_logits_used = False

        if routing_decision.requires_intervention:
            # Expert intervention needed
            expert_name = routing_decision.selected_expert

            # Apply routing decision BEFORE expert forward
            if expert_name != engine.active_expert:
                engine.set_active_expert(expert_name)

            # Expert forward pass for GENERATION
            gen_start = time.perf_counter()
            expert_logits = engine.forward_expert(current_input_ids, current_attention_mask, expert_name)
            expert_latency_ms = (time.perf_counter() - gen_start) * 1000

            # Sample from expert logits
            token_id, token_str = engine.sample_next_token(
                expert_logits, temperature=temperature, do_sample=do_sample
            )

            expert_logits_used = True
            base_logits_used = False

            engine.expert_activation_count += 1

        else:
            # No intervention - use BASE logits for generation
            token_id, token_str = engine.sample_next_token(
                base_logits, temperature=temperature, do_sample=do_sample
            )

            base_logits_used = True
            expert_logits_used = False

        total_latency_ms = routing_latency_ms + expert_latency_ms

        # ──────────────────────────────────────────────────────
        # YIELD EVENT
        # ──────────────────────────────────────────────────────
        event = InferenceEvent(
            token=token_str,
            token_id=token_id,
            token_index=token_idx,
            sequence_position=prompt_length + token_idx,
            is_prompt_token=False,
            global_reliability=rel_result.global_reliability,
            local_reliability=rel_result.local_reliability,
            combined_reliability=rel_result.combined_reliability,
            failure_risk=rel_result.failure_risk,
            is_reliable=rel_result.is_reliable,
            predicted_domain=rel_result.predicted_domain,
            domain_probabilities=rel_result.domain_probabilities,
            domain_confidence=routing_decision.domain_confidence,
            is_domain_certain=routing_decision.is_domain_certain,
            policy=routing_decision.policy,
            selected_expert=routing_decision.selected_expert,
            requires_intervention=routing_decision.requires_intervention,
            routing_reason=routing_decision.reason,
            routing_latency_ms=routing_latency_ms,
            expert_latency_ms=expert_latency_ms,
            total_latency_ms=total_latency_ms,
            base_logits_used=base_logits_used,
            expert_logits_used=expert_logits_used,
            expert_activation_count=engine.expert_activation_count,
            timestamp=time.time(),
            layer_idx=layer_idx,
        )
        yield event

        # ──────────────────────────────────────────────────────
        # PREPARE FOR NEXT ITERATION
        # ──────────────────────────────────────────────────────
        # Append generated token to sequence
        current_input_ids = torch.cat(
            [current_input_ids, torch.tensor([[token_id]], device=engine.input_device)],
            dim=1
        )
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), device=engine.input_device, dtype=current_attention_mask.dtype)],
            dim=1
        )

        generated_tokens += 1

        # Check for EOS
        if token_id == engine.tokenizer.eos_token_id:
            break

    # End of generation


def generate_batch(
    engine: ARESInferenceEngine,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    do_sample: bool = True,
    policy: str = "adaptive",
    layer_idx: Optional[int] = None,
) -> List[List[InferenceEvent]]:
    """Generate multiple prompts in batch (sequential for now).

    Args:
        engine: Initialized ARESInferenceEngine
        prompts: List of input prompts
        max_new_tokens: Maximum tokens per prompt
        temperature: Sampling temperature
        do_sample: Whether to sample
        policy: Routing policy
        layer_idx: Layer for hidden state extraction

    Returns:
        List of event lists (one per prompt)
    """
    results = []
    for prompt in prompts:
        events = list(generate_stream(
            engine=engine,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            policy=policy,
            layer_idx=layer_idx,
        ))
        results.append(events)
    return results