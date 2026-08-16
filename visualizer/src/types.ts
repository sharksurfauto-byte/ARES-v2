export type InferenceEvent = {
  token: string;
  token_id: number;
  token_index: number;
  sequence_position: number;
  is_prompt_token: boolean;

  global_reliability: number;
  local_reliability: number;
  combined_reliability: number;
  failure_risk: number;
  is_reliable: boolean;

  predicted_domain: string;
  domain_probabilities: Record<string, number>;
  domain_confidence: number;
  is_domain_certain: boolean;

  policy: string;
  selected_expert: string | null;
  requires_intervention: boolean;
  routing_reason: string;

  routing_latency_ms: number;
  expert_latency_ms: number;
  total_latency_ms: number;

  base_logits_used: boolean;
  expert_logits_used: boolean;
  expert_activation_count: number;
  timestamp: number;
  layer_idx: number;
};

export type TelemetrySnapshot = {
  tokens_generated: number;
  prompt_tokens: number;
  expert_activations: number;
  base_activations: number;
  expert_compute_percentage: number;
  expert_activation_reduction_vs_always_on: number;
  average_reliability: number;
  average_routing_latency_ms: number;
  average_expert_latency_ms: number;
  average_total_latency_ms: number;
  tokens_per_second: number;
  domain_distribution: Record<string, number>;
  expert_distribution: Record<string, number>;
};
