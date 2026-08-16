import React, { useState } from 'react';
import type { InferenceEvent } from '../types';

interface PipelineFlowGraphProps {
  currentEvent: InferenceEvent | null;
  activeExpert: string | null;
  onHoverNode?: (nodeName: string | null) => void;
}

export const PipelineFlowGraph: React.FC<PipelineFlowGraphProps> = ({
  currentEvent,
  activeExpert,
  onHoverNode,
}) => {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  const isIntervention = currentEvent?.requires_intervention ?? false;
  const isReliable = currentEvent?.is_reliable ?? true;
  const combinedR = currentEvent?.combined_reliability ?? 0.0;
  const globalR = currentEvent?.global_reliability ?? 0.0;
  const localR = currentEvent?.local_reliability ?? 0.0;
  const domain = currentEvent?.predicted_domain ?? 'general';
  const domainConf = currentEvent?.domain_confidence ?? 0.0;
  const selectedExpert = currentEvent?.selected_expert ?? activeExpert ?? 'None';

  const handleMouseEnter = (name: string) => {
    setHoveredNode(name);
    if (onHoverNode) onHoverNode(name);
  };

  const handleMouseLeave = () => {
    setHoveredNode(null);
    if (onHoverNode) onHoverNode(null);
  };

  // Helper for expert node highlighting
  const isExpertActive = (expKey: string) => {
    if (!isIntervention) return false;
    if (selectedExpert === expKey) return true;
    if (selectedExpert.toLowerCase().includes(expKey.toLowerCase())) return true;
    return false;
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col gap-2 relative overflow-hidden">
      {/* Header Caption */}
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
            ARES Neural Computation Architecture
          </span>
          <span className="text-[10px] text-slate-400 font-mono">
            Layer -1 Hidden State (3584d) → Dual Probes → Adaptive Router
          </span>
        </div>

        {/* Real-time Status Badge */}
        {currentEvent && (
          <div className="flex items-center gap-2 text-xs font-mono">
            {isIntervention ? (
              <span className="px-2 py-0.5 rounded bg-rose-50 border border-rose-200 text-rose-700 font-semibold">
                🔴 EXPERT ROUTED ({selectedExpert})
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-700 font-semibold">
                🟢 BASE QWEN RELIABLE (R={combinedR.toFixed(2)} ≥ 0.70)
              </span>
            )}
          </div>
        )}
      </div>

      {/* Main SVG Interactive Neural Canvas */}
      <div className="relative bg-slate-50/60 border border-slate-200/80 rounded-lg p-2 min-h-[380px] flex items-center justify-center">
        <svg className="w-full h-full min-h-[360px]" viewBox="0 0 960 360">
          <defs>
            {/* Glow Filter for Active Elements */}
            <filter id="soft-glow-emerald" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="soft-glow-rose" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* ─── CURVED COMPUTATIONAL FLOW PATHS ─── */}

          {/* 1. Prompt -> Qwen Backbone */}
          <path d="M 90 180 C 130 180, 130 180, 170 180" stroke="#cbd5e1" strokeWidth="2" strokeDasharray="3 3" />

          {/* 2. Qwen -> Layer -1 Representation */}
          <path d="M 260 180 L 320 180" stroke="#6366f1" strokeWidth="2.5" />

          {/* 3. Layer -1 -> GRM Probe (Upper Curve) */}
          <path d="M 380 180 C 420 180, 420 80, 460 80" stroke="#0284c7" strokeWidth="2" fill="none" />

          {/* 4. Layer -1 -> LRM Probe (Lower Curve) */}
          <path d="M 380 180 C 420 180, 420 280, 460 280" stroke="#9333ea" strokeWidth="2" fill="none" />

          {/* 5. GRM & LRM -> R(x) Aggregator */}
          <path d="M 540 80 C 580 80, 580 180, 600 180" stroke="#0284c7" strokeWidth="2" fill="none" />
          <path d="M 540 280 C 580 280, 580 180, 600 180" stroke="#9333ea" strokeWidth="2" fill="none" />

          {/* 6. R(x) -> Router Gate */}
          <path d="M 660 180 L 690 180" stroke="#a855f7" strokeWidth="2.5" />

          {/* 7A. BASE PATH: Router -> BASE Qwen (Top Path) */}
          <path
            d="M 750 180 C 780 180, 780 60, 810 60"
            stroke={!isIntervention ? '#059669' : '#e2e8f0'}
            strokeWidth={!isIntervention ? '3.5' : '1.5'}
            opacity={!isIntervention ? 1.0 : 0.25}
            fill="none"
            filter={!isIntervention ? 'url(#soft-glow-emerald)' : undefined}
          />

          {/* 7B. EXPERT BRANCHES: Router -> Expert Nodes (Bottom Paths) */}
          {/* Router to E0 General (y=160) */}
          <path
            d="M 750 180 C 780 180, 780 160, 810 160"
            stroke={isExpertActive('E0') || isExpertActive('E0_general') ? '#dc2626' : '#e2e8f0'}
            strokeWidth={isExpertActive('E0') || isExpertActive('E0_general') ? '3.5' : '1.5'}
            opacity={isExpertActive('E0') || isExpertActive('E0_general') ? 1.0 : 0.25}
            fill="none"
          />
          {/* Router to E1 Math (y=210) */}
          <path
            d="M 750 180 C 780 180, 780 210, 810 210"
            stroke={isExpertActive('E1') || isExpertActive('E1_math') ? '#d97706' : '#e2e8f0'}
            strokeWidth={isExpertActive('E1') || isExpertActive('E1_math') ? '3.5' : '1.5'}
            opacity={isExpertActive('E1') || isExpertActive('E1_math') ? 1.0 : 0.25}
            fill="none"
          />
          {/* Router to E2 Code (y=260) */}
          <path
            d="M 750 180 C 780 180, 780 260, 810 260"
            stroke={isExpertActive('E2') || isExpertActive('E2_code') ? '#16a34a' : '#e2e8f0'}
            strokeWidth={isExpertActive('E2') || isExpertActive('E2_code') ? '3.5' : '1.5'}
            opacity={isExpertActive('E2') || isExpertActive('E2_code') ? 1.0 : 0.25}
            fill="none"
          />
          {/* Router to E3 Science (y=310) */}
          <path
            d="M 750 180 C 780 180, 780 310, 810 310"
            stroke={isExpertActive('E3') || isExpertActive('E3_science') ? '#2563eb' : '#e2e8f0'}
            strokeWidth={isExpertActive('E3') || isExpertActive('E3_science') ? '3.5' : '1.5'}
            opacity={isExpertActive('E3') || isExpertActive('E3_science') ? 1.0 : 0.25}
            fill="none"
          />

          {/* 8. Convergence to Output Next Token */}
          <path d="M 890 60 L 920 180" stroke={!isIntervention ? '#059669' : '#e2e8f0'} strokeWidth="1.5" opacity="0.6" />
          <path d="M 890 235 L 920 180" stroke={isIntervention ? '#dc2626' : '#e2e8f0'} strokeWidth="1.5" opacity="0.6" />

          {/* Traveling Animated Pulse Particle */}
          {currentEvent && (
            <circle
              cx="920"
              cy="180"
              r="4"
              fill={isIntervention ? '#dc2626' : '#059669'}
              className="animate-ping"
            />
          )}

          {/* ─── NODE COMPONENTS (LIGHT THEMING) ─── */}

          {/* Node 1: Input Prompt */}
          <g
            transform="translate(10, 155)"
            onMouseEnter={() => handleMouseEnter('Prompt')}
            onMouseLeave={handleMouseLeave}
            className="cursor-pointer"
          >
            <rect width="80" height="50" rx="8" fill="#ffffff" stroke="#cbd5e1" strokeWidth="1.5" />
            <text x="40" y="22" textAnchor="middle" fill="#64748b" fontSize="9" fontWeight="600">
              INPUT
            </text>
            <text x="40" y="38" textAnchor="middle" fill="#0f172a" fontSize="11" fontWeight="700">
              Token
            </text>
          </g>

          {/* Node 2: Qwen 2.5-7B Backbone */}
          <g
            transform="translate(170, 150)"
            onMouseEnter={() => handleMouseEnter('Qwen')}
            onMouseLeave={handleMouseLeave}
            className="cursor-pointer"
          >
            <rect width="90" height="60" rx="10" fill="#f8fafc" stroke="#4f46e5" strokeWidth="2" />
            <text x="45" y="25" textAnchor="middle" fill="#4f46e5" fontSize="10" fontWeight="700">
              QWEN 7B
            </text>
            <text x="45" y="42" textAnchor="middle" fill="#475569" fontSize="9" fontWeight="500">
              Frozen Base
            </text>
          </g>

          {/* Node 3: Hidden Representation */}
          <g
            transform="translate(320, 155)"
            onMouseEnter={() => handleMouseEnter('HiddenState')}
            onMouseLeave={handleMouseLeave}
            className="cursor-pointer"
          >
            <rect width="60" height="50" rx="8" fill="#ffffff" stroke="#6366f1" strokeWidth="1.5" />
            <text x="30" y="22" textAnchor="middle" fill="#64748b" fontSize="8" fontWeight="600">
              LAYER -1
            </text>
            <text x="30" y="38" textAnchor="middle" fill="#4f46e5" fontSize="10" fontFamily="mono" fontWeight="700">
              3584d
            </text>
          </g>

          {/* Node 4: GRM Probe (Global Reliability Model) */}
          <g
            transform="translate(460, 55)"
            onMouseEnter={() => handleMouseEnter('GRM')}
            onMouseLeave={handleMouseLeave}
            className="cursor-pointer"
          >
            <rect width="80" height="50" rx="8" fill="#f0f9ff" stroke="#0284c7" strokeWidth="1.5" />
            <text x="40" y="20" textAnchor="middle" fill="#0369a1" fontSize="10" fontWeight="700">
              GRM Probe
            </text>
            <text x="40" y="34" textAnchor="middle" fill="#0c4a6e" fontSize="9" fontFamily="mono">
              R_g = {globalR.toFixed(2)}
            </text>
            <text x="40" y="45" textAnchor="middle" fill="#0284c7" fontSize="8" fontWeight="600">
              {domain} ({(domainConf * 100).toFixed(0)}%)
            </text>
          </g>

          {/* Node 5: LRM Probe (Local Reliability Model) */}
          <g
            transform="translate(460, 255)"
            onMouseEnter={() => handleMouseEnter('LRM')}
            onMouseLeave={handleMouseLeave}
            className="cursor-pointer"
          >
            <rect width="80" height="50" rx="8" fill="#faf5ff" stroke="#9333ea" strokeWidth="1.5" />
            <text x="40" y="20" textAnchor="middle" fill="#7e22ce" fontSize="10" fontWeight="700">
              LRM Probe
            </text>
            <text x="40" y="34" textAnchor="middle" fill="#581c87" fontSize="9" fontFamily="mono">
              R_l = {localR.toFixed(2)}
            </text>
            <text x="40" y="45" textAnchor="middle" fill="#9333ea" fontSize="8" fontWeight="600">
              P(correct)
            </text>
          </g>

          {/* Node 6: Combined R(x) Aggregator Score */}
          <g
            transform="translate(600, 150)"
            onMouseEnter={() => handleMouseEnter('Rx')}
            onMouseLeave={handleMouseLeave}
            className="cursor-pointer"
          >
            <rect
              width="60"
              height="60"
              rx="12"
              fill={isReliable ? '#ecfdf5' : '#fff1f2'}
              stroke={isReliable ? '#10b981' : '#f43f5e'}
              strokeWidth="2"
            />
            <text x="30" y="22" textAnchor="middle" fill="#64748b" fontSize="8" fontWeight="700">
              R(x) Score
            </text>
            <text
              x="30"
              y="42"
              textAnchor="middle"
              fill={isReliable ? '#047857' : '#be123c'}
              fontSize="14"
              fontWeight="800"
              fontFamily="mono"
            >
              {combinedR.toFixed(2)}
            </text>
          </g>

          {/* Node 7: Adaptive Router Gate */}
          <g
            transform="translate(690, 155)"
            onMouseEnter={() => handleMouseEnter('Router')}
            onMouseLeave={handleMouseLeave}
            className="cursor-pointer"
          >
            <rect width="60" height="50" rx="8" fill="#f3e8ff" stroke="#a855f7" strokeWidth="1.5" />
            <text x="30" y="22" textAnchor="middle" fill="#6b21a8" fontSize="9" fontWeight="700">
              ROUTER
            </text>
            <text x="30" y="38" textAnchor="middle" fill="#581c87" fontSize="8" fontWeight="600" textTransform="uppercase">
              {domain}
            </text>
          </g>

          {/* Node 8: BASE Qwen Output Node */}
          <g
            transform="translate(810, 35)"
            onMouseEnter={() => handleMouseEnter('Base')}
            onMouseLeave={handleMouseLeave}
            className="cursor-pointer"
          >
            <rect
              width="80"
              height="50"
              rx="8"
              fill={!isIntervention ? '#d1fae5' : '#ffffff'}
              stroke={!isIntervention ? '#059669' : '#cbd5e1'}
              strokeWidth={!isIntervention ? '2' : '1'}
              opacity={!isIntervention ? 1.0 : 0.4}
            />
            <text
              x="40"
              y="22"
              textAnchor="middle"
              fill={!isIntervention ? '#065f46' : '#64748b'}
              fontSize="10"
              fontWeight="700"
            >
              🟢 BASE
            </text>
            <text
              x="40"
              y="38"
              textAnchor="middle"
              fill={!isIntervention ? '#047857' : '#94a3b8'}
              fontSize="8"
            >
              Qwen Logits
            </text>
          </g>

          {/* ─── 4 EXPERT NODES (E0..E3) ─── */}

          {/* E0: General */}
          <g
            transform="translate(810, 140)"
            onMouseEnter={() => handleMouseEnter('E0')}
            onMouseLeave={handleMouseLeave}
            className="cursor-pointer"
          >
            <rect
              width="80"
              height="38"
              rx="6"
              fill={isExpertActive('E0') || isExpertActive('E0_general') ? '#ffe4e6' : '#ffffff'}
              stroke={isExpertActive('E0') || isExpertActive('E0_general') ? '#e11d48' : '#cbd5e1'}
              strokeWidth={isExpertActive('E0') || isExpertActive('E0_general') ? '2' : '1'}
              opacity={isExpertActive('E0') || isExpertActive('E0_general') ? 1.0 : 0.35}
            />
            <text x="40" y="16" textAnchor="middle" fill="#0f172a" fontSize="9" fontWeight="700">
              E0 General
            </text>
            <text x="40" y="28" textAnchor="middle" fill="#64748b" fontSize="7" fontFamily="mono">
              LoRA r=16
            </text>
          </g>

          {/* E1: Math */}
          <g
            transform="translate(810, 190)"
            onMouseEnter={() => handleMouseEnter('E1')}
            onMouseLeave={handleMouseLeave}
            className="cursor-pointer"
          >
            <rect
              width="80"
              height="38"
              rx="6"
              fill={isExpertActive('E1') || isExpertActive('E1_math') ? '#fef3c7' : '#ffffff'}
              stroke={isExpertActive('E1') || isExpertActive('E1_math') ? '#d97706' : '#cbd5e1'}
              strokeWidth={isExpertActive('E1') || isExpertActive('E1_math') ? '2' : '1'}
              opacity={isExpertActive('E1') || isExpertActive('E1_math') ? 1.0 : 0.35}
            />
            <text x="40" y="16" textAnchor="middle" fill="#0f172a" fontSize="9" fontWeight="700">
              E1 Math
            </text>
            <text x="40" y="28" textAnchor="middle" fill="#64748b" fontSize="7" fontFamily="mono">
              LoRA r=16
            </text>
          </g>

          {/* E2: Code */}
          <g
            transform="translate(810, 240)"
            onMouseEnter={() => handleMouseEnter('E2')}
            onMouseLeave={handleMouseLeave}
            className="cursor-pointer"
          >
            <rect
              width="80"
              height="38"
              rx="6"
              fill={isExpertActive('E2') || isExpertActive('E2_code') ? '#dcfce7' : '#ffffff'}
              stroke={isExpertActive('E2') || isExpertActive('E2_code') ? '#16a34a' : '#cbd5e1'}
              strokeWidth={isExpertActive('E2') || isExpertActive('E2_code') ? '2' : '1'}
              opacity={isExpertActive('E2') || isExpertActive('E2_code') ? 1.0 : 0.35}
            />
            <text x="40" y="16" textAnchor="middle" fill="#0f172a" fontSize="9" fontWeight="700">
              E2 Code
            </text>
            <text x="40" y="28" textAnchor="middle" fill="#64748b" fontSize="7" fontFamily="mono">
              LoRA r=16
            </text>
          </g>

          {/* E3: Science */}
          <g
            transform="translate(810, 290)"
            onMouseEnter={() => handleMouseEnter('E3')}
            onMouseLeave={handleMouseLeave}
            className="cursor-pointer"
          >
            <rect
              width="80"
              height="38"
              rx="6"
              fill={isExpertActive('E3') || isExpertActive('E3_science') ? '#dbeafe' : '#ffffff'}
              stroke={isExpertActive('E3') || isExpertActive('E3_science') ? '#2563eb' : '#cbd5e1'}
              strokeWidth={isExpertActive('E3') || isExpertActive('E3_science') ? '2' : '1'}
              opacity={isExpertActive('E3') || isExpertActive('E3_science') ? 1.0 : 0.35}
            />
            <text x="40" y="16" textAnchor="middle" fill="#0f172a" fontSize="9" fontWeight="700">
              E3 Science
            </text>
            <text x="40" y="28" textAnchor="middle" fill="#64748b" fontSize="7" fontFamily="mono">
              LoRA r=16
            </text>
          </g>

          {/* Node 9: Output Next Token */}
          <g transform="translate(900, 155)">
            <rect width="55" height="50" rx="8" fill="#f8fafc" stroke="#475569" strokeWidth="1.5" />
            <text x="27" y="22" textAnchor="middle" fill="#64748b" fontSize="8" fontWeight="600">
              OUTPUT
            </text>
            <text x="27" y="38" textAnchor="middle" fill="#0f172a" fontSize="10" fontWeight="700">
              Token
            </text>
          </g>
        </svg>
      </div>

      {/* Subtle Hover Tooltip Bar */}
      {hoveredNode && (
        <div className="bg-slate-100 border border-slate-200 rounded px-3 py-1.5 text-xs text-slate-700 flex items-center justify-between font-sans">
          <span>
            {hoveredNode === 'GRM' && 'GRM (Global Reliability Model): Predicts coarse domain feasibility & domain probability distribution.'}
            {hoveredNode === 'LRM' && 'LRM (Local Reliability Model): Evaluates token-level correctness probability P(correct).'}
            {hoveredNode === 'Rx' && 'R(x) Combined Reliability: Weighted score R(x) = w_g*R_g + w_l*R_l evaluated against threshold (0.70).'}
            {hoveredNode === 'Router' && 'Adaptive Router: Routes to BASE model when R(x) ≥ 0.70, or activates domain expert when R(x) < 0.70.'}
            {hoveredNode.startsWith('E') && 'LoRA Expert Adapter: Domain-specialized adapter providing targeted low-rank correction.'}
            {hoveredNode === 'Base' && 'BASE Model: Standard Qwen 2.5-7B backbone forward pass without expert adapter intervention.'}
            {hoveredNode === 'Qwen' && 'Qwen 2.5-7B Backbone: Frozen target causal language model.'}
            {hoveredNode === 'HiddenState' && 'Layer -1 Representation: Pooled 3584-dimensional representation vector extracted at target layer.'}
          </span>
          <span className="text-[10px] text-slate-400 uppercase font-mono">{hoveredNode}</span>
        </div>
      )}
    </div>
  );
};
