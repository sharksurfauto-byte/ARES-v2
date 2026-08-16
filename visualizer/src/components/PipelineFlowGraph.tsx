import React from 'react';
import type { InferenceEvent } from '../types';
import { Network, Activity, Zap, CheckCircle2, AlertTriangle, ArrowRight } from 'lucide-react';

interface PipelineFlowGraphProps {
  currentEvent: InferenceEvent | null;
  activeExpert: string | null;
}

export const PipelineFlowGraph: React.FC<PipelineFlowGraphProps> = ({
  currentEvent,
  activeExpert,
}) => {
  const isIntervention = currentEvent?.requires_intervention ?? false;
  const isReliable = currentEvent?.is_reliable ?? true;
  const combinedR = currentEvent?.combined_reliability ?? 0.0;
  const globalR = currentEvent?.global_reliability ?? 0.0;
  const localR = currentEvent?.local_reliability ?? 0.0;
  const domain = currentEvent?.predicted_domain ?? 'general';
  const expertName = currentEvent?.selected_expert ?? activeExpert ?? 'None';

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <Network className="w-4 h-4 text-indigo-400" /> ARES Dynamic Pipeline Flow Graph
        </h2>
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="flex items-center gap-1 text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" /> BASE Path
          </span>
          <span className="flex items-center gap-1 text-rose-400">
            <span className="w-2 h-2 rounded-full bg-rose-400 inline-block" /> EXPERT Path
          </span>
        </div>
      </div>

      {/* SVG Pipeline Graph */}
      <div className="relative bg-slate-950/90 border border-slate-800/80 rounded-xl p-4 overflow-hidden min-h-[300px] flex items-center justify-center">
        {/* Background Grid Accent */}
        <div className="absolute inset-0 bg-[radial-gradient(#334155_1px,transparent_1px)] [background-size:16px_16px] opacity-20" />

        <svg className="w-full h-full min-h-[260px] relative z-10" viewBox="0 0 900 240">
          <defs>
            {/* Glow Filters */}
            <filter id="glow-emerald" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="glow-rose" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            {/* Gradient Paths */}
            <linearGradient id="grad-base" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#34d399" stopOpacity="1" />
            </linearGradient>
            <linearGradient id="grad-expert" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#fb7185" stopOpacity="1" />
            </linearGradient>
          </defs>

          {/* Connection Lines */}
          {/* Node 1 (Prompt) -> Node 2 (Qwen 7B) */}
          <path d="M 90 120 L 160 120" stroke="#475569" strokeWidth="2" strokeDasharray="4 4" />

          {/* Node 2 (Qwen) -> Layer -1 Representation */}
          <path d="M 240 120 L 300 120" stroke="#6366f1" strokeWidth="2.5" />

          {/* Layer -1 -> GRM Probe (top branch) */}
          <path d="M 360 120 C 390 120, 390 60, 420 60" stroke="#38bdf8" strokeWidth="2" fill="none" />

          {/* Layer -1 -> LRM Probe (bottom branch) */}
          <path d="M 360 120 C 390 120, 390 180, 420 180" stroke="#c084fc" strokeWidth="2" fill="none" />

          {/* Probes -> R(x) Aggregator */}
          <path d="M 500 60 C 530 60, 530 120, 550 120" stroke="#38bdf8" strokeWidth="2" fill="none" />
          <path d="M 500 180 C 530 180, 530 120, 550 120" stroke="#c084fc" strokeWidth="2" fill="none" />

          {/* R(x) -> Router */}
          <path d="M 610 120 L 650 120" stroke="#a855f7" strokeWidth="2.5" />

          {/* Router -> BASE Path (top output) */}
          <path
            d="M 710 120 C 730 120, 730 60, 750 60"
            stroke={!isIntervention ? '#10b981' : '#334155'}
            strokeWidth={!isIntervention ? '3' : '1.5'}
            fill="none"
            filter={!isIntervention ? 'url(#glow-emerald)' : undefined}
          />

          {/* Router -> EXPERT Path (bottom output) */}
          <path
            d="M 710 120 C 730 120, 730 180, 750 180"
            stroke={isIntervention ? '#f43f5e' : '#334155'}
            strokeWidth={isIntervention ? '3' : '1.5'}
            fill="none"
            filter={isIntervention ? 'url(#glow-rose)' : undefined}
          />

          {/* Output Convergence to Next Token */}
          <path d="M 830 60 L 870 120" stroke={!isIntervention ? '#10b981' : '#334155'} strokeWidth="2" />
          <path d="M 830 180 L 870 120" stroke={isIntervention ? '#f43f5e' : '#334155'} strokeWidth="2" />

          {/* Traveling Pulse Particle */}
          {currentEvent && (
            <circle
              cx="870"
              cy="120"
              r="5"
              fill={isIntervention ? '#f43f5e' : '#10b981'}
              className="animate-ping"
            />
          )}

          {/* ─── NODE OVERLAYS ─── */}
          {/* Node 1: Input Tokens */}
          <g transform="translate(10, 95)">
            <rect width="80" height="50" rx="8" fill="#1e293b" stroke="#334155" strokeWidth="1.5" />
            <text x="40" y="25" textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="600">
              INPUT
            </text>
            <text x="40" y="40" textAnchor="middle" fill="#e2e8f0" fontSize="11" fontWeight="700">
              Prompt
            </text>
          </g>

          {/* Node 2: Qwen 2.5-7B */}
          <g transform="translate(160, 90)">
            <rect width="80" height="60" rx="10" fill="#1e1b4b" stroke="#6366f1" strokeWidth="2" />
            <text x="40" y="26" textAnchor="middle" fill="#818cf8" fontSize="10" fontWeight="600">
              BACKBONE
            </text>
            <text x="40" y="44" textAnchor="middle" fill="#ffffff" fontSize="12" fontWeight="700">
              Qwen 7B
            </text>
          </g>

          {/* Node 3: Hidden Rep (Layer -1) */}
          <g transform="translate(300, 95)">
            <rect width="60" height="50" rx="8" fill="#0f172a" stroke="#6366f1" strokeWidth="1.5" />
            <text x="30" y="24" textAnchor="middle" fill="#94a3b8" fontSize="9" fontWeight="600">
              LAYER -1
            </text>
            <text x="30" y="38" textAnchor="middle" fill="#818cf8" fontSize="10" fontFam="mono">
              3584d
            </text>
          </g>

          {/* Top Branch: GRM Probe */}
          <g transform="translate(420, 35)">
            <rect width="80" height="50" rx="8" fill="#0c4a6e" stroke="#38bdf8" strokeWidth="1.5" />
            <text x="40" y="22" textAnchor="middle" fill="#7dd3fc" fontSize="10" fontWeight="700">
              GRM Probe
            </text>
            <text x="40" y="38" textAnchor="middle" fill="#ffffff" fontSize="10" fontFamily="mono">
              R_g={globalR.toFixed(2)}
            </text>
          </g>

          {/* Bottom Branch: LRM Probe */}
          <g transform="translate(420, 155)">
            <rect width="80" height="50" rx="8" fill="#4c1d95" stroke="#c084fc" strokeWidth="1.5" />
            <text x="40" y="22" textAnchor="middle" fill="#e9d5ff" fontSize="10" fontWeight="700">
              LRM Probe
            </text>
            <text x="40" y="38" textAnchor="middle" fill="#ffffff" fontSize="10" fontFamily="mono">
              R_l={localR.toFixed(2)}
            </text>
          </g>

          {/* Reliability Aggregator R(x) */}
          <g transform="translate(550, 90)">
            <rect
              width="60"
              height="60"
              rx="12"
              fill={isReliable ? '#064e3b' : '#881337'}
              stroke={isReliable ? '#10b981' : '#f43f5e'}
              strokeWidth="2"
            />
            <text x="30" y="24" textAnchor="middle" fill="#94a3b8" fontSize="9" fontWeight="600">
              R(x) Score
            </text>
            <text
              x="30"
              y="44"
              textAnchor="middle"
              fill={isReliable ? '#34d399' : '#fb7185'}
              fontSize="14"
              fontWeight="800"
              fontFamily="mono"
            >
              {combinedR.toFixed(2)}
            </text>
          </g>

          {/* Router Gate */}
          <g transform="translate(650, 95)">
            <rect width="60" height="50" rx="8" fill="#3b0764" stroke="#a855f7" strokeWidth="1.5" />
            <text x="30" y="24" textAnchor="middle" fill="#d8b4fe" fontSize="9" fontWeight="700">
              ROUTER
            </text>
            <text x="30" y="38" textAnchor="middle" fill="#ffffff" fontSize="9" textTransform="uppercase">
              {domain}
            </text>
          </g>

          {/* BASE Node */}
          <g transform="translate(750, 35)">
            <rect
              width="80"
              height="50"
              rx="8"
              fill={!isIntervention ? '#065f46' : '#1e293b'}
              stroke={!isIntervention ? '#34d399' : '#475569'}
              strokeWidth={!isIntervention ? '2' : '1'}
            />
            <text
              x="40"
              y="22"
              textAnchor="middle"
              fill={!isIntervention ? '#a7f3d0' : '#64748b'}
              fontSize="10"
              fontWeight="700"
            >
              🟢 BASE QWEN
            </text>
            <text
              x="40"
              y="38"
              textAnchor="middle"
              fill={!isIntervention ? '#ffffff' : '#64748b'}
              fontSize="9"
            >
              No Adapter
            </text>
          </g>

          {/* EXPERT Node */}
          <g transform="translate(750, 155)">
            <rect
              width="80"
              height="50"
              rx="8"
              fill={isIntervention ? '#9f1239' : '#1e293b'}
              stroke={isIntervention ? '#fb7185' : '#475569'}
              strokeWidth={isIntervention ? '2' : '1'}
            />
            <text
              x="40"
              y="22"
              textAnchor="middle"
              fill={isIntervention ? '#fecdd3' : '#64748b'}
              fontSize="10"
              fontWeight="700"
            >
              🔴 EXPERT
            </text>
            <text
              x="40"
              y="38"
              textAnchor="middle"
              fill={isIntervention ? '#ffffff' : '#64748b'}
              fontSize="9"
              fontFamily="mono"
            >
              {expertName}
            </text>
          </g>
        </svg>
      </div>

      {/* Active Pipeline Status Banner */}
      <div className="flex items-center justify-between bg-slate-950/80 p-3 rounded-xl border border-slate-800 text-xs font-mono">
        <div className="flex items-center gap-2">
          <span className="text-slate-400">Current Decision:</span>
          {isIntervention ? (
            <span className="flex items-center gap-1.5 text-rose-400 font-bold px-2 py-0.5 rounded bg-rose-950/80 border border-rose-800/80">
              <AlertTriangle className="w-3.5 h-3.5" /> EXPERT INTERVENTION ({expertName})
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-emerald-400 font-bold px-2 py-0.5 rounded bg-emerald-950/80 border border-emerald-800/80">
              <CheckCircle2 className="w-3.5 h-3.5" /> BASE QWEN RELIABLE (R={combinedR.toFixed(2)} ≥ 0.70)
            </span>
          )}
        </div>
        <div className="text-slate-400 flex items-center gap-2">
          <span>Predicted Domain: <strong className="text-indigo-300 uppercase">{domain}</strong></span>
          <span>•</span>
          <span>Latency: <strong className="text-slate-200">{currentEvent?.total_latency_ms.toFixed(1) ?? 0} ms</strong></span>
        </div>
      </div>
    </div>
  );
};
