import React from 'react';
import type { TelemetrySnapshot } from '../types';
import { BarChart3, Gauge, Cpu, Percent } from 'lucide-react';

interface TelemetryGaugesProps {
  snapshot: TelemetrySnapshot | null;
}

export const TelemetryGauges: React.FC<TelemetryGaugesProps> = ({ snapshot }) => {
  if (!snapshot) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl text-center text-slate-500 text-xs py-8">
        Run generation stream to populate execution telemetry gauges...
      </div>
    );
  }

  const {
    tokens_generated,
    expert_activations,
    expert_compute_percentage,
    expert_activation_reduction_vs_always_on,
    average_reliability,
    average_routing_latency_ms,
    average_expert_latency_ms,
    domain_distribution,
  } = snapshot;

  const savingsPct = (expert_activation_reduction_vs_always_on * 100).toFixed(1);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-5">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-indigo-400" /> ARES Research Telemetry & Compute Savings
        </h2>
        <span className="text-xs text-emerald-400 font-bold bg-emerald-950/80 border border-emerald-800/80 px-2.5 py-1 rounded-full">
          {savingsPct}% Compute Reduction vs Always-On
        </span>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5 flex flex-col justify-between">
          <span className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" /> Tokens Generated
          </span>
          <span className="text-xl font-bold font-mono text-white mt-1">{tokens_generated}</span>
        </div>

        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5 flex flex-col justify-between">
          <span className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
            <Percent className="w-3.5 h-3.5 text-rose-400" /> Expert Rate
          </span>
          <span className="text-xl font-bold font-mono text-rose-300 mt-1">
            {expert_compute_percentage.toFixed(1)}%
          </span>
          <span className="text-[10px] text-slate-500 font-mono">({expert_activations} tokens)</span>
        </div>

        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5 flex flex-col justify-between">
          <span className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
            <BarChart3 className="w-3.5 h-3.5 text-emerald-400" /> Savings
          </span>
          <span className="text-xl font-bold font-mono text-emerald-300 mt-1">{savingsPct}%</span>
          <span className="text-[10px] text-slate-500 font-mono">vs Always-Expert</span>
        </div>

        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5 flex flex-col justify-between">
          <span className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
            <Gauge className="w-3.5 h-3.5 text-sky-400" /> Mean R(x) Score
          </span>
          <span className="text-xl font-bold font-mono text-sky-300 mt-1">
            {average_reliability.toFixed(3)}
          </span>
        </div>
      </div>

      {/* Domain Probabilities & Latency Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* GRM Domain Distribution */}
        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 flex flex-col gap-2.5">
          <span className="text-xs font-semibold text-slate-300 block">
            GRM Domain Classification Distribution
          </span>
          {Object.entries(domain_distribution).map(([domain, count]) => {
            const pct = tokens_generated > 0 ? (count / tokens_generated) * 100 : 0;
            return (
              <div key={domain} className="flex flex-col gap-1 text-xs">
                <div className="flex justify-between font-mono">
                  <span className="text-slate-400 capitalize">{domain}</span>
                  <span className="text-slate-200">{pct.toFixed(1)}% ({count})</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-indigo-500 to-indigo-400 h-full rounded-full transition-all duration-500"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Latency Breakdown */}
        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 flex flex-col justify-between gap-3">
          <span className="text-xs font-semibold text-slate-300 block">
            Latency & Compute Overhead Breakdown
          </span>
          <div className="flex flex-col gap-2 font-mono text-xs">
            <div className="flex justify-between items-center bg-slate-900 p-2.5 rounded-lg border border-slate-800">
              <span className="text-slate-400">Routing Latency (GRM+LRM):</span>
              <span className="text-sky-300 font-bold">{average_routing_latency_ms.toFixed(1)} ms</span>
            </div>
            <div className="flex justify-between items-center bg-slate-900 p-2.5 rounded-lg border border-slate-800">
              <span className="text-slate-400">Expert Adapter Pass:</span>
              <span className="text-rose-300 font-bold">{average_expert_latency_ms.toFixed(1)} ms</span>
            </div>
          </div>
          <p className="text-[11px] text-slate-500 italic">
            Zero-copy two-passSemantics ensures base logits are reused when R(x) ≥ 0.70.
          </p>
        </div>
      </div>
    </div>
  );
};
