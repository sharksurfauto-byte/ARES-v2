import React from 'react';
import type { InferenceEvent, TelemetrySnapshot } from '../types';

interface TelemetryGaugesProps {
  snapshot: TelemetrySnapshot | null;
  selectedEvent: InferenceEvent | null;
}

export const TelemetryGauges: React.FC<TelemetryGaugesProps> = ({
  snapshot,
  selectedEvent,
}) => {
  return (
    <div className="flex flex-col gap-4">
      {/* Token Inspector Panel (Compact Side Inspector) */}
      {selectedEvent ? (
        <div className="bg-white border border-indigo-200 rounded-xl p-4 shadow-sm flex flex-col gap-2.5">
          <div className="flex items-center justify-between border-b border-indigo-100 pb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-indigo-900 flex items-center gap-1.5">
              🔍 Token Inspector
            </span>
            <span className="text-xs font-mono font-bold bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded text-indigo-700">
              "{selectedEvent.token}"
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-sans">
            <div className="bg-slate-50 p-2 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 font-medium block">Combined R(x) Score</span>
              <span
                className={`font-mono font-bold ${
                  selectedEvent.is_reliable ? 'text-emerald-700' : 'text-rose-700'
                }`}
              >
                {selectedEvent.combined_reliability.toFixed(3)}
              </span>
            </div>

            <div className="bg-slate-50 p-2 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 font-medium block">Predicted Domain</span>
              <span className="font-mono font-bold text-slate-800 uppercase">
                {selectedEvent.predicted_domain} ({(selectedEvent.domain_confidence * 100).toFixed(0)}%)
              </span>
            </div>

            <div className="bg-slate-50 p-2 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 font-medium block">Routing Decision</span>
              <span
                className={`font-mono font-bold ${
                  selectedEvent.requires_intervention ? 'text-rose-700' : 'text-emerald-700'
                }`}
              >
                {selectedEvent.requires_intervention ? 'EXPERT' : 'BASE'}
              </span>
            </div>

            <div className="bg-slate-50 p-2 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 font-medium block">Selected Expert</span>
              <span className="font-mono font-bold text-slate-800">
                {selectedEvent.selected_expert || 'None (Base Qwen)'}
              </span>
            </div>

            <div className="bg-slate-50 p-2 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 font-medium block">Global Feasibility R_g</span>
              <span className="font-mono text-slate-700">{selectedEvent.global_reliability.toFixed(3)}</span>
            </div>

            <div className="bg-slate-50 p-2 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 font-medium block">Local Correctness R_l</span>
              <span className="font-mono text-slate-700">{selectedEvent.local_reliability.toFixed(3)}</span>
            </div>
          </div>

          <div className="flex justify-between items-center text-[10px] font-mono text-slate-500 bg-slate-50 px-2.5 py-1.5 rounded border border-slate-200 mt-1">
            <span>Routing: {selectedEvent.routing_latency_ms.toFixed(1)} ms</span>
            <span>Expert: {selectedEvent.expert_latency_ms.toFixed(1)} ms</span>
            <span className="font-bold text-slate-700">Total: {selectedEvent.total_latency_ms.toFixed(1)} ms</span>
          </div>
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm text-center text-xs text-slate-400 py-6 italic font-sans">
          Click any generated token to open its detailed inspector view...
        </div>
      )}

      {/* Aggregate Telemetry Summary (Light & Restrained) */}
      {snapshot && (
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col gap-3">
          <div className="flex items-center justify-between border-b border-slate-100 pb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
              ARES Execution Telemetry
            </span>
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">
              {(snapshot.expert_activation_reduction_vs_always_on * 100).toFixed(1)}% Compute Savings
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2 font-sans text-xs">
            <div className="bg-slate-50 p-2 rounded border border-slate-200 text-center">
              <span className="text-[10px] text-slate-500 block">Total Tokens</span>
              <span className="font-mono font-bold text-slate-800">{snapshot.tokens_generated}</span>
            </div>
            <div className="bg-slate-50 p-2 rounded border border-slate-200 text-center">
              <span className="text-[10px] text-slate-500 block">Expert Rate</span>
              <span className="font-mono font-bold text-rose-700">{snapshot.expert_compute_percentage.toFixed(1)}%</span>
            </div>
            <div className="bg-slate-50 p-2 rounded border border-slate-200 text-center">
              <span className="text-[10px] text-slate-500 block">Mean R(x)</span>
              <span className="font-mono font-bold text-sky-700">{snapshot.average_reliability.toFixed(3)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
