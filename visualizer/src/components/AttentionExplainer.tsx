import React, { useState } from 'react';
import type { InferenceEvent } from '../types';
import { Eye, Grid, Network, Layers, Cpu, HelpCircle } from 'lucide-react';

interface AttentionExplainerProps {
  selectedEvent: InferenceEvent | null;
  events: InferenceEvent[];
  attnLayer: number;
  setAttnLayer: (n: number) => void;
  attnHead: number;
  setAttnHead: (n: number) => void;
  collectAttentions: boolean;
  setCollectAttentions: (b: boolean) => void;
}

export const AttentionExplainer: React.FC<AttentionExplainerProps> = ({
  selectedEvent,
  events,
  attnLayer,
  setAttnLayer,
  attnHead,
  setAttnHead,
  collectAttentions,
  setCollectAttentions,
}) => {
  const [viewMode, setViewMode] = useState<'arcs' | 'matrix'>('arcs');

  const generatedEvents = events.filter((e) => !e.is_prompt_token);
  const activeEvent = selectedEvent || (generatedEvents.length > 0 ? generatedEvents[generatedEvents.length - 1] : null);

  const attnWeights = activeEvent?.attention_weights ?? null;
  const sourceTokens = activeEvent?.source_tokens ?? null;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col gap-3">
      {/* Top Header & Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-indigo-600" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800">
            Qwen 2.5-7B Self-Attention Interpretability
          </h2>
          <span className="text-[10px] text-slate-400 font-mono">
            softmax(QK^T / √d_k)
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs">
          {/* Collect Attentions Toggle */}
          <label className="flex items-center gap-1.5 cursor-pointer font-medium text-slate-700 bg-slate-50 border border-slate-200 px-2.5 py-1 rounded-lg hover:bg-slate-100 transition-colors">
            <input
              type="checkbox"
              checked={collectAttentions}
              onChange={(e) => setCollectAttentions(e.target.checked)}
              className="accent-indigo-600 rounded cursor-pointer"
            />
            <span className="text-[11px]">Capture Qwen Attentions</span>
          </label>

          {/* Layer Selector */}
          <div className="flex items-center gap-1">
            <span className="text-[11px] text-slate-500 font-medium">Layer:</span>
            <button
              type="button"
              onClick={() => setAttnLayer(Math.max(0, attnLayer - 1))}
              className="w-5 h-5 bg-slate-100 hover:bg-slate-200 rounded font-bold text-slate-600 flex items-center justify-center text-xs"
            >
              -
            </button>
            <span className="font-mono text-xs font-bold w-6 text-center text-indigo-700">{attnLayer}</span>
            <button
              type="button"
              onClick={() => setAttnLayer(Math.min(27, attnLayer + 1))}
              className="w-5 h-5 bg-slate-100 hover:bg-slate-200 rounded font-bold text-slate-600 flex items-center justify-center text-xs"
            >
              +
            </button>
          </div>

          {/* Head Selector */}
          <div className="flex items-center gap-1">
            <span className="text-[11px] text-slate-500 font-medium">Head:</span>
            <select
              value={attnHead}
              onChange={(e) => setAttnHead(Number(e.target.value))}
              className="bg-white border border-slate-200 rounded px-2 py-0.5 text-xs text-indigo-700 font-mono font-bold focus:outline-none"
            >
              <option value={-1}>Average All Heads</option>
              {Array.from({ length: 28 }, (_, i) => (
                <option key={i} value={i}>Head {i}</option>
              ))}
            </select>
          </div>

          {/* View Mode Buttons */}
          <div className="flex items-center bg-slate-100 p-0.5 rounded-lg border border-slate-200">
            <button
              type="button"
              onClick={() => setViewMode('arcs')}
              className={`px-2 py-0.5 rounded text-[11px] font-semibold flex items-center gap-1 transition-colors ${
                viewMode === 'arcs' ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-500 hover:text-slate-900'
              }`}
            >
              <Network className="w-3 h-3" /> Arcs
            </button>
            <button
              type="button"
              onClick={() => setViewMode('matrix')}
              className={`px-2 py-0.5 rounded text-[11px] font-semibold flex items-center gap-1 transition-colors ${
                viewMode === 'matrix' ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-500 hover:text-slate-900'
              }`}
            >
              <Grid className="w-3 h-3" /> Matrix
            </button>
          </div>
        </div>
      </div>

      {/* Main Attention Display Area */}
      {!collectAttentions ? (
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 text-center text-xs text-slate-500 flex flex-col items-center gap-2">
          <HelpCircle className="w-5 h-5 text-indigo-400" />
          <span>
            Check "Capture Qwen Attentions" above to stream true self-attention tensors directly from PyTorch during inference.
          </span>
        </div>
      ) : !activeEvent || !attnWeights || !sourceTokens ? (
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 text-center text-xs text-slate-400 italic">
          Generate text or select a token to inspect true Qwen self-attention weights for Layer {attnLayer}, Head {attnHead === -1 ? 'Avg' : attnHead}...
        </div>
      ) : viewMode === 'arcs' ? (
        /* Token-to-Token Arc Visualizer */
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 flex flex-col gap-3 relative overflow-hidden">
          <div className="flex items-center justify-between text-xs text-slate-600 font-mono border-b border-slate-200 pb-2">
            <span>
              Inspecting Token: <strong className="text-indigo-700 font-mono font-bold">"{activeEvent.token}"</strong>
            </span>
            <span>
              Layer {attnLayer} · {attnHead === -1 ? 'Mean Across All Heads' : `Head ${attnHead}`}
            </span>
          </div>

          {/* SVG Arc Diagram */}
          <div className="relative min-h-[160px] flex items-center justify-center">
            <svg className="w-full h-full min-h-[140px]" viewBox="0 0 800 140">
              {/* Render Token Boxes along bottom line */}
              {sourceTokens.map((tok, idx) => {
                const weight = attnWeights[idx] || 0.0;
                const totalToks = sourceTokens.length;
                const step = 720 / Math.max(1, totalToks - 1);
                const startX = 40 + idx * step;
                const activeX = 40 + (totalToks - 1) * step;

                const isTarget = idx === totalToks - 1;
                const opacity = Math.max(0.1, Math.min(1.0, weight * 4));
                const strokeWidth = Math.max(0.5, Math.min(4.5, weight * 8));

                return (
                  <g key={`${tok}-${idx}`}>
                    {/* Connection Arc to Active Token */}
                    {!isTarget && weight > 0.01 && (
                      <path
                        d={`M ${startX} 110 C ${startX} ${110 - Math.min(90, Math.abs(activeX - startX) * 0.35)}, ${activeX} ${110 - Math.min(90, Math.abs(activeX - startX) * 0.35)}, ${activeX} 110`}
                        stroke="#4f46e5"
                        strokeWidth={strokeWidth}
                        opacity={opacity}
                        fill="none"
                      />
                    )}

                    {/* Weight percentage label over prominent tokens */}
                    {!isTarget && weight > 0.12 && (
                      <text
                        x={(startX + activeX) / 2}
                        y={105 - Math.min(75, Math.abs(activeX - startX) * 0.35)}
                        textAnchor="middle"
                        fill="#4f46e5"
                        fontSize="9"
                        fontWeight="700"
                        fontFamily="mono"
                      >
                        {(weight * 100).toFixed(0)}%
                      </text>
                    )}

                    {/* Token Box */}
                    <rect
                      x={startX - 18}
                      y="110"
                      width="36"
                      height="24"
                      rx="4"
                      fill={isTarget ? '#4f46e5' : weight > 0.15 ? '#e0e7ff' : '#ffffff'}
                      stroke={isTarget ? '#4f46e5' : weight > 0.15 ? '#6366f1' : '#cbd5e1'}
                      strokeWidth={isTarget ? '2' : '1'}
                    />
                    <text
                      x={startX}
                      y="125"
                      textAnchor="middle"
                      fill={isTarget ? '#ffffff' : '#0f172a'}
                      fontSize="9"
                      fontFamily="mono"
                      fontWeight="600"
                    >
                      {tok.trim() || tok}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Top Attended Tokens Bar */}
          <div className="flex items-center gap-2 text-xs font-mono bg-white p-2 rounded border border-slate-200">
            <span className="text-slate-400 text-[10px]">Top Attention Targets:</span>
            {sourceTokens
              .map((tok, i) => ({ tok, w: attnWeights[i] || 0 }))
              .sort((a, b) => b.w - a.w)
              .slice(0, 4)
              .map((item, idx) => (
                <span
                  key={idx}
                  className="px-2 py-0.5 rounded bg-indigo-50 border border-indigo-100 text-indigo-800 font-semibold"
                >
                  "{item.tok}" ({(item.w * 100).toFixed(1)}%)
                </span>
              ))}
          </div>
        </div>
      ) : (
        /* Self-Attention Matrix View */
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 flex flex-col gap-2">
          <span className="text-xs font-semibold text-slate-700">
            Layer {attnLayer} Attention Vector for Token "{activeEvent.token}"
          </span>
          <div className="flex flex-wrap gap-1">
            {sourceTokens.map((tok, idx) => {
              const w = attnWeights[idx] || 0;
              const intensity = Math.min(100, Math.round(w * 100 * 2.5));
              return (
                <div
                  key={idx}
                  className="flex flex-col items-center p-1.5 rounded border border-slate-200 text-[10px] font-mono"
                  style={{ backgroundColor: `rgba(99, 102, 241, ${Math.max(0.05, w)})` }}
                >
                  <span className="font-bold text-slate-900">{tok}</span>
                  <span className="text-slate-600">{(w * 100).toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Explicit Bridge Connection from Qwen Attention -> ARES Routing */}
      <div className="bg-slate-100 border border-slate-200 rounded p-2.5 text-[11px] font-mono text-slate-600 flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-slate-700">
          <Cpu className="w-3.5 h-3.5 text-indigo-600" />
          <span>Qwen Self-Attention</span>
        </span>
        <span>→</span>
        <span className="text-indigo-700 font-bold">Layer -1 Rep (3584d)</span>
        <span>→</span>
        <span className="text-sky-700 font-bold">GRM + LRM Probes</span>
        <span>→</span>
        <span className="text-slate-800 font-bold">R(x) Score ({activeEvent?.combined_reliability.toFixed(2) ?? 0})</span>
        <span>→</span>
        <span className={activeEvent?.requires_intervention ? 'text-rose-700 font-bold' : 'text-emerald-700 font-bold'}>
          {activeEvent?.requires_intervention ? `EXPERT (${activeEvent.selected_expert})` : 'BASE QWEN'}
        </span>
      </div>
    </div>
  );
};
