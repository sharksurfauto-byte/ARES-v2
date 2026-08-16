import React, { useState, useRef } from 'react';
import type { InferenceEvent } from '../types';
import { Layers, HelpCircle, Cpu, ZoomIn, ZoomOut, RotateCcw, Move } from 'lucide-react';

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
  const [hoveredCell, setHoveredCell] = useState<{ queryIdx: number; keyIdx: number; weight: number } | null>(null);

  // Interactive Pan & Zoom State
  const [scale, setScale] = useState<number>(1.0);
  const [panX, setPanX] = useState<number>(0);
  const [panY, setPanY] = useState<number>(0);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const generatedEvents = events.filter((e) => !e.is_prompt_token);
  const activeEvent = selectedEvent || (generatedEvents.length > 0 ? generatedEvents[generatedEvents.length - 1] : null);

  // Full source token list for active step (prompt tokens + generated tokens so far)
  const sourceTokens = activeEvent?.source_tokens || (events.length > 0 ? events[0]?.source_tokens || events.map(e => e.token) : []);
  const activeTokenIndex = activeEvent ? activeEvent.sequence_position : (sourceTokens.length > 0 ? sourceTokens.length - 1 : 0);

  // Prompt attention matrix from first event
  const promptMatrix = events.length > 0 ? events[0].prompt_attention_matrix : null;
  const promptLen = promptMatrix ? promptMatrix.length : 0;

  // Authoritative PyTorch attention weight lookup (NO FAKE FALLBACKS)
  const getAttentionWeight = (queryIdx: number, keyIdx: number): number => {
    if (keyIdx > queryIdx) return 0.0; // Causal upper triangular mask is zero

    // Case 1: Query is a prompt token
    if (queryIdx < promptLen && promptMatrix && promptMatrix[queryIdx]) {
      return promptMatrix[queryIdx][keyIdx] ?? 0.0;
    }

    // Case 2: Query is a generated token
    const genIdx = queryIdx - promptLen;
    if (genIdx >= 0 && genIdx < generatedEvents.length) {
      const genEvt = generatedEvents[genIdx];
      if (genEvt.attention_weights && keyIdx < genEvt.attention_weights.length) {
        return genEvt.attention_weights[keyIdx];
      }
    }

    return 0.0;
  };

  const N = sourceTokens.length;

  // Pan & Zoom Event Handlers
  const handleZoomIn = () => setScale((prev) => Math.min(3.0, prev + 0.2));
  const handleZoomOut = () => setScale((prev) => Math.max(0.4, prev - 0.2));
  const handleResetZoom = () => {
    setScale(1.0);
    setPanX(0);
    setPanY(0);
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      setScale((prev) => Math.max(0.4, Math.min(3.0, prev + delta)));
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - panX, y: e.clientY - panY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPanX(e.clientX - dragStart.x);
    setPanY(e.clientY - dragStart.y);
  };

  const handleMouseUp = () => setIsDragging(false);

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col gap-3">
      {/* Top Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-indigo-600" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-1.5">
            PyTorch Genuine Self-Attention Matrix
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
              Qwen 2.5-7B
            </span>
          </h2>
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

          {/* Pan & Zoom Controls */}
          <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200">
            <button
              type="button"
              onClick={handleZoomIn}
              className="p-1 text-slate-600 hover:text-indigo-600 hover:bg-white rounded transition-colors"
              title="Zoom In (+)"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={handleZoomOut}
              className="p-1 text-slate-600 hover:text-indigo-600 hover:bg-white rounded transition-colors"
              title="Zoom Out (-)"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={handleResetZoom}
              className="p-1 text-slate-600 hover:text-indigo-600 hover:bg-white rounded transition-colors flex items-center gap-1 px-1.5"
              title="Reset Zoom (100%)"
            >
              <RotateCcw className="w-3 h-3" />
              <span className="text-[10px] font-mono">{Math.round(scale * 100)}%</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Transformer Explainer Interactive Canvas */}
      {!collectAttentions ? (
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 text-center text-xs text-slate-500 flex flex-col items-center gap-2">
          <HelpCircle className="w-5 h-5 text-indigo-400" />
          <span>
            Check "Capture Qwen Attentions" above to stream true self-attention tensors directly from PyTorch during inference.
          </span>
        </div>
      ) : N === 0 ? (
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 text-center text-xs text-slate-400 italic">
          Generate text or select a token to inspect true Qwen self-attention matrix for Layer {attnLayer}, Head {attnHead === -1 ? 'Avg' : attnHead}...
        </div>
      ) : (
        <div
          className="bg-slate-50/70 border border-slate-200 rounded-lg p-4 flex flex-col gap-3 relative overflow-hidden select-none cursor-grab active:cursor-grabbing"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {/* Header Legend */}
          <div className="flex items-center justify-between text-xs text-slate-600 font-mono border-b border-slate-200 pb-2">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1 text-rose-600 font-bold">
                <span className="w-2.5 h-0.5 bg-rose-500 inline-block" /> Key (K) Tokens
              </span>
              <span className="flex items-center gap-1 text-indigo-600 font-bold">
                <span className="w-2.5 h-0.5 bg-indigo-500 inline-block" /> Query (Q) Tokens
              </span>
              <span className="flex items-center gap-1 text-purple-600 font-bold">
                <span className="w-2.5 h-2.5 rounded-full bg-purple-600 inline-block" /> Attention Matrix ({N}×{N})
              </span>
            </div>

            <span className="text-[11px] text-slate-500 font-mono flex items-center gap-2">
              <Move className="w-3 h-3 text-slate-400" /> Drag to pan · Ctrl + Scroll to zoom · Layer {attnLayer} · {attnHead === -1 ? 'Mean Across All Heads' : `Head ${attnHead}`} · {N} Tokens
            </span>
          </div>

          {/* Full Transformer Explainer SVG Grid Canvas with Pan & Zoom Transform Group */}
          <div className="relative min-h-[380px] flex items-center justify-center">
            <svg
              className="w-full h-full min-h-[360px]"
              viewBox={`0 0 ${Math.max(900, 260 + N * 34)} ${Math.max(360, 160 + N * 28)}`}
            >
              <defs>
                <linearGradient id="purple-flow" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#818cf8" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#c084fc" stopOpacity="0.9" />
                </linearGradient>
              </defs>

              {/* Inner Transform Group for Interactive Panning & Zooming */}
              <g transform={`translate(${panX}, ${panY}) scale(${scale})`}>
                {/* ─── LEFT: TOKEN LIST WITH EMBEDDINGS ─── */}
                {sourceTokens.map((tok, idx) => {
                  const y = 140 + idx * 26;
                  const isSelected = activeTokenIndex === idx;

                  return (
                    <g key={`left-tok-${idx}`}>
                      {/* Token Label */}
                      <text
                        x="90"
                        y={y + 4}
                        textAnchor="end"
                        fill={isSelected ? '#4f46e5' : '#334155'}
                        fontSize="10"
                        fontFamily="mono"
                        fontWeight={isSelected ? '700' : '500'}
                      >
                        {tok.length > 12 ? tok.substring(0, 10) + '..' : tok}
                      </text>

                      {/* Embedding Dot */}
                      <circle cx="102" cy={y} r="3" fill={isSelected ? '#4f46e5' : '#94a3b8'} />

                      {/* Ribbon to Q, K, V projections */}
                      <path
                        d={`M 105 ${y} C 130 ${y}, 130 ${140 + idx * 26}, 150 ${140 + idx * 26}`}
                        stroke={isSelected ? '#6366f1' : '#cbd5e1'}
                        strokeWidth={isSelected ? '2' : '1'}
                        opacity={isSelected ? '0.9' : '0.4'}
                        fill="none"
                      />

                      {/* Q / K / V Projection Badge */}
                      <g transform={`translate(150, ${y - 8})`}>
                        <rect width="28" height="16" rx="3" fill="#f1f5f9" stroke="#cbd5e1" strokeWidth="1" />
                        <text x="14" y="11" textAnchor="middle" fill="#64748b" fontSize="8" fontWeight="700">
                          Q K V
                        </text>
                      </g>
                    </g>
                  );
                })}

                {/* ─── KEY TOKENS HEADER (TOP CURVED RED LINES) ─── */}
                {sourceTokens.map((tok, kIdx) => {
                  const kX = 240 + kIdx * 32;
                  return (
                    <g key={`key-col-${kIdx}`}>
                      {/* Curved Red Line from K Projection to Matrix Top */}
                      <path
                        d={`M 178 ${140 + kIdx * 26} C 210 ${140 + kIdx * 26}, 210 70, ${kX} 70`}
                        stroke="#f43f5e"
                        strokeWidth="1.5"
                        opacity="0.6"
                        fill="none"
                      />

                      {/* Top Key Token Text */}
                      <text
                        x={kX}
                        y="65"
                        textAnchor="end"
                        transform={`rotate(-45 ${kX} 65)`}
                        fill="#e11d48"
                        fontSize="9"
                        fontFamily="mono"
                        fontWeight="600"
                      >
                        {tok}
                      </text>
                    </g>
                  );
                })}

                {/* ─── QUERY TOKENS SIDE (LEFT CURVED BLUE LINES) ─── */}
                {sourceTokens.map((tok, qIdx) => {
                  const qY = 100 + qIdx * 26;
                  const isSelectedRow = activeTokenIndex === qIdx;

                  return (
                    <g key={`query-row-${qIdx}`}>
                      {/* Curved Blue Line from Q Projection to Matrix Left */}
                      <path
                        d={`M 178 ${140 + qIdx * 26} C 200 ${140 + qIdx * 26}, 200 ${qY}, 225 ${qY}`}
                        stroke={isSelectedRow ? '#4f46e5' : '#6366f1'}
                        strokeWidth={isSelectedRow ? '2.5' : '1.5'}
                        opacity={isSelectedRow ? '1.0' : '0.5'}
                        fill="none"
                      />

                      {/* Left Query Token Text */}
                      <text
                        x="220"
                        y={qY + 3}
                        textAnchor="end"
                        fill={isSelectedRow ? '#4f46e5' : '#475569'}
                        fontSize="9"
                        fontFamily="mono"
                        fontWeight={isSelectedRow ? '700' : '500'}
                      >
                        {tok}
                      </text>
                    </g>
                  );
                })}

                {/* ─── THE DYNAMIC $N \times N$ TRIANGULAR ATTENTION MATRIX GRID ─── */}
                {sourceTokens.map((_, qIdx) => {
                  const qY = 100 + qIdx * 26;
                  const isSelectedRow = activeTokenIndex === qIdx;

                  return sourceTokens.slice(0, qIdx + 1).map((_, kIdx) => {
                    const kX = 240 + kIdx * 32;

                    // Genuine PyTorch attention weight value A_ij
                    const weight = getAttentionWeight(qIdx, kIdx);

                    // True PyTorch attention score scaling
                    const hasWeight = weight > 0.005;
                    const r = hasWeight ? Math.max(3, Math.min(11, 3 + Math.sqrt(weight) * 9.5)) : 2;
                    const opacity = hasWeight ? Math.max(0.2, Math.min(1.0, weight * 3.0)) : 0.05;
                    const isHovered = hoveredCell?.queryIdx === qIdx && hoveredCell?.keyIdx === kIdx;

                    return (
                      <g
                        key={`cell-${qIdx}-${kIdx}`}
                        className="cursor-pointer transition-all"
                        onMouseEnter={() => setHoveredCell({ queryIdx: qIdx, keyIdx: kIdx, weight })}
                        onMouseLeave={() => setHoveredCell(null)}
                      >
                        {/* Background grid line intersect */}
                        <circle cx={kX} cy={qY} r="11" fill="#ffffff" stroke="#e2e8f0" strokeWidth="1" />

                        {/* Dynamic Genuine Attention Dot */}
                        <circle
                          cx={kX}
                          cy={qY}
                          r={r}
                          fill={hasWeight ? '#4f46e5' : '#cbd5e1'}
                          opacity={opacity}
                        />

                        {/* Active Row or Hover Highlight Ring */}
                        {(isSelectedRow || isHovered) && (
                          <circle
                            cx={kX}
                            cy={qY}
                            r={r + 3}
                            fill="none"
                            stroke={isHovered ? '#4f46e5' : '#818cf8'}
                            strokeWidth="2"
                            className="animate-pulse"
                          />
                        )}
                      </g>
                    );
                  });
                })}

                {/* ─── RIGHT: OUT / MLP FLUID FLOW RIBBONS ─── */}
                {sourceTokens.map((_, idx) => {
                  const qY = 100 + idx * 26;
                  const outX = 240 + idx * 32 + 20;

                  return (
                    <path
                      key={`out-flow-${idx}`}
                      d={`M ${outX} ${qY} C ${outX + 60} ${qY}, ${outX + 60} 180, ${240 + N * 32 + 60} 180`}
                      stroke="url(#purple-flow)"
                      strokeWidth="1.5"
                      opacity="0.35"
                      fill="none"
                    />
                  );
                })}

                {/* MLP / Output Container */}
                <g transform={`translate(${240 + N * 32 + 60}, 140)`}>
                  <rect width="70" height="80" rx="10" fill="#faf5ff" stroke="#c084fc" strokeWidth="2" />
                  <text x="35" y="32" textAnchor="middle" fill="#7e22ce" fontSize="10" fontWeight="700">
                    MLP Block
                  </text>
                  <text x="35" y="52" textAnchor="middle" fill="#9333ea" fontSize="8" fontFamily="mono">
                    Weighted Out
                  </text>
                </g>

                {/* Output Arrow to ARES Representation */}
                <path
                  d={`M ${240 + N * 32 + 130} 180 L ${240 + N * 32 + 170} 180`}
                  stroke="#a855f7"
                  strokeWidth="2.5"
                />
                <circle cx={240 + N * 32 + 170} cy="180" r="4" fill="#a855f7" />
              </g>
            </svg>
          </div>

          {/* Interactive Cell Hover Detail Tooltip */}
          {hoveredCell ? (
            <div className="bg-indigo-50 border border-indigo-200 rounded p-2.5 text-xs text-indigo-900 flex items-center justify-between font-sans">
              <span>
                Query Token: <strong className="font-mono">"{sourceTokens[hoveredCell.queryIdx]}"</strong> attends to Key Token: <strong className="font-mono">"{sourceTokens[hoveredCell.keyIdx]}"</strong>
              </span>
              <span className="font-mono font-bold text-indigo-700 bg-white px-2 py-0.5 rounded border border-indigo-200">
                PyTorch Attention A_{hoveredCell.queryIdx},{hoveredCell.keyIdx} = {(hoveredCell.weight * 100).toFixed(2)}%
              </span>
            </div>
          ) : (
            <div className="bg-slate-100 border border-slate-200 rounded p-2 text-xs text-slate-500 font-sans flex items-center justify-between">
              <span>💡 Drag to pan · Scroll mouse wheel to zoom in/out · Hover any dot to inspect attention probabilities.</span>
              <span className="font-mono text-[10px] text-slate-400">100% Genuine PyTorch Tensors</span>
            </div>
          )}
        </div>
      )}

      {/* Explicit Bridge Connection from Qwen Attention -> ARES Routing */}
      <div className="bg-slate-100 border border-slate-200 rounded p-2.5 text-[11px] font-mono text-slate-600 flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-slate-700">
          <Cpu className="w-3.5 h-3.5 text-indigo-600" />
          <span>Qwen Self-Attention Matrix ({N}×{N})</span>
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
