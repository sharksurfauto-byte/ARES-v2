import React from 'react';
import type { InferenceEvent } from '../types';
import { FileText, Eye, CheckCircle2, AlertTriangle } from 'lucide-react';

interface TokenStreamProps {
  events: InferenceEvent[];
  selectedTokenIndex: number | null;
  onSelectToken: (idx: number) => void;
  isGenerating: boolean;
}

export const TokenStream: React.FC<TokenStreamProps> = ({
  events,
  selectedTokenIndex,
  onSelectToken,
  isGenerating,
}) => {
  const generatedEvents = events.filter((e) => !e.is_prompt_token);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <FileText className="w-4 h-4 text-indigo-400" /> Interactive Token Output Stream
        </h2>
        <span className="text-xs text-slate-400 font-mono">
          {generatedEvents.length} tokens generated {isGenerating && '• Streaming...'}
        </span>
      </div>

      {/* Raw Response Container */}
      <div className="bg-slate-950/90 border border-slate-800/80 rounded-xl p-4 min-h-[100px] font-mono text-sm text-slate-100 whitespace-pre-wrap leading-relaxed">
        {generatedEvents.length > 0 ? (
          generatedEvents.map((e, idx) => {
            const isSelected = selectedTokenIndex === idx;
            const isIntervention = e.requires_intervention;

            return (
              <span
                key={`${e.token_id}-${idx}`}
                onClick={() => onSelectToken(idx)}
                className={`inline-block px-1 py-0.5 mx-0.5 my-0.5 rounded cursor-pointer transition-all border ${
                  isSelected
                    ? 'ring-2 ring-indigo-400 font-bold border-indigo-400 bg-indigo-950/90 text-white'
                    : isIntervention
                    ? 'bg-rose-950/60 text-rose-200 border-rose-800/60 hover:bg-rose-900/80'
                    : 'bg-emerald-950/40 text-emerald-200 border-emerald-900/40 hover:bg-emerald-900/60'
                }`}
                title={`Token: "${e.token}" | Route: ${e.requires_intervention ? 'EXPERT' : 'BASE'} | R(x)=${e.combined_reliability.toFixed(2)}`}
              >
                {e.token}
              </span>
            );
          })
        ) : (
          <div className="text-slate-500 text-xs italic flex items-center justify-center py-6">
            Click "Generate Stream with ARES Pipeline" to view live response...
          </div>
        )}
      </div>

      {/* Selected Token Detail Card */}
      {selectedTokenIndex !== null && generatedEvents[selectedTokenIndex] && (
        <div className="bg-slate-950/90 border border-indigo-900/60 rounded-xl p-4 flex flex-col gap-2 font-sans">
          <div className="flex items-center justify-between text-xs font-semibold">
            <span className="text-indigo-400 flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5" /> Inspecting Token #{selectedTokenIndex + 1}:
              <code className="bg-slate-800 px-1.5 py-0.5 rounded text-white font-mono">
                "{generatedEvents[selectedTokenIndex].token}"
              </code>
            </span>
            <span className="text-slate-400 font-mono">
              Seq Pos: {generatedEvents[selectedTokenIndex].sequence_position}
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono mt-1">
            <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
              <span className="text-slate-400 block text-[10px]">R(x) Score</span>
              <span
                className={`font-bold ${
                  generatedEvents[selectedTokenIndex].is_reliable ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {generatedEvents[selectedTokenIndex].combined_reliability.toFixed(3)}
              </span>
            </div>

            <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
              <span className="text-slate-400 block text-[10px]">Predicted Domain</span>
              <span className="font-bold text-indigo-300 uppercase">
                {generatedEvents[selectedTokenIndex].predicted_domain}
              </span>
            </div>

            <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
              <span className="text-slate-400 block text-[10px]">Selected Route</span>
              <span
                className={`font-bold ${
                  generatedEvents[selectedTokenIndex].requires_intervention ? 'text-rose-400' : 'text-emerald-400'
                }`}
              >
                {generatedEvents[selectedTokenIndex].requires_intervention ? '🔴 EXPERT' : '🟢 BASE'}
              </span>
            </div>

            <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
              <span className="text-slate-400 block text-[10px]">Total Latency</span>
              <span className="font-bold text-slate-200">
                {generatedEvents[selectedTokenIndex].total_latency_ms.toFixed(1)} ms
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
