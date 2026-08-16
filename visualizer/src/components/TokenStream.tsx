import React from 'react';
import type { InferenceEvent } from '../types';

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
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-2">
          Token Stream Execution Path
        </h2>
        <span className="text-[11px] text-slate-400 font-mono">
          {generatedEvents.length} tokens generated {isGenerating && '• Streaming...'}
        </span>
      </div>

      {/* Horizontal Flowing Token Stream Container */}
      <div className="bg-slate-50/80 border border-slate-200/80 rounded-lg p-3 min-h-[90px] max-h-[160px] overflow-y-auto flex flex-wrap gap-1.5 items-center leading-relaxed">
        {generatedEvents.length > 0 ? (
          generatedEvents.map((e, idx) => {
            const isSelected = selectedTokenIndex === idx;
            const isIntervention = e.requires_intervention;
            const expertName = e.selected_expert || 'EXPERT';

            return (
              <button
                key={`${e.token_id}-${idx}`}
                type="button"
                onClick={() => onSelectToken(idx)}
                className={`text-xs font-mono px-2 py-1 rounded transition-all flex items-center gap-1 border ${
                  isSelected
                    ? 'ring-2 ring-indigo-500 font-bold border-indigo-600 bg-indigo-50 text-indigo-900 shadow-sm'
                    : isIntervention
                    ? 'bg-rose-50 text-rose-800 border-rose-200 hover:bg-rose-100 hover:border-rose-300'
                    : 'bg-emerald-50 text-emerald-800 border-emerald-200 hover:bg-emerald-100 hover:border-emerald-300'
                }`}
                title={`Token "${e.token}" | Route: ${e.requires_intervention ? expertName : 'BASE'} | R(x)=${e.combined_reliability.toFixed(2)}`}
              >
                <span>{e.token}</span>
                {isIntervention ? (
                  <span className="text-[9px] font-sans font-bold px-1 rounded bg-rose-200/80 text-rose-900">
                    {expertName.split('_')[0]}
                  </span>
                ) : (
                  <span className="text-[9px] font-sans font-bold px-1 rounded bg-emerald-200/80 text-emerald-900">
                    BASE
                  </span>
                )}
              </button>
            );
          })
        ) : (
          <div className="w-full text-slate-400 text-xs italic text-center py-4 font-sans">
            Enter a prompt above and click "Generate" to watch tokens flow live...
          </div>
        )}
      </div>

      <div className="flex items-center justify-between text-[11px] text-slate-500 font-sans px-1">
        <span>💡 Click any token box to freeze and inspect its ARES reliability metrics in the inspector panel.</span>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1 text-emerald-700">
            <span className="w-2 h-2 rounded-full bg-emerald-500" /> Base Qwen
          </span>
          <span className="flex items-center gap-1 text-rose-700">
            <span className="w-2 h-2 rounded-full bg-rose-500" /> Expert Intervention
          </span>
        </div>
      </div>
    </div>
  );
};
