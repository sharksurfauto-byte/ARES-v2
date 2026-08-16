import React, { useState } from 'react';
import type { InferenceEvent } from '../types';
import { Copy, Check, Sparkles, MessageSquare } from 'lucide-react';

interface CleanOutputProps {
  events: InferenceEvent[];
  isGenerating: boolean;
}

export const CleanOutput: React.FC<CleanOutputProps> = ({ events, isGenerating }) => {
  const [copied, setCopied] = useState(false);

  const generatedEvents = events.filter((e) => !e.is_prompt_token);
  const cleanText = generatedEvents.map((e) => e.token).join('');

  const handleCopy = () => {
    if (!cleanText) return;
    navigator.clipboard.writeText(cleanText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col gap-3">
      <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-indigo-600" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 flex items-center gap-1.5">
            Clean Model Response
          </h2>
        </div>

        <div className="flex items-center gap-3 text-xs">
          {cleanText && (
            <span className="text-[11px] font-mono text-slate-400">
              {cleanText.length} characters
            </span>
          )}

          <button
            type="button"
            onClick={handleCopy}
            disabled={!cleanText}
            className={`px-2.5 py-1 rounded text-xs font-medium flex items-center gap-1.5 transition-colors border ${
              copied
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700 border-slate-200 disabled:opacity-50'
            }`}
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" />
                <span>Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-500" />
                <span>Copy Output</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Text Output Box */}
      <div className="bg-slate-50/80 border border-slate-200/80 rounded-lg p-3.5 min-h-[100px] text-sm text-slate-900 font-sans leading-relaxed whitespace-pre-wrap">
        {cleanText ? (
          <div>
            {cleanText}
            {isGenerating && (
              <span className="inline-block w-2 h-4 bg-indigo-600 animate-pulse ml-1 align-middle" />
            )}
          </div>
        ) : (
          <div className="text-slate-400 text-xs italic text-center py-6">
            Response will stream here cleanly as text is generated...
          </div>
        )}
      </div>
    </div>
  );
};
