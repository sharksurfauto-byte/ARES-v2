import React from 'react';
import { Cpu, Play, Sparkles, SlidersHorizontal, RefreshCw } from 'lucide-react';

interface HeaderProps {
  prompt: string;
  setPrompt: (p: string) => void;
  policy: string;
  setPolicy: (p: string) => void;
  maxTokens: number;
  setMaxTokens: (n: number) => void;
  temperature: number;
  setTemperature: (n: number) => void;
  onGenerate: () => void;
  isGenerating: boolean;
  engineConnected: boolean;
  modelName: string;
}

const PRESETS = [
  { label: 'Math', prompt: 'Solve step-by-step: If 3x + 7 = 22, what is x?' },
  { label: 'Code', prompt: 'Write a python function for binary search with type hints.' },
  { label: 'Science', prompt: 'Explain the thermodynamic cycle of a heat pump in physics.' },
  { label: 'General', prompt: 'Summarize the primary historical impact of the printing press.' },
];

export const Header: React.FC<HeaderProps> = ({
  prompt,
  setPrompt,
  policy,
  setPolicy,
  maxTokens,
  setMaxTokens,
  temperature,
  setTemperature,
  onGenerate,
  isGenerating,
  engineConnected,
  modelName,
}) => {
  return (
    <header className="bg-white border-b border-slate-200 text-slate-800 px-5 py-3 sticky top-0 z-50 shadow-sm flex flex-col gap-2.5">
      {/* Row 1: Logo, Integrated Prompt Bar, Primary Generate Button */}
      <div className="flex items-center gap-3">
        {/* Logo Branding */}
        <div className="flex items-center gap-2 pr-3 border-r border-slate-200 shrink-0">
          <div className="bg-indigo-50 p-1.5 rounded-lg border border-indigo-100 text-indigo-600">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-slate-900 flex items-center gap-1.5">
              ARES Live
              <span className="text-[10px] font-semibold px-1.5 py-0.2 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
                v2.0
              </span>
            </h1>
            <span className="text-[10px] text-slate-400 font-mono block leading-none">
              {modelName || 'Qwen 2.5-7B'}
            </span>
          </div>
        </div>

        {/* Integrated Prompt Input Bar */}
        <div className="flex-1 flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 focus-within:border-indigo-400 focus-within:bg-white focus-within:ring-1 focus-within:ring-indigo-300 transition-all">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !isGenerating && onGenerate()}
            placeholder="Enter input prompt to visualize ARES adaptive routing..."
            className="w-full bg-transparent text-xs text-slate-900 placeholder-slate-400 focus:outline-none font-sans"
          />

          {/* Quick Presets Dropdown/Buttons */}
          <div className="flex items-center gap-1 shrink-0 border-l border-slate-200 pl-2">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => setPrompt(p.prompt)}
                className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-white hover:bg-indigo-50 text-slate-600 hover:text-indigo-600 border border-slate-200 transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Primary Generate Button */}
        <button
          type="button"
          onClick={onGenerate}
          disabled={isGenerating || !prompt.trim()}
          className={`px-4 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 shadow-sm transition-all shrink-0 ${
            isGenerating
              ? 'bg-indigo-100 text-indigo-700 border border-indigo-200 cursor-wait'
              : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-200 active:scale-[0.98]'
          }`}
        >
          {isGenerating ? (
            <>
              <Sparkles className="w-3.5 h-3.5 animate-spin text-indigo-600" />
              <span>Processing...</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-white" />
              <span>Generate</span>
            </>
          )}
        </button>
      </div>

      {/* Row 2: Compact Parameter Controls & Engine Status */}
      <div className="flex items-center justify-between text-xs text-slate-600 pt-1 border-t border-slate-100">
        <div className="flex items-center gap-6">
          {/* Policy Radio/Select */}
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-semibold text-slate-500">Policy:</span>
            <select
              value={policy}
              onChange={(e) => setPolicy(e.target.value)}
              className="bg-white border border-slate-200 rounded px-2 py-0.5 text-xs text-slate-800 focus:outline-none focus:border-indigo-400 font-medium"
            >
              <option value="adaptive">🧠 Adaptive Routing (GRM+LRM)</option>
              <option value="always_base">🟢 Base Only (0% Expert)</option>
              <option value="always_expert">🔴 Always Expert (100% Expert)</option>
              <option value="random_expert">🎲 Random Expert (Baseline)</option>
            </select>
          </div>

          {/* Temperature Slider */}
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-semibold text-slate-500">Temp:</span>
            <input
              type="range"
              min={0.1}
              max={1.5}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-20 accent-indigo-600 cursor-pointer"
            />
            <span className="font-mono text-[11px] text-slate-700 w-8">{temperature.toFixed(2)}</span>
          </div>

          {/* Max Tokens Slider */}
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-semibold text-slate-500">Max Tokens:</span>
            <input
              type="range"
              min={16}
              max={256}
              step={8}
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              className="w-24 accent-indigo-600 cursor-pointer"
            />
            <span className="font-mono text-[11px] text-slate-700 w-8">{maxTokens}</span>
          </div>
        </div>

        {/* Engine Status Indicator */}
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              engineConnected ? 'bg-emerald-500' : 'bg-amber-500 animate-pulse'
            }`}
          />
          <span className="text-[11px] font-medium text-slate-500">
            {engineConnected ? 'Backend Connected' : 'Connecting to PyTorch Server...'}
          </span>
        </div>
      </div>
    </header>
  );
};
