import React from 'react';
import { Play, Sparkles, Sliders, MessageSquare, Shield, HelpCircle } from 'lucide-react';

interface ControlPanelProps {
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
}

const PRESET_PROMPTS = [
  {
    label: '🧮 Math',
    prompt: 'Solve step-by-step: If 3x + 7 = 22, what is x?',
  },
  {
    label: '💻 Code',
    prompt: 'Write a python function for binary search on a sorted array with type hints.',
  },
  {
    label: '🔬 Science',
    prompt: 'Explain the principle of conservation of energy in thermodynamics.',
  },
  {
    label: '🌐 General',
    prompt: 'Summarize the primary historical impact of the printing press.',
  },
];

export const ControlPanel: React.FC<ControlPanelProps> = ({
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
}) => {
  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-indigo-400" /> Input Prompt & Execution Parameters
        </h2>
        <span className="text-xs text-slate-500 font-mono">Qwen 2.5-7B + Probes</span>
      </div>

      {/* Preset Chips */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-slate-400">Presets:</span>
        {PRESET_PROMPTS.map((preset) => (
          <button
            key={preset.label}
            type="button"
            onClick={() => setPrompt(preset.prompt)}
            className="text-xs px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 hover:text-white border border-slate-700/60 transition-all"
          >
            {preset.label}
          </button>
        ))}
      </div>

      {/* Text Area */}
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={3}
        placeholder="Enter your input prompt here..."
        className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500/80 focus:ring-1 focus:ring-indigo-500/50 transition-all font-sans"
      />

      {/* Controls Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-slate-950/50 p-3.5 rounded-xl border border-slate-800/80">
        {/* Policy Selector */}
        <div>
          <label className="text-xs font-semibold text-slate-400 block mb-1.5 flex items-center gap-1.5">
            <Shield className="w-3.5 h-3.5 text-indigo-400" /> Routing Policy
          </label>
          <select
            value={policy}
            onChange={(e) => setPolicy(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-200 p-2 focus:outline-none focus:border-indigo-500"
          >
            <option value="adaptive">🧠 Adaptive (GRM + LRM Probes)</option>
            <option value="always_base">🟢 Base Only (0% Expert)</option>
            <option value="always_expert">🔴 Always Expert (100% Expert)</option>
            <option value="random_expert">🎲 Random Expert (Baseline)</option>
          </select>
        </div>

        {/* Max Tokens Slider */}
        <div>
          <div className="flex justify-between text-xs font-semibold text-slate-400 mb-1.5">
            <span>Max New Tokens</span>
            <span className="text-indigo-400 font-mono">{maxTokens}</span>
          </div>
          <input
            type="range"
            min={16}
            max={256}
            step={8}
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
            className="w-full accent-indigo-500 bg-slate-800 rounded-lg cursor-pointer"
          />
        </div>

        {/* Temperature Slider */}
        <div>
          <div className="flex justify-between text-xs font-semibold text-slate-400 mb-1.5">
            <span>Temperature</span>
            <span className="text-indigo-400 font-mono">{temperature.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0.1}
            max={1.5}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            className="w-full accent-indigo-500 bg-slate-800 rounded-lg cursor-pointer"
          />
        </div>
      </div>

      {/* Primary Action Button */}
      <button
        type="button"
        onClick={onGenerate}
        disabled={isGenerating || !prompt.strip?.() && !prompt.trim()}
        className={`w-full py-3 px-6 rounded-xl font-semibold text-sm flex items-center justify-center gap-2.5 transition-all shadow-lg ${
          isGenerating
            ? 'bg-indigo-900/50 text-indigo-300 border border-indigo-700/50 cursor-wait'
            : 'bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white shadow-indigo-900/30 active:scale-[0.99]'
        }`}
      >
        {isGenerating ? (
          <>
            <Sparkles className="w-4 h-4 animate-spin text-indigo-300" />
            <span>Streaming Tokens with ARES Adaptive Routing...</span>
          </>
        ) : (
          <>
            <Play className="w-4 h-4 fill-white" />
            <span>Generate Stream with ARES Pipeline</span>
          </>
        )}
      </button>
    </div>
  );
};
