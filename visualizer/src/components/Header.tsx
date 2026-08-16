import React from 'react';
import { Cpu, Activity, ShieldCheck, GitBranch } from 'lucide-react';

interface HeaderProps {
  engineConnected: boolean;
  modelName: string;
  activePolicy: string;
}

export const Header: React.FC<HeaderProps> = ({
  engineConnected,
  modelName,
  activePolicy,
}) => {
  return (
    <header className="bg-slate-900/80 backdrop-blur border-b border-slate-800 text-white px-6 py-4 flex items-center justify-between sticky top-0 z-50 shadow-lg">
      <div className="flex items-center gap-3">
        <div className="bg-indigo-600/30 p-2 rounded-xl border border-indigo-500/40 text-indigo-400">
          <Cpu className="w-6 h-6 animate-pulse" />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">
            ARES Live <span className="text-xs font-semibold uppercase px-2 py-0.5 rounded-full bg-indigo-900/60 text-indigo-300 border border-indigo-700/50 ml-2">v2.0</span>
          </h1>
          <p className="text-xs text-slate-400 font-medium">
            Adaptive Reliability & Expert System — Real-time Pipeline Visualizer
          </p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Engine Status Badge */}
        <div
          className={`flex items-center gap-2 text-xs font-semibold px-3 py-1.5 rounded-full border ${
            engineConnected
              ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800/60'
              : 'bg-rose-950/60 text-rose-300 border-rose-800/60'
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              engineConnected ? 'bg-emerald-400 animate-ping' : 'bg-rose-400'
            }`}
          />
          {engineConnected ? `Engine Online (${modelName || 'Qwen 7B'})` : 'Engine Offline'}
        </div>

        {/* Policy Badge */}
        <div className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
          <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
          <span className="capitalize">Policy: {activePolicy}</span>
        </div>

        {/* GitHub Link */}
        <a
          href="https://github.com/sharksurfauto-byte/ARES-v2"
          target="_blank"
          rel="noopener noreferrer"
          className="text-slate-400 hover:text-white transition-colors p-2 rounded-lg hover:bg-slate-800"
          title="ARES v2 Repository"
        >
          <GitBranch className="w-5 h-5" />
        </a>
      </div>
    </header>
  );
};
