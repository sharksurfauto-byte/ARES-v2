import React, { useState, useEffect } from 'react';
import type { InferenceEvent, TelemetrySnapshot } from './types';
import { Header } from './components/Header';
import { PipelineFlowGraph } from './components/PipelineFlowGraph';
import { AttentionExplainer } from './components/AttentionExplainer';
import { TokenStream } from './components/TokenStream';
import { CleanOutput } from './components/CleanOutput';

export function App() {
  const [prompt, setPrompt] = useState('Solve step-by-step: If 3x + 7 = 22, what is x?');
  const [policy, setPolicy] = useState('adaptive');
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.7);

  const [collectAttentions, setCollectAttentions] = useState(true);
  const [attnLayer, setAttnLayer] = useState(12);
  const [attnHead, setAttnHead] = useState(7);
  const [canvasTab, setCanvasTab] = useState<'pipeline' | 'attention' | 'combined'>('combined');

  const [events, setEvents] = useState<InferenceEvent[]>([]);
  const [snapshot, setSnapshot] = useState<TelemetrySnapshot | null>(null);
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(null);

  const [isGenerating, setIsGenerating] = useState(false);
  const [engineConnected, setEngineConnected] = useState(false);
  const [modelName, setModelName] = useState('Qwen 2.5-7B');

  // Check backend health on mount
  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then((data) => {
        if (data.status === 'healthy' || data.engine_initialized) {
          setEngineConnected(true);
          setModelName(data.model_name || 'Qwen 2.5-7B');
        }
      })
      .catch(() => {
        setEngineConnected(false);
      });
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim() || isGenerating) return;

    setIsGenerating(true);
    setEvents([]);
    setSnapshot(null);
    setSelectedTokenIndex(null);

    try {
      const response = await fetch('/api/generate_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          temperature,
          do_sample: true,
          policy,
          collect_attentions: collectAttentions,
          attn_layer: attnLayer,
          attn_head: attnHead,
        }),
      });

      if (!response.body) {
        setIsGenerating(false);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.replace(/^data:\s*/, '').trim();
            if (!jsonStr) continue;

            try {
              const parsed = JSON.parse(jsonStr);
              if (parsed.type === 'snapshot') {
                setSnapshot(parsed.data);
              } else if (parsed.token) {
                setEvents((prev) => [...prev, parsed]);
              }
            } catch (err) {
              console.error('SSE parse error:', err);
            }
          }
        }
      }
    } catch (err) {
      console.error('Generation error:', err);
    } finally {
      setIsGenerating(false);
    }
  };

  const latestEvent = events.length > 0 ? events[events.length - 1] : null;
  const currentInspectEvent =
    selectedTokenIndex !== null && events[selectedTokenIndex]
      ? events[selectedTokenIndex]
      : latestEvent;

  const activeExpert = currentInspectEvent?.selected_expert ?? null;

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 font-sans antialiased selection:bg-indigo-500 selection:text-white">
      {/* Top Control Bar */}
      <Header
        prompt={prompt}
        setPrompt={setPrompt}
        policy={policy}
        setPolicy={setPolicy}
        maxTokens={maxTokens}
        setMaxTokens={setMaxTokens}
        temperature={temperature}
        setTemperature={setTemperature}
        onGenerate={handleGenerate}
        isGenerating={isGenerating}
        engineConnected={engineConnected}
        modelName={modelName}
      />

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 py-4 flex flex-col gap-4">
        {/* Canvas View Switcher */}
        <div className="flex items-center justify-between bg-white border border-slate-200 rounded-xl px-4 py-2 shadow-sm">
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setCanvasTab('combined')}
              className={`px-3 py-1 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                canvasTab === 'combined'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Dual Combined View</span>
            </button>
            <button
              type="button"
              onClick={() => setCanvasTab('pipeline')}
              className={`px-3 py-1 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                canvasTab === 'pipeline'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
              }`}
            >
              <Network className="w-3.5 h-3.5" />
              <span>ARES Pipeline Graph</span>
            </button>
            <button
              type="button"
              onClick={() => setCanvasTab('attention')}
              className={`px-3 py-1 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                canvasTab === 'attention'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Qwen Attention Explainer</span>
            </button>
          </div>

          <div className="text-[11px] text-slate-500 font-mono">
            Qwen 2.5-7B (28 Layers, 28 Heads) + ARES Adaptive Expert Router
          </div>
        </div>

        {/* Dynamic Main Canvas Section */}
        {canvasTab === 'combined' && (
          <div className="flex flex-col gap-4">
            <PipelineFlowGraph
              currentEvent={currentInspectEvent}
              activeExpert={activeExpert}
            />
            <AttentionExplainer
              selectedEvent={currentInspectEvent}
              events={events}
              attnLayer={attnLayer}
              setAttnLayer={setAttnLayer}
              attnHead={attnHead}
              setAttnHead={setAttnHead}
              collectAttentions={collectAttentions}
              setCollectAttentions={setCollectAttentions}
            />
          </div>
        )}

        {canvasTab === 'pipeline' && (
          <PipelineFlowGraph
            currentEvent={currentInspectEvent}
            activeExpert={activeExpert}
          />
        )}

        {canvasTab === 'attention' && (
          <AttentionExplainer
            selectedEvent={currentInspectEvent}
            events={events}
            attnLayer={attnLayer}
            setAttnLayer={setAttnLayer}
            attnHead={attnHead}
            setAttnHead={setAttnHead}
            collectAttentions={collectAttentions}
            setCollectAttentions={setCollectAttentions}
          />
        )}

        {/* Bottom Section: Flowing Token Stream, Clean Output & Side Inspector */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2 flex flex-col gap-4">
            <CleanOutput
              events={events}
              isGenerating={isGenerating}
            />
            <TokenStream
              events={events}
              selectedTokenIndex={selectedTokenIndex}
              onSelectToken={(idx) => setSelectedTokenIndex(idx)}
              isGenerating={isGenerating}
            />
          </div>
          <div>
            <TelemetryGauges
              snapshot={snapshot}
              selectedEvent={currentInspectEvent}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
