import React, { useState, useEffect } from 'react';
import type { InferenceEvent, TelemetrySnapshot } from './types';
import { Header } from './components/Header';
import { ControlPanel } from './components/ControlPanel';
import { PipelineFlowGraph } from './components/PipelineFlowGraph';
import { TokenStream } from './components/TokenStream';
import { TelemetryGauges } from './components/TelemetryGauges';

export function App() {
  const [prompt, setPrompt] = useState('Solve step-by-step: If 3x + 7 = 22, what is x?');
  const [policy, setPolicy] = useState('adaptive');
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.7);

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
        // Fallback for standalone frontend dev
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
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased selection:bg-indigo-500 selection:text-white">
      <Header
        engineConnected={engineConnected}
        modelName={modelName}
        activePolicy={policy}
      />

      <main className="max-w-7xl mx-auto px-4 py-6 flex flex-col gap-6">
        {/* Input & Parameters */}
        <ControlPanel
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
        />

        {/* Transformer Explainer Flow Graph */}
        <PipelineFlowGraph
          currentEvent={currentInspectEvent}
          activeExpert={activeExpert}
        />

        {/* Output & Telemetry Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <TokenStream
            events={events}
            selectedTokenIndex={selectedTokenIndex}
            onSelectToken={(idx) => setSelectedTokenIndex(idx)}
            isGenerating={isGenerating}
          />
          <TelemetryGauges snapshot={snapshot} />
        </div>
      </main>
    </div>
  );
}

export default App;
