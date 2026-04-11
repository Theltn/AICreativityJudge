import React, { useState } from 'react';
import { Terminal, Activity, Zap, BarChart4, ChevronRight, Loader2, Clock, Brain } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_URL = 'http://localhost:8000';

function App() {
  const [text, setText] = useState('');
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const isOverLimit = wordCount > 500;

  const handleEvaluate = async () => {
    if (!text.trim() || isOverLimit) return;
    setIsEvaluating(true);
    setResults(null);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message.includes('Failed to fetch')
        ? 'Cannot connect to API. Make sure the backend is running (python src/api.py)'
        : err.message
      );
    } finally {
      setIsEvaluating(false);
    }
  };

  const getVerdictType = (score) => {
    if (score >= 5.5) return 'human';
    if (score >= 3.5) return 'neutral';
    return 'ai';
  };

  return (
    <>
      <header className="app-header">
        <div className="app-title">
          <Terminal size={24} className="text-accent" />
          <span>Creativity_Eval_Engine</span>
          <span className="app-title-badge">v2.0</span>
        </div>
        <div className="mono-text text-muted" style={{ fontSize: '0.8rem' }}>
          MODEL: <span className="text-accent">RoBERTa-base</span> · R²=0.936
        </div>
      </header>

      <main className="layout-grid">
        {/* INPUT PANEL */}
        <section className="neo-panel editor-container">
          <div className="neo-header">
            <span className="mono-text" style={{ fontSize: '0.85rem' }}>// INPUT_STREAM</span>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <span className={`mono-text ${isOverLimit ? 'text-cyan' : 'text-muted'}`} style={{ fontSize: '0.85rem' }}>
                WORDS: {wordCount}/500
              </span>
              <Activity size={16} className={isEvaluating ? 'text-accent spin' : 'text-muted'} />
            </div>
          </div>

          <textarea
            className="neo-textarea"
            placeholder="[ PASTE STORY TEXT HERE — AI OR HUMAN ]"
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={isEvaluating}
            spellCheck="false"
          />

          <div className="editor-footer">
            <span className="mono-text text-muted" style={{ fontSize: '0.75rem' }}>
              {isEvaluating ? 'ROBERTA_PROCESSING...' : 'ROBERTA_READY'}
            </span>
            <button
              className="neo-btn"
              onClick={handleEvaluate}
              disabled={wordCount === 0 || isEvaluating || isOverLimit}
            >
              {isEvaluating ? (
                <><Loader2 size={18} className="spin" /> PROCESSING</>
              ) : (
                <><Zap size={18} /> EVALUATE</>
              )}
            </button>
          </div>
        </section>

        {/* OUTPUT PANEL */}
        <section className="neo-panel">
          <div className="neo-header">
            <span className="mono-text" style={{ fontSize: '0.85rem' }}>// OUTPUT_METRICS</span>
            <BarChart4 size={16} className="text-muted" />
          </div>

          <div className="metrics-scroll">
            {!results && !isEvaluating && !error && (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--border)', fontFamily: 'var(--font-mono)' }}>
                &gt; WAITING FOR DATA INPUT...
              </div>
            )}

            {isEvaluating && (
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem', color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                <Loader2 size={32} className="spin" />
                <span>RUNNING INFERENCE PIPELINE...</span>
              </div>
            )}

            {error && (
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem', color: 'var(--accent-red)', fontFamily: 'var(--font-mono)', textAlign: 'center', padding: '2rem' }}>
                <span style={{ fontSize: '1.5rem' }}>⚠</span>
                <span>{error}</span>
              </div>
            )}

            <AnimatePresence>
              {results && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: '100%' }}
                >
                  {/* Two score boxes side by side */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div className="metric-block active" style={{ borderColor: 'var(--accent-green)' }}>
                      <div className="metric-header">
                        <div>
                          <span className="metric-title text-accent" style={{ fontSize: '0.7rem' }}>ROBERTA PREDICTION</span>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.25rem' }}>
                            <Brain size={14} className="text-accent" />
                            <span className="mono-text text-muted" style={{ fontSize: '0.65rem' }}>DEEP LEARNING</span>
                          </div>
                        </div>
                        <span className="metric-score-display text-accent">{results.roberta_score}</span>
                      </div>
                      <div className="structural-bar">
                        <div className="structural-fill green" style={{ width: `${(results.roberta_score / 10) * 100}%` }}></div>
                      </div>
                    </div>

                    <div className="metric-block active" style={{ borderColor: 'var(--accent-cyan)' }}>
                      <div className="metric-header">
                        <div>
                          <span className="metric-title text-cyan" style={{ fontSize: '0.7rem' }}>RUBRIC COMPOSITE</span>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.25rem' }}>
                            <BarChart4 size={14} className="text-cyan" />
                            <span className="mono-text text-muted" style={{ fontSize: '0.65rem' }}>FEATURE ENGINEERING</span>
                          </div>
                        </div>
                        <span className="metric-score-display text-cyan">{results.composite_score}</span>
                      </div>
                      <div className="structural-bar">
                        <div className="structural-fill" style={{ width: `${(results.composite_score / 10) * 100}%` }}></div>
                      </div>
                    </div>
                  </div>

                  {/* 5 dimension cards */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    {Object.entries(results.dimensions).map(([key, dim]) => (
                      <MetricCard
                        key={key}
                        title={key.replace(/_/g, ' ').toUpperCase()}
                        value={dim.score}
                        context={dim.context}
                      />
                    ))}
                  </div>

                  {/* Meta & Verdict */}
                  <div style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                    <span><Clock size={12} style={{ display: 'inline', verticalAlign: 'middle' }} /> {results.processing_time_ms}ms</span>
                    <span>WORDS: {results.word_count}</span>
                  </div>

                  <div className={`verdict-banner ${getVerdictType(results.roberta_score)}`} style={{ marginTop: 'auto' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                      <ChevronRight size={18} />
                      <span className="mono-text" style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>VERDICT</span>
                    </div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', lineHeight: '1.5', color: 'var(--text)' }}>
                      {results.verdict}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>
      </main>
    </>
  );
}

function MetricCard({ title, value, context }) {
  const percentage = (value / 10) * 100;
  const dir = context?.direction || 'neutral';
  const label = context?.label || '';
  const vsMean = context?.vs_mean ?? 0;

  const arrow = dir === 'up' ? '▲' : dir === 'down' ? '▼' : '—';
  const colorClass = dir === 'up' ? 'text-accent' : dir === 'down' ? 'text-red' : 'text-muted';
  const barClass = dir === 'up' ? 'green' : dir === 'down' ? 'red' : '';

  return (
    <div className="metric-block" style={{ padding: '1rem' }}>
      <div className="metric-header" style={{ marginBottom: '0.5rem' }}>
        <span className="metric-title" style={{ fontSize: '0.7rem' }}>{title}</span>
        <span className="mono-text text-accent">{value}<span className="text-muted" style={{ fontSize: '0.7rem' }}>/10</span></span>
      </div>
      <div className="structural-bar" style={{ height: '4px', marginBottom: '0.5rem' }}>
        <div className={`structural-fill ${barClass}`} style={{ width: `${percentage}%` }}></div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="mono-text" style={{ fontSize: '0.65rem' }}>
          <span className={colorClass}>{arrow} {label}</span>
        </span>
        <span className="mono-text text-muted" style={{ fontSize: '0.6rem' }}>
          {vsMean >= 0 ? '+' : ''}{vsMean} vs avg
        </span>
      </div>
    </div>
  );
}

export default App;
