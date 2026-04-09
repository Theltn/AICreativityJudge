import React, { useState } from 'react';
import { Terminal, Activity, Zap, FileText, BarChart4, ChevronRight, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [text, setText] = useState('');
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [results, setResults] = useState(null);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const isOverLimit = wordCount > 500;

  const handleEvaluate = () => {
    if (!text.trim() || isOverLimit) return;
    setIsEvaluating(true);
    setResults(null);

    // Simulate Backend Analytics Delay
    setTimeout(() => {
      setResults({
        overall: (Math.random() * 3 + 6).toFixed(1),
        lexical: (Math.random() * 4 + 4).toFixed(1),
        syntax: (Math.random() * 3 + 5).toFixed(1),
        novelty: (Math.random() * 5 + 3).toFixed(1),
        imagery: (Math.random() * 4 + 5).toFixed(1),
        narrative: (Math.random() * 5 + 4).toFixed(1)
      });
      setIsEvaluating(false);
    }, 1200);
  };

  const getVerdict = (score) => {
    if (score > 8.0) return { label: 'HUMAN-LIKE CREATIVITY', type: 'human' };
    if (score > 6.5) return { label: 'STANDARD LLM GENERATION', type: 'neutral' };
    return { label: 'FORMULAIC / BASELINE AI', type: 'ai' };
  };

  return (
    <>
      <header className="app-header">
        <div className="app-title">
          <Terminal size={24} className="text-accent" />
          <span>Creativity_Eval_Engine</span>
          <span className="app-title-badge">v1.0.4</span>
        </div>
        <div className="mono-text text-muted" style={{ fontSize: '0.8rem' }}>
          SYSTEM.STATUS: <span className="text-accent">ONLINE</span>
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
              <Activity size={16} className={isEvaluating ? "text-accent spin" : "text-muted"} />
            </div>
          </div>
          
          <textarea 
            className="neo-textarea"
            placeholder="[ PASTE EVALUATION CORPUS HERE ]"
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={isEvaluating}
            spellCheck="false"
          />
          
          <div className="editor-footer">
            <span className="mono-text text-muted" style={{ fontSize: '0.75rem' }}>
              ROBERTA_MODEL_WAITING
            </span>
            <button 
              className="neo-btn" 
              onClick={handleEvaluate}
              disabled={wordCount === 0 || isEvaluating || isOverLimit}
            >
              {isEvaluating ? (
                <><Loader2 size={18} className="spin" /> PROCESSING</>
              ) : (
                <><Zap size={18} /> EXECUTE EVALUATION</>
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
            {!results && !isEvaluating && (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--border)', fontFamily: 'var(--font-mono)' }}>
                 &gt; WAITING FOR DATA INPUT...
              </div>
            )}

            {isEvaluating && (
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem', color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                <Loader2 size={32} className="spin" />
                <span>ANALYZING HIGH-DIMENSIONAL VECTORS...</span>
              </div>
            )}

            <AnimatePresence>
              {results && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: '100%' }}
                >
                  <div className="metric-block active" style={{ borderColor: 'var(--accent-cyan)' }}>
                    <div className="metric-header">
                      <span className="metric-title text-cyan">AGGREGATE SCORE</span>
                      <span className="metric-score-display text-cyan">{results.overall}</span>
                    </div>
                    <div className="structural-bar">
                      <div className="structural-fill" style={{ width: `${(results.overall / 10) * 100}%` }}></div>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <MetricCard title="LEXICAL RICHNESS" value={results.lexical} />
                    <MetricCard title="SYNTACTIC CPLX" value={results.syntax} />
                    <MetricCard title="NOVELTY INDEX" value={results.novelty} />
                    <MetricCard title="IMAGERY DENSITY" value={results.imagery} />
                  </div>
                  
                  <MetricCard title="NARRATIVE DYNAMICS" value={results.narrative} fullWidth />

                  <div className={`verdict-banner ${getVerdict(results.overall).type}`} style={{ marginTop: 'auto' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                      <ChevronRight size={18} />
                      <span className="mono-text" style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>FINAL VERDICT</span>
                    </div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 700, color: 'var(--text)' }}>
                      [{getVerdict(results.overall).label}]
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

function MetricCard({ title, value, fullWidth = false }) {
  const percentage = (value / 10) * 100;
  return (
    <div className={`metric-block ${fullWidth ? 'full-width' : ''}`} style={{ padding: '1rem' }}>
      <div className="metric-header" style={{ marginBottom: '0.75rem' }}>
        <span className="metric-title" style={{ fontSize: '0.75rem' }}>{title}</span>
        <span className="mono-text text-accent">{value}</span>
      </div>
      <div className="structural-bar" style={{ height: '4px' }}>
        <div className="structural-fill green" style={{ width: `${percentage}%` }}></div>
      </div>
    </div>
  );
}

export default App;
