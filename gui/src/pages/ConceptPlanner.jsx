import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { generateConcept } from '../services/api';
import './Pages.css';

const SAMPLE_PROMPTS = [
  "Retro 16-bit dark fantasy RPG about a rogue exploring a sunken temple with acid traps and giant serpent bosses",
  "Cyberpunk action platformer featuring a neon hacker battling rogue AI drones in a rain-slicked metropolis",
  "Chibi dungeon crawler with a wizard casting flame spells against slime monsters in crystal catacombs",
  "8-bit retro adventure with a paladin wielding a holy broadsword against skeletal warlords"
];

const ConceptPlanner = () => {
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conceptSpec, setConceptSpec] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleGenerate = async (selectedPrompt) => {
    const textToUse = selectedPrompt || prompt;
    if (!textToUse || !textToUse.strip?.() && !textToUse.length) return;
    
    setIsLoading(true);
    setError(null);
    try {
      const res = await generateConcept(textToUse);
      if (res && res.data) {
        setConceptSpec(res.data);
      } else {
        throw new Error("Invalid response format");
      }
    } catch (err) {
      console.error(err);
      setError("Failed to generate concept specification. Check backend connection.");
    } finally {
      setIsLoading(false);
    }
  };

  const [isApproved, setIsApproved] = useState(false);

  const handleGenerate = async (selectedPrompt) => {
    const textToUse = selectedPrompt || prompt;
    if (!textToUse || (!textToUse.strip?.() && !textToUse.length)) return;
    
    setIsLoading(true);
    setError(null);
    setIsApproved(false);
    try {
      const res = await generateConcept(textToUse);
      if (res && res.data) {
        setConceptSpec(res.data);
      } else {
        throw new Error("Invalid response format");
      }
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to generate concept specification. Prompt safety check or backend connection issue.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearchTag = (tag) => {
    // Navigate to Create Asset page with pre-filled tag or query parameter
    navigate('/create', { state: { searchQuery: tag } });
  };

  return (
    <div className="page-container max-w-4xl">
      <div className="page-header">
        <h1>Transformer Semantic Planner 🎯</h1>
        <p>Parse free-form natural language prompts into structured game specifications with Responsible AI safety checks and human-in-the-loop oversight.</p>
      </div>

      {/* Natural Language Prompt Input */}
      <div className="card input-card">
        <div className="flex justify-between items-center mb-2">
          <h3>Enter Game Concept Description</h3>
          <span className="badge badge-genre" style={{ backgroundColor: '#10b98122', color: '#10b981', border: '1px solid #10b98144' }}>
            🛡️ Safety Audit Enabled
          </span>
        </div>
        <p className="card-subtitle">Describe your game theme, character role, hazards, and art style in natural language.</p>

        <div className="prompt-input-group">
          <textarea
            className="prompt-textarea"
            rows={3}
            placeholder="e.g. A 16-bit dark fantasy RPG with a rogue exploring a sunken temple..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <button 
            className="btn btn-primary generate-btn" 
            onClick={() => handleGenerate()}
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? "Parsing & Auditing..." : "Generate Game Spec ✨"}
          </button>
        </div>

        {/* Sample Prompt Chips */}
        <div className="sample-chips-container">
          <span className="chips-label">Try a sample prompt:</span>
          <div className="chips-grid">
            {SAMPLE_PROMPTS.map((p, idx) => (
              <button 
                key={idx} 
                className="chip-btn"
                onClick={() => {
                  setPrompt(p);
                  handleGenerate(p);
                }}
              >
                {p.length > 55 ? p.substring(0, 55) + "..." : p}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Loading Spinner */}
      {isLoading && (
        <div className="card loading-card">
          <div className="spinner"></div>
          <p>Transformer Model is parsing semantic request & conducting safety boundary check...</p>
        </div>
      )}

      {/* Error Message / Safety Audit Violation */}
      {error && (
        <div className="card error-card" style={{ borderColor: '#ef4444', backgroundColor: '#ef444415' }}>
          <h4 style={{ color: '#ef4444', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            ⚠️ Responsible AI Safety Alert
          </h4>
          <p className="error-text" style={{ color: '#f87171' }}>{error}</p>
        </div>
      )}

      {/* Structured Game Spec Card */}
      {conceptSpec && !isLoading && (
        <div className="card spec-card animated-fade-in">
          {/* Ethical Guardrails Banner */}
          <div className="flex justify-between items-center p-3 mb-4 rounded" style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
            <div className="flex items-center gap-2">
              <span style={{ fontSize: '1.2rem' }}>🛡️</span>
              <div>
                <strong style={{ color: '#10b981', fontSize: '0.85rem' }}>Passed Safety Audit</strong>
                <p style={{ fontSize: '0.75rem', color: '#9ca3af', margin: 0 }}>
                  Content verified safe • License: {conceptSpec.ethical_guardrails?.license || "CC-BY-NC-SA 4.0"}
                </p>
              </div>
            </div>
            <span className="badge" style={{ background: '#3b82f622', color: '#60a5fa', border: '1px solid #3b82f644' }}>
              Non-Commercial Research Asset
            </span>
          </div>

          <div className="spec-header">
            <div className="spec-title-group">
              <span className="badge badge-genre">{conceptSpec.genre || "Pixel Art RPG"}</span>
              <h2>{conceptSpec.game_title || "Game Concept Spec"}</h2>
            </div>
            <span className="badge badge-style">{conceptSpec.art_style || "16-bit Pixel Art"}</span>
          </div>

          <div className="spec-body-grid">
            {/* Main Character Box */}
            <div className="spec-box">
              <h4>🛡️ Main Character</h4>
              <p className="spec-main-role">{conceptSpec.main_character?.role || "Hero"}</p>
              <div className="tag-cloud">
                {conceptSpec.main_character?.attributes?.map((attr, i) => (
                  <span key={i} className="attr-tag">{attr}</span>
                ))}
              </div>
            </div>

            {/* Environment & Theme Box */}
            <div className="spec-box">
              <h4>🏰 Environment & Theme</h4>
              <p className="spec-main-role">{conceptSpec.environment?.theme || "Dungeon"}</p>
              <div className="tag-cloud">
                {conceptSpec.environment?.hazards?.map((h, i) => (
                  <span key={i} className="hazard-tag">⚠️ {h}</span>
                ))}
              </div>
            </div>

            {/* Enemies & Bosses */}
            <div className="spec-box">
              <h4>👾 Enemies & Bosses</h4>
              <div className="tag-cloud">
                {conceptSpec.enemies?.map((e, i) => (
                  <span key={i} className="enemy-tag">👾 {e}</span>
                ))}
              </div>
            </div>
          </div>

          {/* Recommended VAE Asset Tags */}
          <div className="recommended-assets-section">
            <h3>🎨 Recommended VAE Asset Search Tags</h3>
            <p className="section-sub">Click any tag to search matching game assets in the VAE latent asset gallery:</p>
            <div className="asset-tag-buttons">
              {conceptSpec.recommended_asset_tags?.map((tag, i) => (
                <button key={i} className="btn-asset-tag" onClick={() => handleSearchTag(tag)}>
                  🔍 {tag}
                </button>
              ))}
            </div>
          </div>

          {/* Human-in-the-Loop Oversight Approval Section */}
          <div className="mt-6 pt-4 border-t border-border flex flex-col md:flex-row justify-between items-center gap-4" style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
            <label className="flex items-center gap-3 cursor-pointer" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <input 
                type="checkbox" 
                checked={isApproved} 
                onChange={(e) => setIsApproved(e.target.checked)}
                style={{ width: '18px', height: '18px', accentColor: '#10b981', cursor: 'pointer' }}
              />
              <span style={{ fontSize: '0.85rem', color: isApproved ? '#10b981' : '#d1d5db', fontWeight: isApproved ? '600' : '400' }}>
                🤝 Human Designer Oversight: I approve this specification for asset pipeline generation
              </span>
            </label>

            <button 
              className="btn btn-primary"
              disabled={!isApproved}
              onClick={() => handleSearchTag(conceptSpec.recommended_asset_tags?.[0] || "sprite")}
              style={{ 
                opacity: isApproved ? 1 : 0.5, 
                cursor: isApproved ? 'pointer' : 'not-allowed',
                backgroundColor: isApproved ? '#10b981' : '#4b5563',
                borderColor: isApproved ? '#059669' : '#374151'
              }}
            >
              {isApproved ? "Proceed to Asset Generation 🚀" : "Approval Required to Proceed"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConceptPlanner;
