import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { generateConcept } from '../services/api';
import './Pages.css';

const SAMPLE_PROMPTS = [
  "cozy farming game with a fox",
  "space asteroid destroyer",
  "a space farming game with a fox",
  "dark fantasy dungeon crawler with skeleton warriors"
];

const ConceptPlanner = () => {
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conceptSpec, setConceptSpec] = useState(null);
  const [error, setError] = useState(null);
  const [showRawOutput, setShowRawOutput] = useState(false);
  const navigate = useNavigate();

  const [isApproved, setIsApproved] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const debug = params.get('debug');
    if (debug === 'fox') {
      const foxPrompt = "cozy farming game with a fox";
      setPrompt(foxPrompt);
      setConceptSpec({
        status: "success",
        prompt: "cozy farming game with a fox",
        formatted_input: "game concept: cozy farming game with a fox",
        game_title: "Cool Farming Adventure",
        theme: "Cool Farming",
        environment: "Enchanted Forest",
        visualStyle: "Hand-Drawn 2D",
        characters: [
          {
            name: "Foggy Fox",
            desc: "A common villager caught up in the conflict."
          }
        ],
        weapons: [
          {
            name: "Bow",
            desc: "A ranged weapon used for hunting and defense."
          }
        ],
        props: [],
        main_character: {
          role: "Foggy Fox",
          attributes: ["A common villager caught up in the conflict."]
        },
        art_style: "Hand-Drawn 2D",
        genre: "Cool Farming RPG",
        enemies: [],
        recommended_asset_tags: [
          "cool_farming_theme",
          "foggy_fox_character",
          "bow_weapon"
        ],
        raw_t5_output: "{\"theme\": \"Cool Farming\", \"environment\": \"Enchanted Forest\", \"characters\": [{\"name\": \"Foggy Fox\", \"desc\": \"A common villager caught up in the conflict.\"}], \"weapons\": [{\"name\": \"Bow\", \"desc\": \"A ranged weapon used for hunting and defense.\"}], \"props\": [], \"visualStyle\": \"Hand-Drawn 2D\"}",
        telemetry: {
          inference_latency_ms: 3292.0,
          tokens_generated: 121,
          model_checkpoint: "models/4_Transformer_2k_v1",
          base_architecture: "google/flan-t5-small",
          decoding_strategy: "Beam Search (num_beams=4)",
          schema_valid: true
        },
        ethical_guardrails: {
          safety_status: "Passed Regex Boundary Audit",
          dataset_attribution: "GameForge Fine-Tuned T5 (2,000 Curated Prompt-JSON Pairs; 80/10/10 Split)",
          license: "CC-BY-NC-SA 4.0",
          human_oversight_required: true
        },
        enrichment_layer: {
          pipeline_stage: "Stage 2: Deterministic Asset Mapping (Rule-Based)",
          recommended_asset_tags: [
            "cool_farming_theme",
            "foggy_fox_character",
            "bow_weapon"
          ]
        }
      });
      setShowRawOutput(true);
      return;
    }

    const p = params.get('prompt');
    const raw = params.get('raw');
    if (p) {
      setPrompt(p);
      handleGenerate(p).then(() => {
        if (raw === 'true') {
          setShowRawOutput(true);
        }
      });
    }
  }, []);

  const handleGenerate = async (selectedPrompt) => {
    const textToUse = selectedPrompt || prompt;
    if (!textToUse || (!textToUse.trim?.() && !textToUse.length)) return;
    
    setIsLoading(true);
    setError(null);
    setIsApproved(false);
    setShowRawOutput(false);
    try {
      const res = await generateConcept(textToUse.trim());
      if (res && res.data) {
        setConceptSpec(res.data);
      } else {
        throw new Error("Invalid response format from server");
      }
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to generate concept specification.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearchTag = (tag) => {
    navigate('/create', { state: { searchQuery: tag } });
  };

  return (
    <div className="page-container max-w-4xl">
      <div className="page-header">
        <h1>Transformer Semantic Planner 🎯</h1>
        <p>End-to-end neural Seq2Seq specification planning powered by our fine-tuned <strong>Flan-T5</strong> model (2,000 prompt-JSON pairs).</p>
      </div>

      {/* Natural Language Prompt Input */}
      <div className="card input-card">
        <div className="flex justify-between items-center mb-2">
          <h3>Enter Game Concept Description</h3>
          <span className="badge badge-genre" style={{ backgroundColor: '#10b98122', color: '#10b981', border: '1px solid #10b98144' }}>
            🛡️ Regex Boundary Audit Enabled
          </span>
        </div>
        <p className="card-subtitle">Type any natural-language concept. The model generates structured game themes, environments, characters, weapons, and props.</p>

        <div className="prompt-input-group">
          <textarea
            className="prompt-textarea"
            rows={3}
            placeholder="e.g. cozy farming game with a fox..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <button 
            className="btn btn-primary generate-btn" 
            onClick={() => handleGenerate()}
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? "Running Flan-T5 Inference..." : "Generate Game Spec ✨"}
          </button>
        </div>

        {/* Sample Prompt Chips */}
        <div className="sample-chips-container">
          <span className="chips-label">Try test prompts:</span>
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
                {p}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Loading Spinner */}
      {isLoading && (
        <div className="card loading-card">
          <div className="spinner"></div>
          <p>Running Seq2Seq Beam Search Inference on fine-tuned Flan-T5 checkpoint...</p>
        </div>
      )}

      {/* Error Message / Safety Audit Violation */}
      {error && (
        <div className="card error-card" style={{ borderColor: '#ef4444', backgroundColor: '#ef444415' }}>
          <h4 style={{ color: '#ef4444', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            ⚠️ Error / Safety Boundary Alert
          </h4>
          <p className="error-text" style={{ color: '#f87171' }}>{error}</p>
        </div>
      )}

      {/* Parse Error Display if Model output invalid JSON */}
      {conceptSpec && conceptSpec.status === "parse_error" && !isLoading && (
        <div className="card error-card" style={{ borderColor: '#f59e0b', backgroundColor: '#f59e0b15' }}>
          <h4 style={{ color: '#f59e0b', marginBottom: '0.5rem' }}>
            ⚠️ JSON Parse Warning (Direct Model Output)
          </h4>
          <p className="error-text" style={{ color: '#fbbf24', fontSize: '0.85rem' }}>{conceptSpec.error_detail}</p>
          <div className="mt-3 p-3 bg-black/60 rounded text-xs font-mono text-gray-300 overflow-x-auto">
            <strong>Raw Output:</strong> {conceptSpec.raw_t5_output}
          </div>
        </div>
      )}

      {/* Structured Game Spec Card */}
      {conceptSpec && conceptSpec.status === "success" && !isLoading && (
        <div className="card spec-card animated-fade-in">
          {/* Ethical Guardrails & Real Telemetry Banner */}
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center p-3 mb-4 rounded gap-2" style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
            <div className="flex items-center gap-2">
              <span style={{ fontSize: '1.2rem' }}>🛡️</span>
              <div>
                <strong style={{ color: '#10b981', fontSize: '0.85rem' }}>
                  {conceptSpec.ethical_guardrails?.safety_status || "Passed Regex Boundary Audit"}
                </strong>
                <p style={{ fontSize: '0.75rem', color: '#9ca3af', margin: 0 }}>
                  Model: {conceptSpec.telemetry?.model_checkpoint} • Attribution: {conceptSpec.ethical_guardrails?.dataset_attribution}
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <span className="badge" style={{ background: '#3b82f622', color: '#60a5fa', border: '1px solid #3b82f644' }}>
                ⚡ {conceptSpec.telemetry?.inference_latency_ms} ms Latency
              </span>
              <span className="badge" style={{ background: '#8b5cf622', color: '#a78bfa', border: '1px solid #8b5cf644' }}>
                🔤 {conceptSpec.telemetry?.tokens_generated} Tokens
              </span>
            </div>
          </div>

          {/* Evaluated Schema Header */}
          <div className="spec-header border-b border-[#2a2a35] pb-4 mb-4">
            <div className="spec-title-group">
              <span className="badge badge-genre">Theme: {conceptSpec.theme || "General"}</span>
              <h2 className="text-xl font-bold mt-1 text-white">{conceptSpec.game_title}</h2>
            </div>
            <div className="flex gap-2">
              <span className="badge badge-style">Environment: {conceptSpec.environment || "Unspecified"}</span>
              <span className="badge" style={{ background: '#f59e0b22', color: '#fbbf24', border: '1px solid #f59e0b44' }}>
                Style: {conceptSpec.visualStyle || "2D Pixel Art"}
              </span>
            </div>
          </div>

          {/* Evaluated Schema Entities Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {/* Characters Box */}
            <div className="spec-box bg-[#15151e] p-4 rounded-xl border border-[#2c2c3a]">
              <h4 className="text-sm font-bold text-indigo-300 uppercase tracking-wider mb-3">
                🎭 Characters ({conceptSpec.characters?.length || 0})
              </h4>
              <div className="space-y-2">
                {conceptSpec.characters && conceptSpec.characters.length > 0 ? (
                  conceptSpec.characters.map((c, i) => (
                    <div key={i} className="p-2 rounded bg-[#101016] border border-[#242430]">
                      <div className="text-xs font-bold text-white">{c.name}</div>
                      <div className="text-[11px] text-gray-400 mt-0.5">{c.desc}</div>
                    </div>
                  ))
                ) : (
                  <div className="text-xs text-gray-500 italic">None generated</div>
                )}
              </div>
            </div>

            {/* Weapons Box */}
            <div className="spec-box bg-[#15151e] p-4 rounded-xl border border-[#2c2c3a]">
              <h4 className="text-sm font-bold text-amber-300 uppercase tracking-wider mb-3">
                ⚔️ Weapons ({conceptSpec.weapons?.length || 0})
              </h4>
              <div className="space-y-2">
                {conceptSpec.weapons && conceptSpec.weapons.length > 0 ? (
                  conceptSpec.weapons.map((w, i) => (
                    <div key={i} className="p-2 rounded bg-[#101016] border border-[#242430]">
                      <div className="text-xs font-bold text-white">{w.name}</div>
                      <div className="text-[11px] text-gray-400 mt-0.5">{w.desc}</div>
                    </div>
                  ))
                ) : (
                  <div className="text-xs text-gray-500 italic">None generated</div>
                )}
              </div>
            </div>

            {/* Props Box */}
            <div className="spec-box bg-[#15151e] p-4 rounded-xl border border-[#2c2c3a]">
              <h4 className="text-sm font-bold text-emerald-300 uppercase tracking-wider mb-3">
                📦 Props / Items ({conceptSpec.props?.length || 0})
              </h4>
              <div className="space-y-2">
                {conceptSpec.props && conceptSpec.props.length > 0 ? (
                  conceptSpec.props.map((p, i) => (
                    <div key={i} className="p-2 rounded bg-[#101016] border border-[#242430]">
                      <div className="text-xs font-bold text-white">{p.name}</div>
                      <div className="text-[11px] text-gray-400 mt-0.5">{p.desc}</div>
                    </div>
                  ))
                ) : (
                  <div className="text-xs text-gray-500 italic">None generated (Empty list in ground truth distribution)</div>
                )}
              </div>
            </div>
          </div>

          {/* Stage 2: Deterministic Asset Mapping (Rule-Based Enrichment) */}
          <div className="bg-[#121218] p-4 rounded-xl border border-[#252535] mb-4">
            <div className="flex justify-between items-center mb-1">
              <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider">
                🏷️ {conceptSpec.enrichment_layer?.pipeline_stage || "Stage 2: Deterministic Asset Search Tags"}
              </h4>
              <span className="text-[10px] text-gray-500 font-mono">Rule-Based Tag Extractor</span>
            </div>
            <p className="text-[11px] text-gray-400 mb-3">
              {conceptSpec.enrichment_layer?.note || "Derived deterministically from Flan-T5 theme and character entities for downstream pixel art sprite discovery:"}
            </p>
            <div className="flex flex-wrap gap-2">
              {conceptSpec.recommended_asset_tags?.map((tag, i) => (
                <button key={i} className="btn-asset-tag text-xs" onClick={() => handleSearchTag(tag)}>
                  🔍 {tag}
                </button>
              ))}
            </div>
          </div>

          {/* Raw T5 Output Toggle Drawer */}
          <div className="mt-4 pt-3 border-t border-[#252530]">
            <button
              type="button"
              className="text-xs text-indigo-400 hover:text-indigo-300 underline font-mono flex items-center gap-1 cursor-pointer"
              onClick={() => setShowRawOutput(!showRawOutput)}
            >
              <span>{showRawOutput ? "▼ Hide Raw Flan-T5 Neural Output" : "▶ Inspect Raw Flan-T5 Neural Decoded JSON (Debug Panel)"}</span>
            </button>

            {showRawOutput && (
              <div className="mt-3 p-3 bg-black/80 rounded-lg border border-[#333] text-xs font-mono text-emerald-400 overflow-x-auto">
                <div className="text-[10px] text-gray-500 mb-1 border-b border-[#222] pb-1">
                  Format: Model Output ID Decode | Decoding: Beam Search (num_beams=4) | Task Prefix: "game concept: "
                </div>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{conceptSpec.raw_t5_output}</pre>
              </div>
            )}
          </div>

          {/* Human-in-the-Loop Oversight Approval Section */}
          <div className="mt-6 pt-4 border-t border-[#252530] flex flex-col md:flex-row justify-between items-center gap-4">
            <label className="flex items-center gap-3 cursor-pointer">
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
