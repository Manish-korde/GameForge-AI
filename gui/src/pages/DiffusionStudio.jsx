import { useState } from "react";
import { generateDiffusionSprite } from "../services/api";

export default function DiffusionStudio() {
  const [prompt, setPrompt] = useState("Armored Holy Knight with silver plate and heavy broadsword");
  const [style, setStyle] = useState("16-bit Retro RPG");
  const [steps, setSteps] = useState(25);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [seed, setSeed] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("comparison"); // 'comparison', 'prompts', 'palette'

  const quickPrompts = [
    { label: "⚔️ Holy Paladin Knight", prompt: "Armored Holy Knight with silver plate and heavy broadsword" },
    { label: "🔮 Arcane Sorcerer", prompt: "Arcane Sorcerer in glowing purple robes casting lightning" },
    { label: "🗡️ Stealth Assassin", prompt: "Stealth Rogue Assassin in dark hooded cloak with dual daggers" },
    { label: "💀 Skeleton Necromancer", prompt: "Undead Skeleton Necromancer in dark crypt holding bone scythe" },
    { label: "🏹 Forest Ranger", prompt: "Elven Archer Ranger with leather tunic and recurve bow" },
    { label: "🤖 Cyber Commando", prompt: "Cyberpunk Tactical Commando with glowing visor and plasma rifle" }
  ];

  async function handleGenerate() {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await generateDiffusionSprite({
        prompt: prompt.trim(),
        style: style,
        steps: steps,
        guidanceScale: guidanceScale,
        seed: seed ? parseInt(seed) : null
      });
      setResult(data);
    } catch (err) {
      setError(err.message || "Diffusion generation failed");
    } finally {
      setLoading(false);
    }
  }

  function handleDownload(b64, filename) {
    const a = document.createElement("a");
    a.href = b64;
    a.download = filename || "diffusion_clean_sprite.png";
    a.click();
  }

  return (
    <div style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto", color: "#f1f5f9" }}>
      {/* Header Banner */}
      <div style={{
        background: "linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%)",
        border: "1px solid rgba(99, 102, 241, 0.3)",
        borderRadius: 14,
        padding: "1.75rem 2rem",
        marginBottom: "2rem",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.35)",
        backdropFilter: "blur(12px)"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: 6 }}>
              <span style={{ fontSize: 32 }}>✨</span>
              <h1 style={{ margin: 0, fontSize: 26, fontWeight: 700, letterSpacing: "-0.5px", background: "linear-gradient(135deg, #a5b4fc 0%, #ffffff 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                Latent Diffusion Sprite Studio
              </h1>
            </div>
            <p style={{ color: "#94a3b8", margin: 0, fontSize: 14 }}>
              Generates <strong>clean, isolated character sprites with transparent backgrounds (Alpha = 0)</strong> using Stable Diffusion + Intelligent Chroma Cutout.
            </p>
          </div>
          <div style={{
            background: "rgba(99, 102, 241, 0.15)",
            border: "1px solid rgba(99, 102, 241, 0.4)",
            color: "#a5b4fc",
            padding: "0.5rem 1rem",
            borderRadius: 20,
            fontSize: 13,
            fontWeight: 600,
            display: "flex",
            alignItems: "center",
            gap: 8
          }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#6366f1", display: "inline-block", boxShadow: "0 0 8px #6366f1" }}></span>
            Diffusion Engine Active
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div style={{
        background: "rgba(30, 41, 59, 0.5)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: 14,
        padding: "1.75rem",
        marginBottom: "2rem"
      }}>
        {/* Main Prompt Input */}
        <div style={{ marginBottom: "1.25rem" }}>
          <label style={{ display: "block", fontSize: 14, marginBottom: 8, color: "#cbd5e1", fontWeight: 600 }}>
            Describe Character or Game Concept:
          </label>
          <div style={{ display: "flex", gap: "0.75rem" }}>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleGenerate(); }}
              placeholder="e.g. 'A holy knight in golden plate armor with broadsword' or 'Arcane wizard casting void spells'"
              style={{
                flex: 1,
                padding: "0.85rem 1.15rem",
                borderRadius: 10,
                border: "1px solid #6366f1",
                background: "#0f172a",
                color: "#f8fafc",
                fontSize: 15,
                outline: "none",
                boxShadow: "0 0 16px rgba(99, 102, 241, 0.2)"
              }}
            />
            <button
              onClick={handleGenerate}
              disabled={loading}
              style={{
                padding: "0.85rem 2rem",
                borderRadius: 10,
                border: "none",
                background: loading ? "#475569" : "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
                color: "#ffffff",
                cursor: loading ? "not-allowed" : "pointer",
                fontSize: 15,
                fontWeight: 600,
                boxShadow: "0 4px 16px rgba(99, 102, 241, 0.4)",
                transition: "all 0.2s ease",
                whiteSpace: "nowrap"
              }}
            >
              {loading ? "Synthesizing Diffusion Sprite..." : "✨ Generate Diffusion Sprite"}
            </button>
          </div>

          {/* Quick Prompt Chips */}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
            <span style={{ fontSize: 12, color: "#64748b", alignSelf: "center", marginRight: 4 }}>Presets:</span>
            {quickPrompts.map((item, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setPrompt(item.prompt)}
                style={{
                  background: prompt === item.prompt ? "rgba(99, 102, 241, 0.3)" : "rgba(99, 102, 241, 0.12)",
                  border: prompt === item.prompt ? "1px solid #818cf8" : "1px solid rgba(99, 102, 241, 0.25)",
                  color: "#c7d2fe",
                  borderRadius: 16,
                  padding: "4px 12px",
                  fontSize: 12,
                  cursor: "pointer",
                  transition: "all 0.15s ease"
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        {/* Hyperparameters Row */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "1rem", paddingTop: "1rem", borderTop: "1px solid rgba(148, 163, 184, 0.1)" }}>
          <div>
            <label style={{ display: "block", fontSize: 12, marginBottom: 5, color: "#94a3b8" }}>Style Aesthetic</label>
            <select
              value={style}
              onChange={(e) => setStyle(e.target.value)}
              style={{ width: "100%", padding: "0.55rem 0.75rem", borderRadius: 8, border: "1px solid #475569", background: "#0f172a", color: "#f8fafc", fontSize: 13 }}
            >
              <option value="16-bit Retro RPG">16-bit Retro RPG</option>
              <option value="32-bit Arcade Adventure">32-bit Arcade Adventure</option>
              <option value="Dark Fantasy Gothic">Dark Fantasy Gothic</option>
              <option value="Cyberpunk 2D Platformer">Cyberpunk 2D Platformer</option>
            </select>
          </div>

          <div>
            <label style={{ display: "block", fontSize: 12, marginBottom: 5, color: "#94a3b8" }}>Inference Steps ({steps})</label>
            <input
              type="range"
              min={15}
              max={50}
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value))}
              style={{ width: "100%", accentColor: "#6366f1" }}
            />
          </div>

          <div>
            <label style={{ display: "block", fontSize: 12, marginBottom: 5, color: "#94a3b8" }}>Guidance Scale ({guidanceScale})</label>
            <input
              type="range"
              min={5.0}
              max={15.0}
              step={0.5}
              value={guidanceScale}
              onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
              style={{ width: "100%", accentColor: "#6366f1" }}
            />
          </div>

          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
              <label style={{ fontSize: 12, color: "#94a3b8" }}>Seed</label>
              <button
                type="button"
                onClick={() => setSeed(Math.floor(Math.random() * 999999).toString())}
                style={{ background: "transparent", border: "none", color: "#a5b4fc", fontSize: 11, cursor: "pointer" }}
              >
                🎲 Roll
              </button>
            </div>
            <input
              type="text"
              placeholder="Random"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              style={{ width: "100%", padding: "0.55rem 0.75rem", borderRadius: 8, border: "1px solid #475569", background: "#0f172a", color: "#f8fafc", fontSize: 13 }}
            />
          </div>
        </div>
      </div>

      {error && (
        <div style={{ background: "rgba(239, 68, 68, 0.15)", border: "1px solid rgba(239, 68, 68, 0.4)", color: "#fca5a5", borderRadius: 10, padding: "1rem", marginBottom: "1.5rem" }}>
          ⚠️ {error}
        </div>
      )}

      {/* Results View */}
      {result && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "2rem", alignItems: "start" }}>
          
          {/* Main Inspection Area */}
          <div style={{
            background: "rgba(30, 41, 59, 0.6)",
            border: "1px solid rgba(148, 163, 184, 0.15)",
            borderRadius: 14,
            padding: "1.75rem"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem", flexWrap: "wrap", gap: "1rem" }}>
              <h2 style={{ margin: 0, fontSize: 18, color: "#f8fafc", display: "flex", alignItems: "center", gap: 8 }}>
                <span>🛡️</span> Diffusion Character Synthesis Output
              </h2>
              
              {/* Tabs */}
              <div style={{ display: "flex", background: "#0f172a", borderRadius: 8, padding: 3 }}>
                {[
                  { id: "comparison", label: "Side-by-Side" },
                  { id: "prompts", label: "Diffusion Prompts" },
                  { id: "palette", label: "Palette (20-Color)" }
                ].map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setActiveTab(t.id)}
                    style={{
                      background: activeTab === t.id ? "#6366f1" : "transparent",
                      color: activeTab === t.id ? "#ffffff" : "#94a3b8",
                      border: "none",
                      borderRadius: 6,
                      padding: "5px 12px",
                      fontSize: 12,
                      fontWeight: 500,
                      cursor: "pointer"
                    }}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Tab 1: Comparison */}
            {activeTab === "comparison" && (
              <div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "1.5rem" }}>
                  {/* Raw Diffusion */}
                  <div>
                    <div style={{ fontSize: 13, color: "#cbd5e1", marginBottom: 8, fontWeight: 500, display: "flex", justifyContent: "space-between" }}>
                      <span>1. Raw Diffusion Output (RGB)</span>
                      <span style={{ color: "#ef4444", fontSize: 11 }}>Continuous / Solid Background</span>
                    </div>
                    <div style={{
                      background: "#0f172a",
                      border: "1px solid rgba(148, 163, 184, 0.2)",
                      borderRadius: 10,
                      padding: "1.5rem",
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center"
                    }}>
                      <img
                        src={result.raw_diffusion}
                        alt="Raw Diffusion"
                        style={{ maxWidth: 200, maxHeight: 200, width: "100%", objectFit: "contain", borderRadius: 6 }}
                      />
                    </div>
                  </div>

                  {/* Clean Cutout */}
                  <div>
                    <div style={{ fontSize: 13, color: "#cbd5e1", marginBottom: 8, fontWeight: 600, display: "flex", justifyContent: "space-between" }}>
                      <span>2. Clean Sprite Cutout (RGBA)</span>
                      <span style={{ color: "#34d399", fontSize: 11 }}>Alpha = 0 (100% Transparent)</span>
                    </div>
                    <div style={{
                      background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/16px 16px",
                      border: "2px solid #10b981",
                      borderRadius: 10,
                      padding: "1.5rem",
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center"
                    }}>
                      <img
                        src={result.clean_sprite}
                        alt="Clean Cutout Sprite"
                        style={{ width: 140, height: 140, imageRendering: "pixelated", filter: "drop-shadow(0 8px 16px rgba(0,0,0,0.45))" }}
                      />
                    </div>
                  </div>
                </div>

                <div style={{
                  background: "rgba(16, 185, 129, 0.1)",
                  border: "1px solid rgba(16, 185, 129, 0.25)",
                  borderRadius: 8,
                  padding: "0.85rem 1.15rem",
                  fontSize: 13,
                  color: "#e2e8f0",
                  lineHeight: 1.6
                }}>
                  ✅ <strong>Game Engine Ready:</strong> Solid background successfully eliminated! Character has been isolated, centered in 64x64 RGBA space, despeckled of stray noise, and palette-quantized for zero-lag drag-and-drop into Godot, Unity, Phaser, or Unreal.
                </div>
              </div>
            )}

            {/* Tab 2: Prompts */}
            {activeTab === "prompts" && (
              <div style={{ fontSize: 13, lineHeight: 1.6 }}>
                <div style={{ marginBottom: "1rem" }}>
                  <strong style={{ color: "#34d399", display: "block", marginBottom: 4 }}>✨ Engineered Positive Prompt:</strong>
                  <div style={{ background: "#0f172a", padding: "0.75rem", borderRadius: 8, border: "1px solid #334155", color: "#cbd5e1" }}>
                    {result.diffusion_params.positive_prompt}
                  </div>
                </div>
                <div>
                  <strong style={{ color: "#f87171", display: "block", marginBottom: 4 }}>🚫 Negative Prompt (Tilemap & Sheet Suppression):</strong>
                  <div style={{ background: "#0f172a", padding: "0.75rem", borderRadius: 8, border: "1px solid #334155", color: "#94a3b8" }}>
                    {result.diffusion_params.negative_prompt}
                  </div>
                </div>
              </div>
            )}

            {/* Tab 3: Palette */}
            {activeTab === "palette" && (
              <div>
                <p style={{ fontSize: 13, color: "#94a3b8", marginBottom: 12 }}>
                  Discrete 20-Color Quantized Palette (Median-Cut algorithm with +30% vibrancy boost):
                </p>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  {result.palette.map((hex, idx) => (
                    <div key={idx} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                      <div style={{ width: 36, height: 36, borderRadius: 6, background: hex, border: "1px solid rgba(255,255,255,0.2)" }} />
                      <span style={{ fontSize: 10, color: "#cbd5e1", fontFamily: "monospace" }}>{hex}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Export Sidebar */}
          <div style={{
            background: "rgba(30, 41, 59, 0.6)",
            border: "1px solid rgba(148, 163, 184, 0.15)",
            borderRadius: 14,
            padding: "1.75rem"
          }}>
            <h3 style={{ margin: "0 0 1rem 0", fontSize: 16, color: "#f8fafc" }}>
              Sprite Metadata & Export
            </h3>

            {/* Checkered Canvas Preview */}
            <div style={{
              background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/16px 16px",
              borderRadius: 10,
              padding: "1.5rem",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              marginBottom: "1.25rem",
              border: "1px solid rgba(148, 163, 184, 0.2)"
            }}>
              <img
                src={result.clean_sprite}
                alt="Selected Sprite"
                style={{ width: 128, height: 128, imageRendering: "pixelated", filter: "drop-shadow(0 6px 12px rgba(0,0,0,0.5))" }}
              />
            </div>

            <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.8, marginBottom: "1.5rem" }}>
              <div><strong>Resolution:</strong> 64x64 RGBA (Alpha Mask)</div>
              <div><strong>Transparency:</strong> <span style={{ color: "#34d399", fontWeight: 600 }}>{result.transparency_percent}% Alpha = 0</span></div>
              <div><strong>Format:</strong> Clean Cutout PNG</div>
              <div><strong>Diffusion Pipeline:</strong> Stable Diffusion 1.5 + LoRA</div>
              <div><strong>Compatibility:</strong> Godot, Unity, Phaser, Unreal</div>
            </div>

            <button
              onClick={() => handleDownload(result.clean_sprite, `diffusion_sprite_${result.diffusion_params.seed}.png`)}
              style={{
                width: "100%",
                padding: "0.75rem 1rem",
                borderRadius: 8,
                border: "none",
                background: "#10b981",
                color: "#ffffff",
                cursor: "pointer",
                fontSize: 14,
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                boxShadow: "0 4px 12px rgba(16, 185, 129, 0.35)",
                marginBottom: "0.75rem"
              }}
            >
              <span>📥</span> Download Clean Sprite (PNG)
            </button>

            <button
              onClick={() => handleDownload(result.raw_diffusion, `diffusion_raw_${result.diffusion_params.seed}.png`)}
              style={{
                width: "100%",
                padding: "0.6rem 1rem",
                borderRadius: 8,
                border: "1px solid #475569",
                background: "rgba(15, 23, 42, 0.6)",
                color: "#cbd5e1",
                cursor: "pointer",
                fontSize: 12,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 6
              }}
            >
              <span>🖼️</span> Download Raw Diffusion (RGB)
            </button>
          </div>

        </div>
      )}

      {/* Initial Placeholder */}
      {!result && !loading && (
        <div style={{
          textAlign: "center",
          padding: "4rem 1rem",
          background: "rgba(30, 41, 59, 0.3)",
          border: "1px dashed rgba(148, 163, 184, 0.2)",
          borderRadius: 14,
          color: "#94a3b8"
        }}>
          <div style={{ fontSize: 44, marginBottom: "0.75rem" }}>✨</div>
          <div style={{ fontSize: 18, fontWeight: 600, color: "#e2e8f0", marginBottom: "0.5rem" }}>
            Ready to Synthesize Diffusion Sprites
          </div>
          <p style={{ fontSize: 14, maxWidth: 500, margin: "0 auto", color: "#94a3b8" }}>
            Enter any character description above or click a preset chip, then click <strong>Generate Diffusion Sprite</strong>. The model will synthesize the character and extract a clean transparent cutout.
          </p>
        </div>
      )}
    </div>
  );
}
