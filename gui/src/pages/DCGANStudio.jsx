import { useState, useEffect } from "react";
import { fetchDCGANClasses, generateDCGANCharacters, fetchSafeSeed, fetchRealSprites } from "../services/api";

export default function DCGANStudio() {
  const [classes, setClasses] = useState([
    "🎲 Random Latent (Fresh Synthesis)",
    ...Array.from({ length: 16 }, (_, i) => `Latent Preset #${i + 1}`)
  ]);
  const [selectedRole, setSelectedRole] = useState("🎲 Random Latent (Fresh Synthesis)");
  const [count, setCount] = useState(16);
  const [seed, setSeed] = useState("1337");
  const [qualityFilter, setQualityFilter] = useState(true);
  const [filterLabel, setFilterLabel] = useState("best 16 of 32 by quality score");
  const [rollingSeed, setRollingSeed] = useState(false);
  const [sprites, setSprites] = useState([]);
  const [realSprites, setRealSprites] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dcganStatus, setDcganStatus] = useState("Checking...");
  const [selectedSprite, setSelectedSprite] = useState(null);

  useEffect(() => {
    // 1. Load real training sprites for comparison strip
    fetchRealSprites().then((items) => {
      if (items && items.length > 0) {
        setRealSprites(items.slice(0, 8));
      }
    }).catch(console.error);

    // 2. Fetch classes and handle auto-generation
    fetchDCGANClasses().then((data) => {
      if (data?.classes && data.classes.length > 0) {
        setClasses(data.classes);
      }
      setDcganStatus(data?.dcgan_status || "Online");
      
      const params = new URLSearchParams(window.location.search);
      if (params.get("auto") === "true") {
        const isQf = params.get("quality_filter") !== "false";
        setQualityFilter(isQf);
        const reqSeed = params.get("seed") !== null && params.get("seed") !== "" ? parseInt(params.get("seed")) : 1337;
        setSeed(reqSeed.toString());
        const reqCount = params.get("count") ? parseInt(params.get("count")) : 16;
        setCount(reqCount);

        generateDCGANCharacters({
          role: "🎲 Random Latent (Fresh Synthesis)",
          count: reqCount,
          seed: reqSeed,
          qualityFilter: isQf,
        }).then(res => {
          setSprites(res.sprites || []);
          setFilterLabel(res.filter_label || (isQf ? `best ${reqCount} of 32 by quality score` : "unfiltered (raw model)"));
          if (res.sprites && res.sprites.length > 0) {
            setSelectedSprite(res.sprites[0]);
          }
        }).catch(console.error);
      }
    });
  }, []);

  async function handleRollSeed() {
    setRollingSeed(true);
    try {
      const safe = await fetchSafeSeed();
      setSeed(safe.toString());
    } catch (e) {
      console.warn("Failed to fetch safe seed:", e);
      setSeed(Math.floor(Math.random() * 999999).toString());
    } finally {
      setRollingSeed(false);
    }
  }

  async function handleGenerate() {
    setLoading(true);
    setError(null);
    try {
      const res = await generateDCGANCharacters({
        role: selectedRole,
        count: count,
        seed: seed ? parseInt(seed) : 1337,
        qualityFilter: qualityFilter,
      });
      setSprites(res.sprites || []);
      setFilterLabel(res.filter_label || (qualityFilter ? `best ${count} of 32 by quality score` : "unfiltered (raw model)"));
      setDcganStatus(res.generator_status || "Loaded (Discrete DCGAN 50-Epoch)");
      if (res.sprites && res.sprites.length > 0) {
        setSelectedSprite(res.sprites[0]);
      }
    } catch (e) {
      setError(e.message || "Failed to generate sprites");
    } finally {
      setLoading(false);
    }
  }

  function handleDownload(b64, idx) {
    const a = document.createElement("a");
    a.href = b64;
    const cleanRole = selectedRole.replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase();
    a.download = `dcgan_${cleanRole}_${idx + 1}.png`;
    a.click();
  }

  return (
    <div style={{ padding: "2rem", maxWidth: 1180, margin: "0 auto", color: "#f1f5f9" }}>
      {/* Header Banner */}
      <div style={{
        background: "linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: 12,
        padding: "1.75rem",
        marginBottom: "1.75rem",
        backdropFilter: "blur(10px)"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: 6 }}>
              <span style={{ fontSize: 28 }}>⚔️</span>
              <h2 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: "-0.5px" }}>
                DCGAN Character Studio
              </h2>
            </div>
            <p style={{ color: "#94a3b8", margin: 0, fontSize: 14 }}>
              Trained on <strong>42,212 front-facing character sprites</strong> for 50 Epochs (Dual T4 GPUs, 16,450 gradient updates).
            </p>
            <div style={{ marginTop: 8, fontSize: 13, color: "#cbd5e1", display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ color: "#fbbf24" }}>ℹ️</span>
              <span><em>Unconditional generator. Not driven by the planner's JSON.</em></span>
            </div>
          </div>
          <div style={{
            background: "rgba(16, 185, 129, 0.15)",
            border: "1px solid rgba(16, 185, 129, 0.3)",
            color: "#34d399",
            padding: "0.4rem 0.9rem",
            borderRadius: 20,
            fontSize: 13,
            fontWeight: 600,
            display: "flex",
            alignItems: "center",
            gap: 6
          }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#10b981", display: "inline-block" }}></span>
            {dcganStatus}
          </div>
        </div>
      </div>

      {/* Real vs Generated Strip (Ground Truth Baseline) */}
      <div style={{
        background: "rgba(15, 23, 42, 0.6)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: 12,
        padding: "1.25rem 1.5rem",
        marginBottom: "1.75rem"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.85rem", flexWrap: "wrap", gap: "0.5rem" }}>
          <div>
            <h3 style={{ margin: "0 0 3px 0", fontSize: 15, fontWeight: 700, color: "#f8fafc", display: "flex", alignItems: "center", gap: 8 }}>
              <span>⚖️</span> Real vs Generated Strip (Ground Truth Baseline Comparison)
            </h3>
            <p style={{ margin: 0, fontSize: 12, color: "#94a3b8" }}>
              8 authentic training sprites from <code>evilsocket/alucard-sprites</code> next to 8 synthesized DCGAN sprites (Seed {seed || "1337"}).
            </p>
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8", background: "rgba(30, 41, 59, 0.8)", padding: "4px 8px", borderRadius: 6, border: "1px solid rgba(148, 163, 184, 0.2)" }}>
            Planar Coherence: <strong style={{ color: "#60a5fa" }}>46.18%</strong> (Real) vs <strong style={{ color: "#34d399" }}>48.45%</strong> (Model)
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1rem" }}>
          {/* Real Training Sprites Column */}
          <div style={{
            background: "rgba(30, 41, 59, 0.45)",
            borderRadius: 8,
            padding: "0.85rem",
            border: "1px solid rgba(59, 130, 246, 0.25)"
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "#60a5fa", marginBottom: "0.6rem", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#3b82f6" }}></span>
                Real Training Sprites (8 Audited Front-Facing)
              </span>
              <span style={{ fontSize: 10, color: "#93c5fd", background: "rgba(59, 130, 246, 0.2)", padding: "1px 6px", borderRadius: 4 }}>
                Dataset Baseline
              </span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 6 }}>
              {realSprites.map((b64, idx) => (
                <div key={idx} style={{
                  background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/12px 12px",
                  borderRadius: 6,
                  padding: 3,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  border: "1px solid rgba(148, 163, 184, 0.2)"
                }}>
                  <img src={b64} alt={`Real ${idx + 1}`} style={{ width: "100%", height: "auto", imageRendering: "pixelated" }} />
                </div>
              ))}
            </div>
          </div>

          {/* Generated Sprites Column */}
          <div style={{
            background: "rgba(30, 41, 59, 0.45)",
            borderRadius: 8,
            padding: "0.85rem",
            border: "1px solid rgba(16, 185, 129, 0.25)"
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "#34d399", marginBottom: "0.6rem", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#10b981" }}></span>
                Generated Sprites (First 8 of 16 {qualityFilter ? "Filtered" : "Raw"})
              </span>
              <span style={{ fontSize: 10, color: "#6ee7b7", background: "rgba(16, 185, 129, 0.2)", padding: "1px 6px", borderRadius: 4 }}>
                {qualityFilter ? "Filtered v2" : "Raw"}
              </span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 6 }}>
              {(sprites.length > 0 ? sprites.slice(0, 8) : realSprites).map((b64, idx) => (
                <div key={idx} style={{
                  background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/12px 12px",
                  borderRadius: 6,
                  padding: 3,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  border: "1px solid rgba(148, 163, 184, 0.2)"
                }}>
                  <img src={b64} alt={`Gen ${idx + 1}`} style={{ width: "100%", height: "auto", imageRendering: "pixelated" }} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div style={{
        background: "rgba(30, 41, 59, 0.5)",
        border: "1px solid rgba(148, 163, 184, 0.12)",
        borderRadius: 12,
        padding: "1.5rem",
        marginBottom: "2rem"
      }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1.25rem", alignItems: "flex-end" }}>
          
          {/* Role Preset */}
          <div>
            <label style={{ display: "block", fontSize: 13, marginBottom: 6, color: "#cbd5e1", fontWeight: 500 }}>
              Latent Synthesis Preset
            </label>
            <select
              value={selectedRole}
              onChange={(e) => setSelectedRole(e.target.value)}
              style={{
                width: "100%",
                padding: "0.6rem 0.85rem",
                borderRadius: 8,
                border: "1px solid #475569",
                background: "#0f172a",
                color: "#f8fafc",
                fontSize: 14
              }}
            >
              {classes.map((cls) => (
                <option key={cls} value={cls}>{cls}</option>
              ))}
            </select>
          </div>

          {/* Sprite Count */}
          <div>
            <label style={{ display: "block", fontSize: 13, marginBottom: 6, color: "#cbd5e1", fontWeight: 500 }}>
              Sprite Count (Default 16)
            </label>
            <input
              type="number"
              min={1}
              max={36}
              value={count}
              onChange={(e) => setCount(Math.max(1, Math.min(36, parseInt(e.target.value) || 16)))}
              style={{
                width: "100%",
                padding: "0.6rem 0.85rem",
                borderRadius: 8,
                border: "1px solid #475569",
                background: "#0f172a",
                color: "#f8fafc",
                fontSize: 14
              }}
            />
          </div>

          {/* Random Seed */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <label style={{ fontSize: 13, color: "#cbd5e1", fontWeight: 500 }}>
                Seed (Default 1337)
              </label>
              <div style={{ display: "flex", gap: 4 }}>
                <button
                  type="button"
                  onClick={handleRollSeed}
                  disabled={rollingSeed}
                  title="Roll Safe Seed (Foreground area >= 10%)"
                  style={{
                    background: "rgba(99, 102, 241, 0.2)",
                    border: "1px solid rgba(99, 102, 241, 0.4)",
                    color: "#a5b4fc",
                    borderRadius: 4,
                    padding: "1px 6px",
                    fontSize: 11,
                    cursor: rollingSeed ? "wait" : "pointer"
                  }}
                >
                  {rollingSeed ? "⏳..." : "🎲 Roll"}
                </button>
                {seed && (
                  <button
                    type="button"
                    onClick={() => setSeed("")}
                    title="Clear Seed"
                    style={{
                      background: "rgba(239, 68, 68, 0.2)",
                      border: "1px solid rgba(239, 68, 68, 0.4)",
                      color: "#fca5a5",
                      borderRadius: 4,
                      padding: "1px 6px",
                      fontSize: 11,
                      cursor: "pointer"
                    }}
                  >
                    ✕ Clear
                  </button>
                )}
              </div>
            </div>
            <input
              type="number"
              placeholder="e.g. 1337"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              style={{
                width: "100%",
                padding: "0.6rem 0.85rem",
                borderRadius: 8,
                border: "1px solid #475569",
                background: "#0f172a",
                color: "#f8fafc",
                fontSize: 14
              }}
            />
          </div>

          {/* Quality Filter Toggle */}
          <div>
            <label style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              cursor: "pointer",
              padding: "0.6rem 0",
              color: "#cbd5e1",
              fontSize: 13,
              userSelect: "none"
            }}
            title="Quality filter v2: generates 32 candidates, removes detached components within 2px, ranks by score"
            >
              <input
                type="checkbox"
                checked={qualityFilter}
                onChange={(e) => setQualityFilter(e.target.checked)}
                style={{ width: 18, height: 18, accentColor: "#6366f1" }}
              />
              <span>
                {qualityFilter ? (
                  <strong style={{ color: "#38bdf8" }}>Quality filter v2 (on)</strong>
                ) : (
                  <span style={{ color: "#f59e0b" }}>⚠️ Unfiltered (raw model)</span>
                )}
              </span>
            </label>
          </div>

          {/* Submit Button */}
          <div>
            <button
              onClick={handleGenerate}
              disabled={loading}
              style={{
                width: "100%",
                padding: "0.65rem 1.5rem",
                borderRadius: 8,
                border: "none",
                background: loading ? "#475569" : "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
                color: "#ffffff",
                cursor: loading ? "not-allowed" : "pointer",
                fontSize: 14,
                fontWeight: 600,
                boxShadow: "0 4px 12px rgba(99, 102, 241, 0.35)",
                transition: "all 0.2s ease"
              }}
            >
              {loading ? "Synthesizing 4x4 Grid..." : "Generate Characters"}
            </button>
          </div>

        </div>
      </div>

      {error && (
        <div style={{
          background: "rgba(239, 68, 68, 0.15)",
          border: "1px solid rgba(239, 68, 68, 0.3)",
          color: "#fca5a5",
          borderRadius: 8,
          padding: "1rem",
          marginBottom: "1.5rem",
          fontSize: 14
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* Generated Grid & Inspector */}
      {sprites.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "2rem", alignItems: "start" }}>
          
          {/* Main 4x4 Grid */}
          <div style={{
            background: "rgba(30, 41, 59, 0.5)",
            border: "1px solid rgba(148, 163, 184, 0.12)",
            borderRadius: 12,
            padding: "1.5rem"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem", flexWrap: "wrap", gap: "0.5rem" }}>
              <h3 style={{ margin: 0, fontSize: 16, color: "#f8fafc" }}>
                Generated Sprites ({sprites.length}) — 4×4 Grid
              </h3>
              <span style={{
                fontSize: 12,
                fontWeight: 600,
                padding: "3px 10px",
                borderRadius: 12,
                background: qualityFilter ? "rgba(56, 189, 248, 0.15)" : "rgba(245, 158, 11, 0.15)",
                color: qualityFilter ? "#38bdf8" : "#f59e0b",
                border: `1px solid ${qualityFilter ? "rgba(56, 189, 248, 0.3)" : "rgba(245, 158, 11, 0.3)"}`
              }}>
                {qualityFilter ? "best 16 of 32 by quality score" : "unfiltered (raw model)"}
              </span>
            </div>
            
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: "0.85rem"
            }}>
              {sprites.map((b64, idx) => (
                <div
                  key={idx}
                  onClick={() => setSelectedSprite(b64)}
                  style={{
                    background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/16px 16px",
                    border: selectedSprite === b64 ? "2px solid #6366f1" : "1px solid rgba(148, 163, 184, 0.2)",
                    borderRadius: 8,
                    padding: "0.6rem",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    cursor: "pointer",
                    transition: "all 0.15s ease",
                    boxShadow: selectedSprite === b64 ? "0 0 12px rgba(99, 102, 241, 0.4)" : "none"
                  }}
                >
                  <img
                    src={b64}
                    alt={`Sprite ${idx + 1}`}
                    style={{
                      width: "100%",
                      maxWidth: 104,
                      height: 104,
                      objectFit: "contain",
                      imageRendering: "pixelated"
                    }}
                  />
                  <span style={{ fontSize: 11, color: "#94a3b8", marginTop: 4, fontWeight: 500 }}>
                    #{idx + 1}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Inspector Card */}
          {selectedSprite && (
            <div style={{
              background: "rgba(30, 41, 59, 0.6)",
              border: "1px solid rgba(148, 163, 184, 0.15)",
              borderRadius: 12,
              padding: "1.5rem"
            }}>
              <h3 style={{ margin: "0 0 1rem 0", fontSize: 16, color: "#f8fafc" }}>
                Sprite Inspector & Export
              </h3>

              {/* Engine Canvas View */}
              <div style={{
                background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/20px 20px",
                borderRadius: 8,
                padding: "2rem",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                marginBottom: "1rem"
              }}>
                <img
                  src={selectedSprite}
                  alt="Inspector View"
                  style={{
                    width: 140,
                    height: 140,
                    imageRendering: "pixelated",
                    filter: "drop-shadow(0 8px 16px rgba(0,0,0,0.4))"
                  }}
                />
              </div>

              <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: "1rem", lineHeight: 1.6 }}>
                <div><strong>Resolution:</strong> 64x64 RGBA (8x Nearest-Neighbor)</div>
                <div><strong>Format:</strong> Clean Transparent PNG</div>
                <div><strong>Synthesis Mode:</strong> {selectedRole}</div>
                <div><strong>Filter:</strong> {qualityFilter ? "Quality filter v2 (best-of-32)" : "Raw Model (Unfiltered)"}</div>
                <div><strong>License:</strong> Standard PNG with alpha. Research use only (CC-BY-NC-SA).</div>
              </div>

              <button
                onClick={() => handleDownload(selectedSprite, 0)}
                style={{
                  width: "100%",
                  padding: "0.6rem 1rem",
                  borderRadius: 6,
                  border: "none",
                  background: "#10b981",
                  color: "#ffffff",
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: 600,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 6
                }}
              >
                <span>📥</span> Download Clean Sprite (PNG)
              </button>
            </div>
          )}

        </div>
      )}

      {/* Initial state placeholder */}
      {sprites.length === 0 && !loading && (
        <div style={{
          textAlign: "center",
          padding: "3.5rem 1rem",
          background: "rgba(30, 41, 59, 0.3)",
          border: "1px dashed rgba(148, 163, 184, 0.2)",
          borderRadius: 12,
          color: "#94a3b8"
        }}>
          <div style={{ fontSize: 40, marginBottom: "0.5rem" }}>🎮</div>
          <div style={{ fontSize: 16, fontWeight: 500, color: "#e2e8f0", marginBottom: "0.25rem" }}>
            Ready to generate game characters (4×4 Grid, 16 Sprites)
          </div>
          <div style={{ fontSize: 13 }}>
            Click <strong>Generate Characters</strong> to synthesize 16 sprites with Quality Filter v2.
          </div>
        </div>
      )}
    </div>
  );
}
