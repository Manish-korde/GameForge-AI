import { useState, useEffect } from "react";
import { fetchDCGANClasses, generateDCGANCharacters } from "../services/api";

export default function DCGANStudio() {
  const [classes, setClasses] = useState([
    "🎲 Random Character (Fresh Synthesis)",
    "Hero (Knight)", "Mage (Sorcerer)", "Rogue (Thief)", "Paladin (Guardian)",
    "Archer (Ranger)", "Warrior (Barbarian)", "Cleric (Healer)", "Necromancer (Dark Mage)",
    "Goblin (Scout)", "Skeleton (Fighter)", "Slime (Creature)", "Assassin (Ninja)",
    "Monk (Martial)", "Druid (Shifter)", "Berserker (Heavy)", "Champion (Elite)"
  ]);
  const [selectedRole, setSelectedRole] = useState("🎲 Random Character (Fresh Synthesis)");
  const [count, setCount] = useState(4);
  const [seed, setSeed] = useState("");
  const [pixelArtMode, setPixelArtMode] = useState(true);
  const [sprites, setSprites] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dcganStatus, setDcganStatus] = useState("Checking...");
  const [selectedSprite, setSelectedSprite] = useState(null);

  useEffect(() => {
    fetchDCGANClasses().then((data) => {
      if (data?.classes && data.classes.length > 0) {
        setClasses(data.classes);
      }
      setDcganStatus(data?.dcgan_status || "Online");
    });
  }, []);

  async function handleGenerate() {
    setLoading(true);
    setError(null);
    try {
      const res = await generateDCGANCharacters({
        role: selectedRole,
        count: count,
        seed: seed ? parseInt(seed) : null,
        pixelArtMode: pixelArtMode,
      });
      setSprites(res.sprites || []);
      setDcganStatus(res.generator_status || "Loaded (DCGAN 169K 50-Epoch)");
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
    <div style={{ padding: "2rem", maxWidth: 1100, margin: "0 auto", color: "#f1f5f9" }}>
      {/* Header Banner */}
      <div style={{
        background: "linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: 12,
        padding: "1.75rem",
        marginBottom: "2rem",
        backdropFilter: "blur(10px)"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: 6 }}>
              <span style={{ fontSize: 28 }}>⚔️</span>
              <h2 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: "-0.5px" }}>
                DCGAN Character Generator
              </h2>
            </div>
            <p style={{ color: "#94a3b8", margin: 0, fontSize: 14 }}>
              Trained on <strong>169,094 character sprites</strong> for 50 Epochs (Dual T4 GPUs, 66,050 gradient updates).
            </p>
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
              Character Archetype
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
              Sprite Count (1–16)
            </label>
            <input
              type="number"
              min={1}
              max={16}
              value={count}
              onChange={(e) => setCount(Math.max(1, Math.min(16, parseInt(e.target.value) || 1)))}
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
                Seed (Optional)
              </label>
              <div style={{ display: "flex", gap: 4 }}>
                <button
                  type="button"
                  onClick={() => setSeed(Math.floor(Math.random() * 999999).toString())}
                  title="Roll Random Seed"
                  style={{
                    background: "rgba(99, 102, 241, 0.2)",
                    border: "1px solid rgba(99, 102, 241, 0.4)",
                    color: "#a5b4fc",
                    borderRadius: 4,
                    padding: "1px 6px",
                    fontSize: 11,
                    cursor: "pointer"
                  }}
                >
                  🎲 Roll
                </button>
                {seed && (
                  <button
                    type="button"
                    onClick={() => setSeed("")}
                    title="Clear Seed (Pure Random on each click)"
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
              placeholder="Leave empty for fresh random"
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

          {/* Pixel Art Clamping Toggle */}
          <div>
            <label style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              cursor: "pointer",
              padding: "0.6rem 0",
              color: "#cbd5e1",
              fontSize: 14,
              userSelect: "none"
            }}>
              <input
                type="checkbox"
                checked={pixelArtMode}
                onChange={(e) => setPixelArtMode(e.target.checked)}
                style={{ width: 18, height: 18, accentColor: "#6366f1" }}
              />
              <span>Palette Quantization (Crisp Edges)</span>
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
              {loading ? "Generating Sprites..." : "Generate Characters"}
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
          
          {/* Main Grid */}
          <div style={{
            background: "rgba(30, 41, 59, 0.5)",
            border: "1px solid rgba(148, 163, 184, 0.12)",
            borderRadius: 12,
            padding: "1.5rem"
          }}>
            <h3 style={{ margin: "0 0 1rem 0", fontSize: 16, color: "#f8fafc" }}>
              Generated Sprites ({sprites.length}) — Click to Inspect
            </h3>
            
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))",
              gap: "1rem"
            }}>
              {sprites.map((b64, idx) => (
                <div
                  key={idx}
                  onClick={() => setSelectedSprite(b64)}
                  style={{
                    background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/16px 16px",
                    border: selectedSprite === b64 ? "2px solid #6366f1" : "1px solid rgba(148, 163, 184, 0.2)",
                    borderRadius: 8,
                    padding: "0.75rem",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    cursor: "pointer",
                    transition: "transform 0.15s ease",
                  }}
                >
                  <img
                    src={b64}
                    alt={`Sprite ${idx + 1}`}
                    style={{
                      width: "100%",
                      maxWidth: 96,
                      height: 96,
                      objectFit: "contain",
                      imageRendering: "pixelated"
                    }}
                  />
                  <span style={{ fontSize: 11, color: "#94a3b8", marginTop: 6 }}>
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
                <div><strong>Resolution:</strong> 64x64 RGBA (Transparent)</div>
                <div><strong>Format:</strong> Clean PNG Sprite</div>
                <div><strong>Archetype:</strong> {selectedRole}</div>
                <div><strong>Engine Compatibility:</strong> Godot, Unity, Phaser, Unreal</div>
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
            Ready to generate game characters
          </div>
          <div style={{ fontSize: 13 }}>
            Choose an archetype above (Knight, Mage, Rogue, Necromancer, etc.) and click <strong>Generate Characters</strong>.
          </div>
        </div>
      )}
    </div>
  );
}
