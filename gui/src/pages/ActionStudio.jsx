import { useState, useEffect } from "react";
import { 
  fetchActionDiffusionStatus, 
  generateDiffusionPose, 
  generateDiffusionWalkCycle,
  SAMPLE_SPRITES_35 
} from "../services/api";

const POSE_OPTIONS_WALK = [
  { id: 0, label: "F1: Neutral Passing Lift", desc: "Feet aligned under torso, initiating cycle" },
  { id: 1, label: "F2: Left Forward Extension", desc: "Left leg strides forward, right plants" },
  { id: 2, label: "F3: Left Stride Apex", desc: "Maximum outward left-leg stride extension" },
  { id: 3, label: "F4: Left Return Phase", desc: "Left foot returns toward center plant" },
  { id: 4, label: "F5: Right Passing Lift", desc: "Neutral transition, right leg begins lift" },
  { id: 5, label: "F6: Right Forward Extension", desc: "Right leg strides forward, left plants" },
  { id: 6, label: "F7: Right Stride Apex", desc: "Maximum outward right-leg stride plant" },
  { id: 7, label: "F8: Right Return Phase", desc: "Right foot returns to complete loop" },
];

export default function ActionStudio() {
  const [statusInfo, setStatusInfo] = useState({ status: "Checking...", checkpoint_sha256: "" });
  
  // Controls
  const [action, setAction] = useState("walk"); // 'idle' | 'walk'
  const [poseId, setPoseId] = useState(2); // Default to Walk F3 Apex
  const [steps, setSteps] = useState(50);
  const [seed, setSeed] = useState("42");
  const [enableF7Patch, setEnableF7Patch] = useState(false);
  const [selectedRef, setSelectedRef] = useState(SAMPLE_SPRITES_35[1]?.url || "");
  const [customRefB64, setCustomRefB64] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const [error, setError] = useState(null);
  
  // Results
  const [resultSpriteB64, setResultSpriteB64] = useState(null);
  const [resultMetadata, setResultMetadata] = useState(null);
  const [walkCycleResult, setWalkCycleResult] = useState(null);
  const [activeTab, setActiveTab] = useState("studio"); // 'studio' | 'cycle' | 'metrics'
  
  // Animation Flipbook (8-frame walk loop or 2-frame comparison)
  const [isAnimating, setIsAnimating] = useState(true);
  const [animIndex, setAnimIndex] = useState(0);
  const [fps, setFps] = useState(8);

  useEffect(() => {
    fetchActionDiffusionStatus().then(info => {
      if (info) setStatusInfo(info);
    });
    const params = new URLSearchParams(window.location.search);
    if (params.get("auto") === "true") {
      fetch("/frontfacing_25/front_char_01.png")
        .then(r => r.blob())
        .then(blob => new Promise(res => {
          const reader = new FileReader();
          reader.onloadend = () => res(reader.result);
          reader.readAsDataURL(blob);
        }))
        .then(refB64 => {
          return generateDiffusionPose({
            action: "walk",
            poseId: 2,
            referenceB64: refB64,
            steps: 20,
            seed: 42
          });
        })
        .then(res => {
          if (res && res.status === "success") {
            setResultSpriteB64(res.image_b64);
            setResultMetadata({
              action: res.action,
              pose_name: res.pose_name,
              seed: res.seed,
              steps: res.steps,
              latency_ms: res.latency_ms,
              device: res.device,
              sha256: res.checkpoint_sha256
            });
          }
        })
        .catch(console.error);
    }
  }, []);

  // Walk Cycle Live Animation Ticker
  useEffect(() => {
    if (!isAnimating) return;
    
    if (walkCycleResult && walkCycleResult.frames && walkCycleResult.frames.length > 0) {
      const interval = setInterval(() => {
        setAnimIndex(idx => (idx + 1) % walkCycleResult.frames.length);
      }, 1000 / fps);
      return () => clearInterval(interval);
    } else if (resultSpriteB64) {
      const interval = setInterval(() => {
        setAnimIndex(idx => (idx === 0 ? 1 : 0));
      }, 1000 / fps);
      return () => clearInterval(interval);
    }
  }, [isAnimating, walkCycleResult, resultSpriteB64, fps]);

  const activeRefDisplayUrl = selectedRef === "custom" ? customRefB64 : selectedRef;

  const handleCustomUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setCustomRefB64(reader.result);
      setSelectedRef("custom");
    };
    reader.readAsDataURL(file);
  };

  // Helper to ensure base64
  const ensureBase64 = async (urlOrB64) => {
    if (!urlOrB64) return null;
    if (urlOrB64.startsWith("data:image")) return urlOrB64;
    try {
      const resp = await fetch(urlOrB64);
      const blob = await resp.blob();
      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(blob);
      });
    } catch (e) {
      return urlOrB64;
    }
  };

  // Single Pose Generation
  const handleGenerateSingle = async () => {
    if (!activeRefDisplayUrl) {
      setError("Please select or upload a character reference sprite.");
      return;
    }
    setLoading(true);
    setLoadingMsg(`Sampling single ${action.toUpperCase()} pose via 50-step reverse DDIM...`);
    setError(null);
    try {
      const res = await generateDiffusionPose({
        action,
        poseId: action === "idle" ? 0 : poseId,
        referenceB64: await ensureBase64(activeRefDisplayUrl),
        steps: parseInt(steps) || 50,
        seed: seed ? parseInt(seed) : null,
        enableF7Patch
      });

      if (res.status === "success") {
        setResultSpriteB64(res.image_b64);
        setResultMetadata({
          action: res.action,
          pose_name: res.pose_name,
          seed: res.seed,
          steps: res.steps,
          latency_ms: res.latency_ms,
          device: res.device,
          sha256: res.checkpoint_sha256
        });
      } else {
        throw new Error(res.error || "Generation failed");
      }
    } catch (err) {
      setError(err.message || "Failed to generate diffusion sprite pose.");
    } finally {
      setLoading(false);
    }
  };

  // Complete 8-Frame Walk Cycle Generation
  const handleGenerateWalkCycle = async () => {
    if (!activeRefDisplayUrl) {
      setError("Please select or upload a character reference sprite.");
      return;
    }
    setLoading(true);
    setLoadingMsg("Generating complete 8-frame walk cycle from pure Gaussian noise...");
    setError(null);
    try {
      const res = await generateDiffusionWalkCycle({
        referenceB64: await ensureBase64(activeRefDisplayUrl),
        steps: parseInt(steps) || 50,
        seed: seed ? parseInt(seed) : 42,
        enableF7Patch
      });

      if (res.status === "success") {
        setWalkCycleResult(res);
        setActiveTab("cycle");
      } else {
        throw new Error(res.error || "Walk cycle synthesis failed");
      }
    } catch (err) {
      setError(err.message || "Failed to generate complete walk cycle.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadSingle = () => {
    if (!resultSpriteB64) return;
    const a = document.createElement("a");
    a.href = resultSpriteB64;
    a.download = `gameforge_diffusion_${action}_pose_${resultMetadata?.pose_name?.replace(/[^a-zA-Z0-9]/g, '_') || 'pose'}.png`;
    a.click();
  };

  const handleDownloadSpriteSheet = () => {
    if (!walkCycleResult?.sprite_sheet_b64) return;
    const a = document.createElement("a");
    a.href = walkCycleResult.sprite_sheet_b64;
    a.download = `gameforge_walk_cycle_spritesheet_8frames_seed${walkCycleResult.seed}.png`;
    a.click();
  };

  const handleDownloadGif = () => {
    if (!walkCycleResult?.gif_b64) return;
    const a = document.createElement("a");
    a.href = walkCycleResult.gif_b64;
    a.download = `gameforge_walk_cycle_animated_seed${walkCycleResult.seed}.gif`;
    a.click();
  };

  return (
    <div style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto", color: "#f1f5f9" }}>
      {/* Header Banner */}
      <div style={{
        background: "linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: 14,
        padding: "1.75rem",
        marginBottom: "2rem",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.35)",
        backdropFilter: "blur(12px)"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: 6 }}>
              <span style={{ fontSize: 30 }}>🏃</span>
              <h2 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: "-0.5px" }}>
                Multi-Pose Velocity Diffusion Studio (Module 4)
              </h2>
            </div>
            <p style={{ color: "#94a3b8", margin: 0, fontSize: 14 }}>
              From-scratch Action & Pose Conditioned U-Net (1.98M parameters) synthesizing Idle and full 8-frame Walk cycles from a single reference sprite.
            </p>
          </div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4 }}>
            <div style={{
              background: "rgba(16, 185, 129, 0.15)",
              border: "1px solid rgba(16, 185, 129, 0.4)",
              color: "#6ee7b7",
              padding: "0.4rem 0.9rem",
              borderRadius: 20,
              fontSize: 12,
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              gap: 6
            }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#10b981", display: "inline-block" }}></span>
              Model: {statusInfo.status === "ready" ? "Phase F2 Validated Active" : statusInfo.status}
            </div>
            {statusInfo.checkpoint_sha256 && (
              <span style={{ fontSize: 10, color: "#64748b", fontFamily: "monospace" }}>
                SHA: {statusInfo.checkpoint_sha256.substring(0, 16)}...
              </span>
            )}
          </div>
        </div>

        {/* Tab Navigation */}
        <div style={{ display: "flex", gap: "0.75rem", marginTop: "1.25rem" }}>
          <button
            onClick={() => setActiveTab("studio")}
            style={{
              background: activeTab === "studio" ? "#6366f1" : "rgba(51, 65, 85, 0.5)",
              color: "#fff",
              border: "none",
              padding: "0.5rem 1.25rem",
              borderRadius: 8,
              fontSize: 14,
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.2s"
            }}
          >
            🎯 Pose Generator
          </button>
          <button
            onClick={() => setActiveTab("cycle")}
            style={{
              background: activeTab === "cycle" ? "#6366f1" : "rgba(51, 65, 85, 0.5)",
              color: "#fff",
              border: "none",
              padding: "0.5rem 1.25rem",
              borderRadius: 8,
              fontSize: 14,
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.2s"
            }}
          >
            🎞️ 8-Frame Walk Cycle
          </button>
          <button
            onClick={() => setActiveTab("metrics")}
            style={{
              background: activeTab === "metrics" ? "#6366f1" : "rgba(51, 65, 85, 0.5)",
              color: "#fff",
              border: "none",
              padding: "0.5rem 1.25rem",
              borderRadius: 8,
              fontSize: 14,
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.2s"
            }}
          >
            📊 Verified Metrics & Audit
          </button>
        </div>
      </div>

      {/* TAB 1: STUDIO */}
      {activeTab === "studio" && (
        <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1.3fr", gap: "2rem" }}>
          {/* Controls Column */}
          <div style={{
            background: "rgba(30, 41, 59, 0.6)",
            border: "1px solid rgba(148, 163, 184, 0.15)",
            borderRadius: 12,
            padding: "1.5rem"
          }}>
            <h3 style={{ margin: "0 0 1.25rem 0", fontSize: 17, fontWeight: 600, color: "#e2e8f0" }}>
              1. Reference & Conditioning
            </h3>

            {/* Step 1: Reference Sprite Picker */}
            <div style={{ marginBottom: "1.5rem" }}>
              <label style={{ display: "block", fontSize: 13, color: "#94a3b8", marginBottom: "0.5rem", fontWeight: 600 }}>
                STEP 1: SELECT REFERENCE FRONT-FACING SPRITE
              </label>
              <div style={{ display: "flex", gap: "0.5rem", overflowX: "auto", paddingBottom: "0.5rem", marginBottom: "0.75rem" }}>
                {SAMPLE_SPRITES_35.map((spr) => (
                  <img
                    key={spr.id}
                    src={spr.url}
                    alt={spr.label}
                    onClick={() => { setSelectedRef(spr.url); setCustomRefB64(null); }}
                    style={{
                      width: 48,
                      height: 48,
                      objectFit: "contain",
                      imageRendering: "pixelated",
                      background: "rgba(15, 23, 42, 0.8)",
                      borderRadius: 8,
                      cursor: "pointer",
                      border: selectedRef === spr.url ? "2px solid #6366f1" : "1px solid rgba(148, 163, 184, 0.2)",
                      padding: 2,
                      flexShrink: 0
                    }}
                    title={spr.label}
                  />
                ))}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                <label style={{
                  background: "rgba(51, 65, 85, 0.5)",
                  border: "1px dashed rgba(148, 163, 184, 0.3)",
                  padding: "0.45rem 0.9rem",
                  borderRadius: 6,
                  fontSize: 12,
                  cursor: "pointer",
                  color: "#cbd5e1"
                }}>
                  📁 Upload Custom Character Sprite PNG
                  <input type="file" accept="image/*" onChange={handleCustomUpload} style={{ display: "none" }} />
                </label>
                {selectedRef === "custom" && (
                  <span style={{ fontSize: 12, color: "#10b981", fontWeight: 600 }}>✓ Custom Sprite Loaded</span>
                )}
              </div>
            </div>

            {/* Step 2: Action Selection (Only Idle and Walk) */}
            <div style={{ marginBottom: "1.5rem" }}>
              <label style={{ display: "block", fontSize: 13, color: "#94a3b8", marginBottom: "0.5rem", fontWeight: 600 }}>
                STEP 2: SELECT ACTION CLASS
              </label>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
                <button
                  type="button"
                  onClick={() => { setAction("idle"); setPoseId(0); }}
                  style={{
                    padding: "0.85rem",
                    borderRadius: 10,
                    border: action === "idle" ? "2px solid #6366f1" : "1px solid rgba(148, 163, 184, 0.2)",
                    background: action === "idle" ? "rgba(99, 102, 241, 0.25)" : "rgba(15, 23, 42, 0.6)",
                    color: action === "idle" ? "#a5b4fc" : "#94a3b8",
                    fontWeight: 700,
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 8,
                    fontSize: 14
                  }}
                >
                  <span>🧍</span> Idle Stance (0)
                </button>
                <button
                  type="button"
                  onClick={() => setAction("walk")}
                  style={{
                    padding: "0.85rem",
                    borderRadius: 10,
                    border: action === "walk" ? "2px solid #10b981" : "1px solid rgba(148, 163, 184, 0.2)",
                    background: action === "walk" ? "rgba(16, 185, 129, 0.25)" : "rgba(15, 23, 42, 0.6)",
                    color: action === "walk" ? "#6ee7b7" : "#94a3b8",
                    fontWeight: 700,
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 8,
                    fontSize: 14
                  }}
                >
                  <span>🏃</span> South Walk (1)
                </button>
              </div>
            </div>

            {/* Step 3: Pose Selection */}
            {action === "walk" ? (
              <div style={{ marginBottom: "1.5rem" }}>
                <label style={{ display: "block", fontSize: 13, color: "#94a3b8", marginBottom: "0.5rem", fontWeight: 600 }}>
                  STEP 3: SELECT WALKING POSE FRAME (F1–F8)
                </label>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
                  {POSE_OPTIONS_WALK.map((p) => {
                    const isSelected = poseId === p.id;
                    const isApex = p.id === 2 || p.id === 6;
                    return (
                      <div
                        key={p.id}
                        onClick={() => setPoseId(p.id)}
                        style={{
                          padding: "0.55rem 0.75rem",
                          borderRadius: 8,
                          cursor: "pointer",
                          border: isSelected ? "2px solid #10b981" : "1px solid rgba(148, 163, 184, 0.2)",
                          background: isSelected ? "rgba(16, 185, 129, 0.2)" : "rgba(15, 23, 42, 0.6)",
                          color: isSelected ? "#6ee7b7" : "#cbd5e1"
                        }}
                      >
                        <div style={{ fontSize: 13, fontWeight: 700, display: "flex", justifyContent: "space-between" }}>
                          <span>{p.label}</span>
                          {isApex && <span style={{ fontSize: 10, color: "#fbbf24", background: "rgba(251, 191, 36, 0.2)", padding: "1px 5px", borderRadius: 4 }}>APEX</span>}
                        </div>
                        <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>
                          {p.desc}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div style={{ marginBottom: "1.5rem" }}>
                <label style={{ display: "block", fontSize: 13, color: "#94a3b8", marginBottom: "0.5rem", fontWeight: 600 }}>
                  STEP 3: POSE CONFIGURATION
                </label>
                <div style={{ padding: "0.75rem", background: "rgba(15, 23, 42, 0.6)", borderRadius: 8, border: "1px solid rgba(148, 163, 184, 0.2)", fontSize: 13, color: "#cbd5e1" }}>
                  Idle action supports <strong>Pose 0 (Neutral Standing Stance)</strong>. The model synthesizes the stationary grounded posture.
                </div>
              </div>
            )}

            {/* Inference Settings: Steps & Seed */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1.5rem" }}>
              <div>
                <label style={{ display: "block", fontSize: 12, color: "#94a3b8", marginBottom: "0.3rem", fontWeight: 600 }}>
                  DDIM REVERSE STEPS
                </label>
                <select
                  value={steps}
                  onChange={(e) => setSteps(Number(e.target.value))}
                  style={{
                    width: "100%",
                    padding: "0.55rem",
                    borderRadius: 8,
                    border: "1px solid rgba(148, 163, 184, 0.2)",
                    background: "rgba(15, 23, 42, 0.8)",
                    color: "#f1f5f9",
                    fontSize: 13
                  }}
                >
                  <option value={25}>25 Steps (Fast Preview ~0.6s)</option>
                  <option value={50}>50 Steps (Validated Gold Standard ~1.2s)</option>
                  <option value={100}>100 Steps (Deep Sampling)</option>
                </select>
              </div>
              <div>
                <label style={{ display: "block", fontSize: 12, color: "#94a3b8", marginBottom: "0.3rem", fontWeight: 600 }}>
                  DETERMINISTIC SEED
                </label>
                <input
                  type="number"
                  placeholder="e.g. 42 (leave blank for random)"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "0.55rem 0.75rem",
                    borderRadius: 8,
                    border: "1px solid rgba(148, 163, 184, 0.2)",
                    background: "rgba(15, 23, 42, 0.8)",
                    color: "#f1f5f9",
                    fontSize: 13
                  }}
                />
              </div>
            </div>

            {/* Experimental Bilateral Stride Mirror Patch Toggle */}
            <div style={{
              background: enableF7Patch ? "rgba(16, 185, 129, 0.15)" : "rgba(15, 23, 42, 0.6)",
              border: enableF7Patch ? "1px solid rgba(16, 185, 129, 0.4)" : "1px solid rgba(148, 163, 184, 0.2)",
              borderRadius: 8,
              padding: "0.65rem 0.85rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "0.85rem",
              transition: "all 0.15s ease"
            }}>
              <div>
                <div style={{ fontSize: 12, fontWeight: 700, color: enableF7Patch ? "#34d399" : "#cbd5e1" }}>
                  Bilateral F7 Stride Mirror Patch (Beta)
                </div>
                <div style={{ fontSize: 10, color: "#94a3b8" }}>
                  Feathered mirror from F3 left-apex to address 38.9% right-stride compression.
                </div>
              </div>
              <input
                type="checkbox"
                checked={enableF7Patch}
                onChange={(e) => setEnableF7Patch(e.target.checked)}
                style={{ width: 16, height: 16, accentColor: "#10b981", cursor: "pointer" }}
              />
            </div>

            {/* Action Buttons */}
            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              <button
                onClick={handleGenerateSingle}
                disabled={loading}
                style={{
                  width: "100%",
                  padding: "0.9rem",
                  borderRadius: 10,
                  border: "none",
                  background: loading 
                    ? "rgba(99, 102, 241, 0.4)" 
                    : "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                  color: "#ffffff",
                  fontSize: 14,
                  fontWeight: 700,
                  cursor: loading ? "not-allowed" : "pointer",
                  boxShadow: "0 4px 14px rgba(99, 102, 241, 0.35)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 8
                }}
              >
                {loading ? (
                  <>
                    <span style={{ animation: "spin 1s linear infinite" }}>🔄</span>
                    {loadingMsg || "Generating sprite..."}
                  </>
                ) : (
                  <>⚡ Generate {action.toUpperCase()} Pose (DDIM)</>
                )}
              </button>

              <button
                onClick={handleGenerateWalkCycle}
                disabled={loading}
                style={{
                  width: "100%",
                  padding: "0.85rem",
                  borderRadius: 10,
                  border: "1px solid rgba(16, 185, 129, 0.4)",
                  background: "rgba(16, 185, 129, 0.15)",
                  color: "#6ee7b7",
                  fontSize: 13,
                  fontWeight: 700,
                  cursor: loading ? "not-allowed" : "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 8
                }}
              >
                🎞️ Generate Full 8-Frame Walk Cycle
              </button>
            </div>

            {error && (
              <div style={{
                marginTop: "1rem",
                padding: "0.75rem",
                borderRadius: 8,
                background: "rgba(239, 68, 68, 0.15)",
                border: "1px solid rgba(239, 68, 68, 0.3)",
                color: "#fca5a5",
                fontSize: 13
              }}>
                ⚠️ {error}
              </div>
            )}
          </div>

          {/* Results Column */}
          <div style={{
            background: "rgba(30, 41, 59, 0.6)",
            border: "1px solid rgba(148, 163, 184, 0.15)",
            borderRadius: 12,
            padding: "1.5rem",
            display: "flex",
            flexDirection: "column"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
              <h3 style={{ margin: 0, fontSize: 17, fontWeight: 600, color: "#e2e8f0" }}>
                2. Generated Output & Comparison
              </h3>
              {resultMetadata && (
                <span style={{ fontSize: 12, color: "#10b981", fontWeight: 600, background: "rgba(16, 185, 129, 0.15)", padding: "3px 8px", borderRadius: 12 }}>
                  ⚡ Latency: {resultMetadata.latency_ms} ms
                </span>
              )}
            </div>

            {/* Dual Sprite Inspection Box */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "1.5rem",
              background: "rgba(15, 23, 42, 0.8)",
              border: "1px solid rgba(148, 163, 184, 0.1)",
              borderRadius: 10,
              padding: "1.5rem",
              marginBottom: "1.5rem"
            }}>
              {/* Reference Sprite */}
              <div style={{ textAlign: "center" }}>
                <span style={{ fontSize: 12, color: "#94a3b8", fontWeight: 600, display: "block", marginBottom: 8 }}>
                  INPUT REFERENCE SPRITE (64×64)
                </span>
                <div style={{
                  width: 140,
                  height: 140,
                  margin: "0 auto",
                  background: "radial-gradient(circle, rgba(51,65,85,0.4) 0%, rgba(15,23,42,0.8) 100%)",
                  borderRadius: 10,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  border: "1px solid rgba(148, 163, 184, 0.15)"
                }}>
                  {activeRefDisplayUrl ? (
                    <img
                      src={activeRefDisplayUrl}
                      alt="Reference"
                      style={{ width: 96, height: 96, objectFit: "contain", imageRendering: "pixelated" }}
                    />
                  ) : (
                    <span style={{ color: "#64748b", fontSize: 12 }}>No Reference</span>
                  )}
                </div>
              </div>

              {/* Generated Sprite */}
              <div style={{ textAlign: "center" }}>
                <span style={{ fontSize: 12, color: "#6ee7b7", fontWeight: 600, display: "block", marginBottom: 8 }}>
                  DIFFUSED {resultMetadata?.pose_name?.toUpperCase() || `${action.toUpperCase()} POSE`}
                </span>
                <div style={{
                  width: 140,
                  height: 140,
                  margin: "0 auto",
                  background: "radial-gradient(circle, rgba(99,102,241,0.15) 0%, rgba(15,23,42,0.8) 100%)",
                  borderRadius: 10,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  border: "1px solid rgba(99, 102, 241, 0.3)"
                }}>
                  {resultSpriteB64 ? (
                    <img
                      src={resultSpriteB64}
                      alt="Synthesized Action"
                      style={{ width: 96, height: 96, objectFit: "contain", imageRendering: "pixelated" }}
                    />
                  ) : (
                    <span style={{ color: "#64748b", fontSize: 12 }}>Awaiting Generation</span>
                  )}
                </div>
              </div>
            </div>

            {/* Metadata Badges */}
            {resultMetadata && (
              <div style={{
                background: "rgba(15, 23, 42, 0.5)",
                border: "1px solid rgba(148, 163, 184, 0.1)",
                borderRadius: 8,
                padding: "0.75rem 1rem",
                marginBottom: "1.5rem",
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "0.5rem",
                fontSize: 12
              }}>
                <div><strong>Action:</strong> <span style={{ color: "#a5b4fc" }}>{resultMetadata.action}</span></div>
                <div><strong>Pose:</strong> <span style={{ color: "#6ee7b7" }}>{resultMetadata.pose_name}</span></div>
                <div><strong>Seed:</strong> <span style={{ color: "#cbd5e1" }}>{resultMetadata.seed ?? "Random"}</span></div>
                <div><strong>Reverse Steps:</strong> <span style={{ color: "#cbd5e1" }}>{resultMetadata.steps} (DDIM)</span></div>
                <div><strong>Device:</strong> <span style={{ color: "#cbd5e1" }}>{resultMetadata.device}</span></div>
                <div><strong>Dimensions:</strong> <span style={{ color: "#cbd5e1" }}>64 × 64 RGBA</span></div>
              </div>
            )}

            {/* Action Buttons */}
            {resultSpriteB64 && (
              <button
                onClick={handleDownloadSingle}
                style={{
                  padding: "0.75rem",
                  borderRadius: 8,
                  border: "none",
                  background: "#3b82f6",
                  color: "#ffffff",
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 6
                }}
              >
                💾 Download Generated Sprite (PNG)
              </button>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: COMPLETE WALK CYCLE */}
      {activeTab === "cycle" && (
        <div style={{
          background: "rgba(30, 41, 59, 0.6)",
          border: "1px solid rgba(148, 163, 184, 0.15)",
          borderRadius: 12,
          padding: "1.75rem"
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
            <div>
              <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#e2e8f0" }}>
                Complete 8-Frame South Walk Cycle (Pure Gaussian DDIM)
              </h3>
              <p style={{ margin: "4px 0 0 0", color: "#94a3b8", fontSize: 13 }}>
                Each frame synthesized independently from pure Gaussian noise conditioning on the single reference sprite.
              </p>
            </div>
            {walkCycleResult && (
              <div style={{ display: "flex", gap: "0.75rem" }}>
                <button
                  onClick={handleDownloadSpriteSheet}
                  style={{
                    background: "rgba(16, 185, 129, 0.2)",
                    border: "1px solid rgba(16, 185, 129, 0.4)",
                    color: "#6ee7b7",
                    padding: "0.5rem 1rem",
                    borderRadius: 8,
                    fontSize: 13,
                    fontWeight: 600,
                    cursor: "pointer"
                  }}
                >
                  📥 Download Sprite Sheet (512×64)
                </button>
                <button
                  onClick={handleDownloadGif}
                  style={{
                    background: "rgba(99, 102, 241, 0.2)",
                    border: "1px solid rgba(99, 102, 241, 0.4)",
                    color: "#a5b4fc",
                    padding: "0.5rem 1rem",
                    borderRadius: 8,
                    fontSize: 13,
                    fontWeight: 600,
                    cursor: "pointer"
                  }}
                >
                  🎞️ Download Looping GIF
                </button>
              </div>
            )}
          </div>

          {!walkCycleResult ? (
            <div style={{ textAlign: "center", padding: "4rem 2rem", background: "rgba(15, 23, 42, 0.5)", borderRadius: 10 }}>
              <span style={{ fontSize: 40, display: "block", marginBottom: 12 }}>🎞️</span>
              <p style={{ color: "#94a3b8", fontSize: 14 }}>
                No walk cycle generated yet. Return to the <strong>Pose Generator</strong> tab and click <strong>Generate Full 8-Frame Walk Cycle</strong>.
              </p>
            </div>
          ) : (
            <div>
              {/* Animated Preview & Sprite Sheet */}
              <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: "2rem", marginBottom: "2rem", alignItems: "center" }}>
                {/* Looping Flipbook Box */}
                <div style={{
                  background: "rgba(15, 23, 42, 0.8)",
                  border: "1px solid rgba(148, 163, 184, 0.15)",
                  borderRadius: 12,
                  padding: "1.25rem",
                  textAlign: "center"
                }}>
                  <span style={{ fontSize: 12, color: "#94a3b8", fontWeight: 600, display: "block", marginBottom: 10 }}>
                    LOOPING ANIMATION (8 FPS)
                  </span>
                  <div style={{
                    width: 140,
                    height: 140,
                    margin: "0 auto",
                    background: "radial-gradient(circle, rgba(99,102,241,0.2) 0%, rgba(15,23,42,0.9) 100%)",
                    borderRadius: 10,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    border: "1px solid rgba(99, 102, 241, 0.3)"
                  }}>
                    <img
                      src={walkCycleResult.frames[animIndex]?.image_b64}
                      alt="Walk Loop"
                      style={{ width: 112, height: 112, objectFit: "contain", imageRendering: "pixelated" }}
                    />
                  </div>
                  <div style={{ fontSize: 12, color: "#6ee7b7", fontWeight: 700, marginTop: 8 }}>
                    Frame {animIndex + 1}: {walkCycleResult.frames[animIndex]?.pose_name}
                  </div>
                  <div style={{ display: "flex", justifyContent: "center", gap: 6, marginTop: 10 }}>
                    <button
                      onClick={() => setIsAnimating(!isAnimating)}
                      style={{
                        background: isAnimating ? "rgba(239, 68, 68, 0.2)" : "rgba(16, 185, 129, 0.2)",
                        color: isAnimating ? "#fca5a5" : "#6ee7b7",
                        border: "none",
                        padding: "0.25rem 0.6rem",
                        borderRadius: 6,
                        fontSize: 11,
                        fontWeight: 600,
                        cursor: "pointer"
                      }}
                    >
                      {isAnimating ? "⏸ Pause" : "▶ Play"}
                    </button>
                    <select
                      value={fps}
                      onChange={(e) => setFps(Number(e.target.value))}
                      style={{
                        background: "rgba(30, 41, 59, 0.8)",
                        color: "#94a3b8",
                        border: "1px solid rgba(148, 163, 184, 0.2)",
                        borderRadius: 6,
                        fontSize: 11,
                        padding: "0.2rem 0.4rem"
                      }}
                    >
                      <option value={4}>4 FPS</option>
                      <option value={6}>6 FPS</option>
                      <option value={8}>8 FPS (Standard)</option>
                      <option value={12}>12 FPS</option>
                    </select>
                  </div>
                </div>

                {/* Full Horizontal Sprite Strip */}
                <div style={{
                  background: "rgba(15, 23, 42, 0.8)",
                  border: "1px solid rgba(148, 163, 184, 0.15)",
                  borderRadius: 12,
                  padding: "1.25rem"
                }}>
                  <span style={{ fontSize: 12, color: "#94a3b8", fontWeight: 600, display: "block", marginBottom: 10 }}>
                    HORIZONTAL SPRITE SHEET (512 × 64 RGBA)
                  </span>
                  <div style={{
                    overflowX: "auto",
                    padding: "0.75rem",
                    background: "rgba(30, 41, 59, 0.5)",
                    borderRadius: 8,
                    border: "1px solid rgba(148, 163, 184, 0.1)"
                  }}>
                    <img
                      src={walkCycleResult.sprite_sheet_b64}
                      alt="Composite Sprite Sheet"
                      style={{ height: 64, imageRendering: "pixelated", display: "block" }}
                    />
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#64748b", marginTop: 8 }}>
                    <span>Total Latency: {walkCycleResult.total_latency_ms} ms</span>
                    <span>Device: {walkCycleResult.device}</span>
                    <span>Steps: {walkCycleResult.steps} (DDIM)</span>
                  </div>
                </div>
              </div>

              {/* Individual Frame Grid */}
              <h4 style={{ fontSize: 15, fontWeight: 600, color: "#cbd5e1", marginBottom: 12 }}>
                Sequential Frame Breakdown (F1 through F8)
              </h4>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem" }}>
                {walkCycleResult.frames.map((frame, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: "rgba(15, 23, 42, 0.7)",
                      border: "1px solid rgba(148, 163, 184, 0.15)",
                      borderRadius: 10,
                      padding: "1rem",
                      textAlign: "center"
                    }}
                  >
                    <div style={{ fontSize: 12, fontWeight: 700, color: "#6ee7b7", marginBottom: 6 }}>
                      Frame {idx + 1} (Pose {frame.pose_id})
                    </div>
                    <div style={{
                      width: 80,
                      height: 80,
                      margin: "0 auto",
                      background: "radial-gradient(circle, rgba(51,65,85,0.4) 0%, rgba(15,23,42,0.8) 100%)",
                      borderRadius: 8,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      border: "1px solid rgba(148, 163, 184, 0.1)"
                    }}>
                      <img
                        src={frame.image_b64}
                        alt={frame.pose_name}
                        style={{ width: 64, height: 64, objectFit: "contain", imageRendering: "pixelated" }}
                      />
                    </div>
                    <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 6 }}>
                      {frame.pose_name}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: VERIFIED METRICS & AUDIT */}
      {activeTab === "metrics" && (
        <div style={{
          background: "rgba(30, 41, 59, 0.6)",
          border: "1px solid rgba(148, 163, 184, 0.15)",
          borderRadius: 12,
          padding: "1.75rem"
        }}>
          <h3 style={{ margin: "0 0 1rem 0", fontSize: 18, fontWeight: 700, color: "#e2e8f0" }}>
            Multi-Pose Diffusion Validation &amp; Telemetry Matrix
          </h3>
          <p style={{ color: "#94a3b8", fontSize: 13, marginBottom: "1.5rem" }}>
            Evaluated across 100% of the unseen validation cohort (40 unique character identities × 9 conditions × 16 deterministic seeds = 5,760 samples).
          </p>

          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, textAlign: "left", marginBottom: "2rem" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid rgba(148, 163, 184, 0.2)", color: "#94a3b8" }}>
                <th style={{ padding: "0.75rem" }}>Evaluation Criterion</th>
                <th style={{ padding: "0.75rem" }}>Benchmark Target</th>
                <th style={{ padding: "0.75rem" }}>Measured (40 Val Cohort)</th>
                <th style={{ padding: "0.75rem" }}>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: "1px solid rgba(148, 163, 184, 0.1)" }}>
                <td style={{ padding: "0.75rem" }}>Pure Gaussian DDIM Stability</td>
                <td style={{ padding: "0.75rem" }}>100% Convergence</td>
                <td style={{ padding: "0.75rem", color: "#6ee7b7", fontWeight: 600 }}>100% (5,760 / 5,760)</td>
                <td style={{ padding: "0.75rem", color: "#10b981", fontWeight: 700 }}>MET</td>
              </tr>
              <tr style={{ borderBottom: "1px solid rgba(148, 163, 184, 0.1)" }}>
                <td style={{ padding: "0.75rem" }}>Idle Stance IoU</td>
                <td style={{ padding: "0.75rem" }}>≥ 0.850</td>
                <td style={{ padding: "0.75rem", color: "#6ee7b7", fontWeight: 600 }}>1.0000</td>
                <td style={{ padding: "0.75rem", color: "#10b981", fontWeight: 700 }}>EXCEEDED</td>
              </tr>
              <tr style={{ borderBottom: "1px solid rgba(148, 163, 184, 0.1)" }}>
                <td style={{ padding: "0.75rem" }}>Walk Poses Silhouette IoU</td>
                <td style={{ padding: "0.75rem" }}>≥ 0.820</td>
                <td style={{ padding: "0.75rem", color: "#6ee7b7", fontWeight: 600 }}>0.9426 (Combined: 0.9490)</td>
                <td style={{ padding: "0.75rem", color: "#10b981", fontWeight: 700 }}>EXCEEDED</td>
              </tr>
              <tr style={{ borderBottom: "1px solid rgba(148, 163, 184, 0.1)" }}>
                <td style={{ padding: "0.75rem" }}>Identity Preservation PSNR</td>
                <td style={{ padding: "0.75rem" }}>≥ 17.0 dB</td>
                <td style={{ padding: "0.75rem", color: "#6ee7b7", fontWeight: 600 }}>22.20 dB</td>
                <td style={{ padding: "0.75rem", color: "#10b981", fontWeight: 700 }}>EXCEEDED</td>
              </tr>
              <tr style={{ borderBottom: "1px solid rgba(148, 163, 184, 0.1)" }}>
                <td style={{ padding: "0.75rem" }}>Identity Structural SSIM</td>
                <td style={{ padding: "0.75rem" }}>≥ 0.850</td>
                <td style={{ padding: "0.75rem", color: "#6ee7b7", fontWeight: 600 }}>0.8816</td>
                <td style={{ padding: "0.75rem", color: "#10b981", fontWeight: 700 }}>MET</td>
              </tr>
              <tr style={{ borderBottom: "1px solid rgba(148, 163, 184, 0.1)" }}>
                <td style={{ padding: "0.75rem" }}>Walk Left-Leg Stride Error</td>
                <td style={{ padding: "0.75rem" }}>≤ 0.15 px</td>
                <td style={{ padding: "0.75rem", color: "#fbbf24", fontWeight: 600 }}>0.1692 px</td>
                <td style={{ padding: "0.75rem", color: "#fbbf24", fontWeight: 700 }}>MARGINAL</td>
              </tr>
              <tr style={{ borderBottom: "1px solid rgba(148, 163, 184, 0.1)" }}>
                <td style={{ padding: "0.75rem" }}>Walk Right-Leg Stride Error</td>
                <td style={{ padding: "0.75rem" }}>≤ 0.15 px</td>
                <td style={{ padding: "0.75rem", color: "#fca5a5", fontWeight: 600 }}>0.2087 px (F7 Apex: 0.277 px)</td>
                <td style={{ padding: "0.75rem", color: "#ef4444", fontWeight: 700 }}>FAILED / UNMET</td>
              </tr>
              <tr>
                <td style={{ padding: "0.75rem" }}>Pose Differentiation Rate</td>
                <td style={{ padding: "0.75rem" }}>Distinct configurations</td>
                <td style={{ padding: "0.75rem", color: "#6ee7b7", fontWeight: 600 }}>83.75% Distinguishable</td>
                <td style={{ padding: "0.75rem", color: "#10b981", fontWeight: 700 }}>MET</td>
              </tr>
            </tbody>
          </table>

          <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "1rem", borderRadius: 8, border: "1px solid rgba(148, 163, 184, 0.15)" }}>
            <h4 style={{ margin: "0 0 6px 0", fontSize: 14, color: "#fca5a5" }}>Documented Limitation: F7 Right-Stride Compression</h4>
            <p style={{ margin: 0, fontSize: 12, color: "#94a3b8", lineHeight: 1.5 }}>
              On walk cycle frame F7 (right stride apex), the model demonstrates <strong>38.9% stride compression</strong> (-0.349 px vs -0.561 px ground truth) due to pre-training asymmetry in C2 (which had previously only seen left stride apex F3). This documented architectural limitation is tracked for future training iterations.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
