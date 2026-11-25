import { useState, useEffect } from "react";
import { 
  runAEInference, 
  runNoisyAEInference, 
  FRONT_FACING_SPRITES_25, 
  fetchSampleSpritesManifest,
  BACKEND_URL 
} from "../services/api";

export default function AEStudio() {
  const [sprites, setSprites] = useState(FRONT_FACING_SPRITES_25);
  const [selectedSprite, setSelectedSprite] = useState(FRONT_FACING_SPRITES_25[0]);
  const [aeResult, setAeResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [noiseScale, setNoiseScale] = useState(0.15);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchSampleSpritesManifest().then((data) => {
      if (data && data.length > 0) {
        setSprites(data);
        setSelectedSprite(data[0]);
        executeInference(data[0].url);
      } else {
        executeInference(FRONT_FACING_SPRITES_25[0].url);
      }
    });
  }, []);

  async function executeInference(imgUrl) {
    setIsLoading(true);
    setError(null);
    try {
      const fullUrl = imgUrl.startsWith("http") || imgUrl.startsWith("data:") ? imgUrl : `${BACKEND_URL}${imgUrl}`;
      const res = await runAEInference(fullUrl);
      if (res && res.metrics) {
        setAeResult(res);
      } else {
        throw new Error("Invalid AE response structure");
      }
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to run Autoencoder inference");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleStressNoise() {
    if (!selectedSprite) return;
    setIsLoading(true);
    setError(null);
    try {
      const fullUrl = selectedSprite.url.startsWith("http") || selectedSprite.url.startsWith("data:") 
        ? selectedSprite.url 
        : `${BACKEND_URL}${selectedSprite.url}`;
      const res = await runNoisyAEInference(fullUrl, noiseScale);
      setAeResult(res);
    } catch (err) {
      setError("Noise stress test error: " + err.message);
    } finally {
      setIsLoading(false);
    }
  }

  function handleSelectCharacter(item) {
    setSelectedSprite(item);
    executeInference(item.url);
  }

  function handleFileUpload(e) {
    if (!e.target.files || !e.target.files[0]) return;
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = (event) => {
      const customItem = {
        id: Date.now(),
        label: `Uploaded: ${file.name}`,
        name: file.name,
        category: "Custom Upload",
        url: event.target.result
      };
      setSelectedSprite(customItem);
      executeInference(event.target.result);
    };
    reader.readAsDataURL(file);
  }

  const isDegenerate = aeResult?.metrics?.is_degenerate;
  const mse = aeResult?.metrics?.mse ?? 0;
  const p95Threshold = 0.001313;
  const isPassed = !isDegenerate && mse <= p95Threshold;

  return (
    <div style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto", color: "#f1f5f9" }}>
      {/* Header Banner */}
      <div style={{
        background: "linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: 12,
        padding: "1.75rem",
        marginBottom: "1.75rem",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        flexWrap: "wrap",
        gap: "1rem"
      }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: 6 }}>
            <span style={{ fontSize: 28 }}>🔍</span>
            <h2 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: "-0.5px" }}>
              Autoencoder Outlier Screening & Reconstruction
            </h2>
          </div>
          <p style={{ color: "#94a3b8", margin: 0, fontSize: 14 }}>
            Trained on <strong>280,000 game sprites</strong> to perform automated asset quality assurance, deterministic reconstruction, and P95 distribution outlier screening.
          </p>
        </div>
        <div style={{
          background: isPassed ? "rgba(16, 185, 129, 0.15)" : "rgba(245, 158, 11, 0.15)",
          border: `1px solid ${isPassed ? "rgba(16, 185, 129, 0.3)" : "rgba(245, 158, 11, 0.3)"}`,
          color: isPassed ? "#34d399" : "#fbbf24",
          padding: "0.4rem 0.9rem",
          borderRadius: 20,
          fontSize: 13,
          fontWeight: 600,
          display: "flex",
          alignItems: "center",
          gap: 6
        }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: isPassed ? "#10b981" : "#f59e0b" }}></span>
          {isPassed ? "In-Distribution (Certified)" : "Flagged Outlier (Requires Review)"}
        </div>
      </div>

      {/* 25 Front-Facing Characters Gallery Carousel */}
      <div style={{
        background: "rgba(30, 41, 59, 0.5)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: 12,
        padding: "1.25rem",
        marginBottom: "2rem"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.85rem" }}>
          <div>
            <h3 style={{ margin: "0 0 2px 0", fontSize: 15, fontWeight: 700, color: "#f8fafc" }}>
              Verified Front-Facing Dataset Characters (25 Sprites)
            </h3>
            <p style={{ margin: 0, fontSize: 12, color: "#94a3b8" }}>
              Select any strictly front-facing character sprite to run real-time Autoencoder reconstruction and outlier evaluation.
            </p>
          </div>
          <label style={{
            background: "rgba(99, 102, 241, 0.2)",
            border: "1px solid rgba(99, 102, 241, 0.4)",
            color: "#a5b4fc",
            padding: "0.35rem 0.75rem",
            borderRadius: 6,
            fontSize: 12,
            fontWeight: 600,
            cursor: "pointer",
            display: "inline-flex",
            alignItems: "center",
            gap: 6
          }}>
            <span>📤 Upload Custom PNG</span>
            <input type="file" accept="image/*" onChange={handleFileUpload} style={{ display: "none" }} />
          </label>
        </div>

        {/* 25 Sprites Grid */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))",
          gap: "0.6rem",
          maxHeight: 220,
          overflowY: "auto",
          paddingRight: 4
        }}>
          {sprites.map((item) => {
            const isSelected = selectedSprite?.url === item.url;
            return (
              <div
                key={item.id}
                onClick={() => handleSelectCharacter(item)}
                title={item.label || item.name}
                style={{
                  background: isSelected 
                    ? "rgba(99, 102, 241, 0.25)" 
                    : "conic-gradient(#1e293b 90deg, #0f172a 90deg 180deg, #1e293b 180deg 270deg, #0f172a 270deg) 0 0/12px 12px",
                  border: isSelected ? "2px solid #6366f1" : "1px solid rgba(148, 163, 184, 0.2)",
                  borderRadius: 8,
                  padding: "0.4rem",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                  boxShadow: isSelected ? "0 0 10px rgba(99, 102, 241, 0.4)" : "none"
                }}
              >
                <img
                  src={item.url}
                  alt={item.name}
                  style={{
                    width: 48,
                    height: 48,
                    objectFit: "contain",
                    imageRendering: "pixelated"
                  }}
                />
                <span style={{
                  fontSize: 10,
                  color: isSelected ? "#ffffff" : "#94a3b8",
                  marginTop: 4,
                  textAlign: "center",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  width: "100%"
                }}>
                  #{item.id} {item.name ? item.name.split(" ")[0] : ""}
                </span>
              </div>
            );
          })}
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

      {/* Main Reconstruction Workspace */}
      <div style={{ display: "grid", gridTemplateColumns: aeResult?.noisy_image ? "1fr 1fr 1fr" : "1fr 1fr", gap: "1.5rem", marginBottom: "2rem" }}>
        {/* Original Sprite Panel */}
        <div style={{
          background: "rgba(30, 41, 59, 0.5)",
          border: "1px solid rgba(148, 163, 184, 0.15)",
          borderRadius: 12,
          padding: "1.5rem",
          display: "flex",
          flexDirection: "column",
          alignItems: "center"
        }}>
          <div style={{ width: "100%", display: "flex", justifyContent: "space-between", marginBottom: "1rem" }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", textTransform: "uppercase" }}>Original Input Sprite</span>
            <span style={{ fontSize: 12, color: "#60a5fa" }}>{selectedSprite?.name || "Selected Sprite"}</span>
          </div>
          <div style={{
            background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/16px 16px",
            borderRadius: 8,
            border: "1px solid rgba(148, 163, 184, 0.2)",
            padding: "2rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 220,
            height: 220,
            marginBottom: "1rem"
          }}>
            <img
              src={aeResult?.original_processed || selectedSprite?.url}
              alt="Original"
              style={{
                width: 160,
                height: 160,
                objectFit: "contain",
                imageRendering: "pixelated",
                filter: "drop-shadow(0 6px 12px rgba(0,0,0,0.5))"
              }}
            />
          </div>
          <div style={{ fontSize: 12, color: "#94a3b8", textAlign: "center" }}>
            64×64 RGBA Clean Canvas (Nearest-Neighbor 128×128 Input)
          </div>
        </div>

        {/* Gaussian Noisy Input Panel (Shown during Stress Test) */}
        {aeResult?.noisy_image && (
          <div style={{
            background: "rgba(30, 41, 59, 0.5)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            borderRadius: 12,
            padding: "1.5rem",
            display: "flex",
            flexDirection: "column",
            alignItems: "center"
          }}>
            <div style={{ width: "100%", display: "flex", justifyContent: "space-between", marginBottom: "1rem" }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: "#fca5a5", textTransform: "uppercase" }}>Noisy Input</span>
              <span style={{ fontSize: 12, color: "#f87171" }}>σ = {aeResult?.metrics?.noise_scale ?? noiseScale}</span>
            </div>
            <div style={{
              background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/16px 16px",
              borderRadius: 8,
              border: "1px solid rgba(239, 68, 68, 0.25)",
              padding: "2rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 220,
              height: 220,
              marginBottom: "1rem"
            }}>
              <img
                src={aeResult.noisy_image}
                alt="Noisy Input"
                style={{
                  width: 160,
                  height: 160,
                  objectFit: "contain",
                  imageRendering: "pixelated",
                  filter: "drop-shadow(0 6px 12px rgba(0,0,0,0.5))"
                }}
              />
            </div>
            <div style={{ fontSize: 12, color: "#fca5a5", textAlign: "center" }}>
              Corrupted Sprite Fed into AE Encoder
            </div>
          </div>
        )}

        {/* Autoencoder Reconstructed Panel */}
        <div style={{
          background: "rgba(30, 41, 59, 0.5)",
          border: "1px solid rgba(148, 163, 184, 0.15)",
          borderRadius: 12,
          padding: "1.5rem",
          display: "flex",
          flexDirection: "column",
          alignItems: "center"
        }}>
          <div style={{ width: "100%", display: "flex", justifyContent: "space-between", marginBottom: "1rem" }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", textTransform: "uppercase" }}>AE Reconstructed Output</span>
            <span style={{ fontSize: 12, color: "#34d399" }}>Model: AE_280K_best</span>
          </div>
          <div style={{
            background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/16px 16px",
            borderRadius: 8,
            border: "1px solid rgba(148, 163, 184, 0.2)",
            padding: "2rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 220,
            height: 220,
            marginBottom: "1rem"
          }}>
            {isLoading ? (
              <div style={{ color: "#94a3b8", fontSize: 14 }}>Neural reconstruction running...</div>
            ) : aeResult?.reconstructed ? (
              <img
                src={aeResult.reconstructed}
                alt="Reconstructed"
                style={{
                  width: 160,
                  height: 160,
                  objectFit: "contain",
                  imageRendering: "pixelated",
                  filter: "drop-shadow(0 6px 12px rgba(0,0,0,0.5))"
                }}
              />
            ) : (
              <div style={{ color: "#64748b" }}>Select a sprite to reconstruct</div>
            )}
          </div>
          <div style={{ fontSize: 12, color: "#94a3b8", textAlign: "center" }}>
            Deterministic Outlier Screening & Silhouette Verification
          </div>
        </div>
      </div>

      {/* Outlier Decision Banner */}
      {aeResult?.metrics && (
        <div style={{
          background: isPassed ? "rgba(16, 185, 129, 0.1)" : "rgba(245, 158, 11, 0.1)",
          border: `1px solid ${isPassed ? "rgba(16, 185, 129, 0.3)" : "rgba(245, 158, 11, 0.3)"}`,
          borderRadius: 12,
          padding: "1.25rem",
          marginBottom: "2rem"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ fontSize: 24 }}>{isPassed ? "🟢" : "⚠️"}</span>
            <div>
              <h4 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: isPassed ? "#34d399" : "#fbbf24" }}>
                {isPassed ? "Automated QA Gate: Passed In-Distribution Screening" : "Automated QA Gate: Outlier Flagged for Manual Review"}
              </h4>
              <p style={{ margin: "4px 0 0 0", fontSize: 13, color: "#cbd5e1" }}>
                Reconstruction MSE is <strong style={{ color: isPassed ? "#34d399" : "#f87171" }}>{mse.toFixed(6)}</strong> (P95 threshold: <code style={{ color: "#a5b4fc" }}>0.001313</code>). 
                {isPassed 
                  ? " Sprite morphology and pixel palette match the learned 280,000 game asset distribution." 
                  : " Asset contains high reconstruction divergence or novel high-frequency patterns outside baseline confidence."}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Real-Time Quantitative Metrics Grid */}
      {aeResult?.metrics && (
        <div style={{
          background: "rgba(30, 41, 59, 0.5)",
          border: "1px solid rgba(148, 163, 184, 0.15)",
          borderRadius: 12,
          padding: "1.5rem",
          marginBottom: "2rem"
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem", marginBottom: "1.25rem" }}>
            <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "#f8fafc" }}>
              Per-Reconstruction Quantitative Metrics &mdash; <span style={{ color: "#60a5fa" }}>{selectedSprite?.name || "Selected Sprite"}</span>
            </h3>
            <span style={{
              fontSize: 11,
              fontWeight: 600,
              padding: "3px 10px",
              borderRadius: 999,
              background: aeResult.noisy_image ? "rgba(239, 68, 68, 0.15)" : "rgba(16, 185, 129, 0.15)",
              color: aeResult.noisy_image ? "#fca5a5" : "#34d399",
              border: `1px solid ${aeResult.noisy_image ? "rgba(239, 68, 68, 0.3)" : "rgba(16, 185, 129, 0.3)"}`
            }}>
              {aeResult.noisy_image ? `Gaussian Denoising Mode (σ=${aeResult.metrics.noise_scale ?? noiseScale})` : "Clean Sprite Reconstruction"}
            </span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "1rem" }}>
            <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "1rem", borderRadius: 8, border: "1px solid rgba(148, 163, 184, 0.1)" }}>
              <div style={{ fontSize: 11, color: "#94a3b8" }}>Reconstruction MSE</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#38bdf8", marginTop: 4 }}>{aeResult.metrics.mse.toFixed(6)}</div>
              <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Target: ≤ 0.001313</div>
            </div>

            <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "1rem", borderRadius: 8, border: "1px solid rgba(148, 163, 184, 0.1)" }}>
              <div style={{ fontSize: 11, color: "#94a3b8" }}>Mean Abs Error (MAE)</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#38bdf8", marginTop: 4 }}>{aeResult.metrics.mae !== undefined ? aeResult.metrics.mae.toFixed(6) : "—"}</div>
              <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Per-pixel L1 deviation</div>
            </div>

            <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "1rem", borderRadius: 8, border: "1px solid rgba(148, 163, 184, 0.1)" }}>
              <div style={{ fontSize: 11, color: "#94a3b8" }}>PSNR Fidelity</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#34d399", marginTop: 4 }}>{aeResult.metrics.psnr !== undefined ? aeResult.metrics.psnr.toFixed(2) : "—"} dB</div>
              <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Higher is sharper</div>
            </div>

            <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "1rem", borderRadius: 8, border: "1px solid rgba(148, 163, 184, 0.1)" }}>
              <div style={{ fontSize: 11, color: "#94a3b8" }}>Structural SSIM</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#a5b4fc", marginTop: 4 }}>{aeResult.metrics.ssim !== undefined ? aeResult.metrics.ssim.toFixed(4) : "—"}</div>
              <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Range [0.0, 1.0]</div>
            </div>

            <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "1rem", borderRadius: 8, border: "1px solid rgba(148, 163, 184, 0.1)" }}>
              <div style={{ fontSize: 11, color: "#94a3b8" }}>Exact Pixel Match</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#34d399", marginTop: 4 }}>{aeResult.metrics.exact_match !== undefined ? aeResult.metrics.exact_match.toFixed(2) : "—"}%</div>
              <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Within ±0.05 tolerance</div>
            </div>

            <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "1rem", borderRadius: 8, border: "1px solid rgba(148, 163, 184, 0.1)" }}>
              <div style={{ fontSize: 11, color: "#94a3b8" }}>Alpha Mask IoU</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#34d399", marginTop: 4 }}>{aeResult.metrics.alpha_iou !== undefined ? aeResult.metrics.alpha_iou.toFixed(2) : "—"}%</div>
              <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Silhouette preservation</div>
            </div>

            <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "1rem", borderRadius: 8, border: "1px solid rgba(148, 163, 184, 0.1)" }}>
              <div style={{ fontSize: 11, color: "#94a3b8" }}>Color Palette Match</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#fbbf24", marginTop: 4 }}>{aeResult.metrics.color_match !== undefined ? aeResult.metrics.color_match.toFixed(2) : "—"}%</div>
              <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>Foreground consistency</div>
            </div>
          </div>
        </div>
      )}

      {/* Stress Test & Controls Bar */}
      <div style={{
        background: "rgba(30, 41, 59, 0.5)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: 12,
        padding: "1.25rem",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: "1rem"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 13, color: "#cbd5e1", fontWeight: 600 }}>Anomaly Stress Test:</span>
          <span style={{ fontSize: 12, color: "#94a3b8" }}>Noise scale (σ): {noiseScale}</span>
          <input
            type="range"
            min="0.05"
            max="0.50"
            step="0.05"
            value={noiseScale}
            onChange={(e) => setNoiseScale(parseFloat(e.target.value))}
            style={{ width: 140, accentColor: "#6366f1" }}
          />
          <button
            onClick={handleStressNoise}
            disabled={isLoading}
            style={{
              background: "rgba(239, 68, 68, 0.2)",
              border: "1px solid rgba(239, 68, 68, 0.4)",
              color: "#fca5a5",
              padding: "0.4rem 0.8rem",
              borderRadius: 6,
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer"
            }}
          >
            Inject Gaussian Noise
          </button>
        </div>

        <button
          onClick={() => executeInference(selectedSprite.url)}
          disabled={isLoading}
          style={{
            background: "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
            border: "none",
            color: "#ffffff",
            padding: "0.5rem 1.25rem",
            borderRadius: 6,
            fontSize: 13,
            fontWeight: 600,
            cursor: "pointer",
            boxShadow: "0 2px 8px rgba(99, 102, 241, 0.3)"
          }}
        >
          {isLoading ? "Running..." : "Re-evaluate Active Sprite"}
        </button>
      </div>
    </div>
  );
}
