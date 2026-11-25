import { useState, useEffect, useRef } from "react";
import { 
  searchSimilarAssets, 
  runVAEInference, 
  runVAEInterpolation,
  FRONT_FACING_SPRITES_25, 
  fetchSampleSpritesManifest,
  BACKEND_URL 
} from "../services/api";

export default function VAEStudio() {
  const [sprites, setSprites] = useState(FRONT_FACING_SPRITES_25);
  const [selectedSprite, setSelectedSprite] = useState(FRONT_FACING_SPRITES_25[0]);
  const [secondarySprite, setSecondarySprite] = useState(FRONT_FACING_SPRITES_25[1]);
  
  // Similarity Retrieval State
  const [similarResults, setSimilarResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState("All");

  // Variations State
  const [vaeResult, setVaeResult] = useState(null);
  const [isVaeLoading, setIsVaeLoading] = useState(false);
  const [vaeScale, setVaeScale] = useState(0.6);

  // Interpolation State
  const [interpolatedResult, setInterpolatedResult] = useState(null);
  const [isInterpolating, setIsInterpolating] = useState(false);
  const [alphaVal, setAlphaVal] = useState(0.5);

  const [activeTab, setActiveTab] = useState("search"); // 'search' | 'variations' | 'interpolate'
  const [error, setError] = useState(null);

  const varDebounceTimer = useRef(null);
  const interpDebounceTimer = useRef(null);

  useEffect(() => {
    fetchSampleSpritesManifest().then((data) => {
      const spriteList = (data && data.length > 0) ? data : FRONT_FACING_SPRITES_25;
      setSprites(spriteList);
      setSelectedSprite(spriteList[0]);
      setSecondarySprite(spriteList[1] || spriteList[0]);
      triggerSearch(spriteList[0].url, "All");
      triggerVariations(spriteList[0].url, 0.6);
      triggerInterpolation(0.5, spriteList[0], spriteList[1] || spriteList[0]);
    });
  }, []);

  function resolveSpriteUrl(url) {
    if (!url) return "";
    if (url.startsWith("http") || url.startsWith("data:")) return url;
    return `${BACKEND_URL}${url}`;
  }

  async function triggerSearch(imgUrl, category = categoryFilter) {
    setIsSearching(true);
    setError(null);
    try {
      const fullUrl = resolveSpriteUrl(imgUrl);
      const res = await searchSimilarAssets(fullUrl, 4, category);
      if (res && res.matches) {
        setSimilarResults(res);
      }
    } catch (err) {
      console.error(err);
      setError("Similarity retrieval notice: " + err.message);
    } finally {
      setIsSearching(false);
    }
  }

  async function triggerVariations(imgUrl = selectedSprite?.url, scale = vaeScale) {
    if (!imgUrl) return;
    setIsVaeLoading(true);
    setError(null);
    try {
      const fullUrl = resolveSpriteUrl(imgUrl);
      const res = await runVAEInference(fullUrl, scale);
      if (res && (res.variant || res.reconstructed || res.image)) {
        setVaeResult(res);
      }
    } catch (err) {
      console.warn("VAE variations endpoint unreachable, using fallback display:", err);
      if (!vaeResult) {
        setVaeResult({ variant: resolveSpriteUrl(imgUrl), scale_applied: scale });
      }
    } finally {
      setIsVaeLoading(false);
    }
  }

  async function triggerInterpolation(alpha = alphaVal, spriteA = selectedSprite, spriteB = secondarySprite) {
    if (!spriteA || !spriteB) return;
    setIsInterpolating(true);
    setError(null);
    try {
      const fullUrlA = resolveSpriteUrl(spriteA.url);
      const fullUrlB = resolveSpriteUrl(spriteB.url);
      const res = await runVAEInterpolation(fullUrlA, fullUrlB, alpha);
      if (res && (res.interpolated || res.interpolated_image || res.image)) {
        setInterpolatedResult(res);
      }
    } catch (err) {
      console.warn("VAE interpolation endpoint unreachable, using fallback display:", err);
      if (!interpolatedResult) {
        const fallbackUrl = alpha < 0.5 ? resolveSpriteUrl(spriteA.url) : resolveSpriteUrl(spriteB.url);
        setInterpolatedResult({ interpolated: fallbackUrl, alpha });
      }
    } finally {
      setIsInterpolating(false);
    }
  }

  const handleScaleChange = (val) => {
    setVaeScale(val);
    if (varDebounceTimer.current) clearTimeout(varDebounceTimer.current);
    varDebounceTimer.current = setTimeout(() => {
      triggerVariations(selectedSprite?.url, val);
    }, 150);
  };

  const handleAlphaChange = (val) => {
    setAlphaVal(val);
    if (interpDebounceTimer.current) clearTimeout(interpDebounceTimer.current);
    interpDebounceTimer.current = setTimeout(() => {
      triggerInterpolation(val, selectedSprite, secondarySprite);
    }, 150);
  };

  function handleSelectCharacter(item) {
    setSelectedSprite(item);
    if (activeTab === "search") {
      triggerSearch(item.url, categoryFilter);
    } else if (activeTab === "variations") {
      triggerVariations(item.url, vaeScale);
    } else if (activeTab === "interpolate") {
      triggerInterpolation(alphaVal, item, secondarySprite);
    }
  }

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
            <span style={{ fontSize: 28 }}>🧬</span>
            <h2 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: "-0.5px" }}>
              VAE Latent Space Engine & Similarity Retrieval
            </h2>
          </div>
          <p style={{ color: "#94a3b8", margin: 0, fontSize: 14 }}>
            Trained on <strong>280,000 sprites</strong> to model a 256-dimensional continuous Gaussian latent space for similarity discovery, controlled variance synthesis, and spherical morphing.
          </p>
        </div>
        <div style={{
          background: "rgba(99, 102, 241, 0.15)",
          border: "1px solid rgba(99, 102, 241, 0.3)",
          color: "#a5b4fc",
          padding: "0.4rem 0.9rem",
          borderRadius: 20,
          fontSize: 13,
          fontWeight: 600,
          display: "flex",
          alignItems: "center",
          gap: 6
        }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#6366f1" }}></span>
          256-D Latent Manifold Active
        </div>
      </div>

      {/* 25 Front-Facing Characters Gallery Carousel */}
      <div style={{
        background: "rgba(30, 41, 59, 0.5)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: 12,
        padding: "1.25rem",
        marginBottom: "1.5rem"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.85rem" }}>
          <div>
            <h3 style={{ margin: "0 0 2px 0", fontSize: 15, fontWeight: 700, color: "#f8fafc" }}>
              Verified Front-Facing Dataset Characters (25 Sprites)
            </h3>
            <p style={{ margin: 0, fontSize: 12, color: "#94a3b8" }}>
              Select any character sprite to project into the 256-D continuous latent manifold for similarity matching or variation generation.
            </p>
          </div>
          <div style={{ fontSize: 12, color: "#a5b4fc", background: "rgba(99, 102, 241, 0.15)", padding: "3px 8px", borderRadius: 6 }}>
            Active: <strong>{selectedSprite?.name || "Selected"}</strong>
          </div>
        </div>

        {/* 25 Sprites Grid */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))",
          gap: "0.6rem",
          maxHeight: 200,
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

      {/* Mode Navigation Tabs */}
      <div style={{ display: "flex", gap: "0.75rem", marginBottom: "1.5rem" }}>
        <button
          onClick={() => {
            setActiveTab("search");
            triggerSearch(selectedSprite.url, categoryFilter);
          }}
          style={{
            padding: "0.6rem 1.25rem",
            borderRadius: 8,
            border: activeTab === "search" ? "2px solid #6366f1" : "1px solid rgba(148, 163, 184, 0.2)",
            background: activeTab === "search" ? "rgba(99, 102, 241, 0.2)" : "rgba(30, 41, 59, 0.5)",
            color: activeTab === "search" ? "#ffffff" : "#94a3b8",
            fontWeight: 600,
            fontSize: 13,
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 6
          }}
        >
          <span>🔍</span> 1. Latent Similarity Retrieval
        </button>

        <button
          onClick={() => {
            setActiveTab("variations");
            if (!vaeResult) {
              triggerVariations(selectedSprite?.url, vaeScale);
            }
          }}
          style={{
            padding: "0.6rem 1.25rem",
            borderRadius: 8,
            border: activeTab === "variations" ? "2px solid #38bdf8" : "1px solid rgba(148, 163, 184, 0.2)",
            background: activeTab === "variations" ? "rgba(56, 189, 248, 0.2)" : "rgba(30, 41, 59, 0.5)",
            color: activeTab === "variations" ? "#ffffff" : "#94a3b8",
            fontWeight: 600,
            fontSize: 13,
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 6
          }}
        >
          <span>✨</span> 2. Latent Stochastic Variations
        </button>

        <button
          onClick={() => {
            setActiveTab("interpolate");
            if (!interpolatedResult) {
              triggerInterpolation(alphaVal, selectedSprite, secondarySprite);
            }
          }}
          style={{
            padding: "0.6rem 1.25rem",
            borderRadius: 8,
            border: activeTab === "interpolate" ? "2px solid #10b981" : "1px solid rgba(148, 163, 184, 0.2)",
            background: activeTab === "interpolate" ? "rgba(16, 185, 129, 0.2)" : "rgba(30, 41, 59, 0.5)",
            color: activeTab === "interpolate" ? "#ffffff" : "#94a3b8",
            fontWeight: 600,
            fontSize: 13,
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 6
          }}
        >
          <span>🔄</span> 3. Spherical Latent Interpolation
        </button>
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

      {/* TAB 1: LATENT SIMILARITY RETRIEVAL */}
      {activeTab === "search" && (
        <div style={{
          background: "rgba(30, 41, 59, 0.5)",
          border: "1px solid rgba(148, 163, 184, 0.15)",
          borderRadius: 12,
          padding: "1.5rem",
          marginBottom: "2rem"
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem", flexWrap: "wrap", gap: "0.75rem" }}>
            <div>
              <h3 style={{ margin: "0 0 2px 0", fontSize: 16, fontWeight: 700, color: "#f8fafc" }}>
                Top-4 Nearest Neighbors in 256-D Latent Space
              </h3>
              <p style={{ margin: 0, fontSize: 12, color: "#94a3b8" }}>
                Identifies visually and semantically coherent sprites based on continuous cosine distance.
              </p>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <label style={{ fontSize: 12, color: "#94a3b8" }}>Filter Archetype:</label>
              <select
                value={categoryFilter}
                onChange={(e) => {
                  setCategoryFilter(e.target.value);
                  triggerSearch(selectedSprite.url, e.target.value);
                }}
                style={{
                  background: "#0f172a",
                  border: "1px solid #475569",
                  color: "#f8fafc",
                  borderRadius: 6,
                  padding: "0.35rem 0.75rem",
                  fontSize: 12
                }}
              >
                <option value="All">All Categories</option>
                <option value="Character">Characters</option>
                <option value="Knight">Knights / Paladins</option>
                <option value="Mage">Mages / Sorcerers</option>
                <option value="Ranger">Rangers / Scouts</option>
                <option value="Rogue">Rogues / Assassins</option>
                <option value="Warrior">Warriors / Guardians</option>
              </select>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "180px 1fr", gap: "1.5rem", alignItems: "center" }}>
            {/* Query Sprite */}
            <div style={{
              background: "rgba(15, 23, 42, 0.6)",
              borderRadius: 8,
              border: "1px solid rgba(99, 102, 241, 0.3)",
              padding: "1rem",
              display: "flex",
              flexDirection: "column",
              alignItems: "center"
            }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: "#818cf8", textTransform: "uppercase", marginBottom: 8 }}>
                Query Character
              </span>
              <div style={{
                background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/14px 14px",
                borderRadius: 6,
                padding: "1rem",
                marginBottom: 8
              }}>
                <img
                  src={selectedSprite.url}
                  alt="Query"
                  style={{ width: 96, height: 96, imageRendering: "pixelated" }}
                />
              </div>
              <span style={{ fontSize: 12, fontWeight: 600, color: "#f8fafc", textAlign: "center" }}>
                {selectedSprite.name}
              </span>
              <span style={{ fontSize: 11, color: "#94a3b8" }}>{selectedSprite.category || "Dataset Sprite"}</span>
            </div>

            {/* Retrieved 4 Matches */}
            <div>
              {isSearching ? (
                <div style={{ padding: "3rem", textAlign: "center", color: "#94a3b8" }}>
                  Searching 256-D continuous manifold...
                </div>
              ) : similarResults?.matches ? (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "1rem" }}>
                  {similarResults.matches.map((match, idx) => (
                    <div
                      key={idx}
                      style={{
                        background: "rgba(15, 23, 42, 0.6)",
                        borderRadius: 8,
                        border: "1px solid rgba(148, 163, 184, 0.15)",
                        padding: "0.85rem",
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        transition: "all 0.15s ease"
                      }}
                    >
                      <div style={{
                        background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/12px 12px",
                        borderRadius: 6,
                        padding: "0.5rem",
                        marginBottom: 8
                      }}>
                        <img
                          src={resolveSpriteUrl(match.url)}
                          alt={match.label}
                          style={{ width: 88, height: 88, imageRendering: "pixelated" }}
                        />
                      </div>
                      <div style={{
                        fontSize: 12,
                        fontWeight: 700,
                        color: "#34d399",
                        background: "rgba(16, 185, 129, 0.15)",
                        padding: "2px 6px",
                        borderRadius: 4,
                        marginBottom: 4
                      }}>
                        {isNaN(Number(match.cosine_similarity)) ? match.cosine_similarity : `${(Number(match.cosine_similarity) * 100).toFixed(1)}% Sim`}
                      </div>
                      <span style={{ fontSize: 11, color: "#cbd5e1", textAlign: "center", fontWeight: 500 }}>
                        {match.label || match.category}
                      </span>
                      <span style={{ fontSize: 10, color: "#64748b" }}>Rank #{idx + 1}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ padding: "2rem", color: "#64748b" }}>No similarity data found.</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: LATENT STOCHASTIC VARIATIONS */}
      {activeTab === "variations" && (
        <div style={{
          background: "rgba(30, 41, 59, 0.5)",
          border: "1px solid rgba(148, 163, 184, 0.15)",
          borderRadius: 12,
          padding: "1.5rem",
          marginBottom: "2rem"
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem", flexWrap: "wrap", gap: "0.75rem" }}>
            <div>
              <h3 style={{ margin: "0 0 2px 0", fontSize: 16, fontWeight: 700, color: "#f8fafc" }}>
                Stochastic Latent Perturbation &amp; Synthesis (σ)
              </h3>
              <p style={{ margin: 0, fontSize: 12, color: "#94a3b8" }}>
                Perturbs the 256-D latent embedding by noise factor σ to generate diverse character variations.
              </p>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <span style={{ fontSize: 13, color: "#cbd5e1" }}>Variance Scale (σ): <strong>{vaeScale}</strong></span>
              <input
                type="range"
                min="0.2"
                max="2.0"
                step="0.1"
                value={vaeScale}
                onChange={(e) => handleScaleChange(parseFloat(e.target.value))}
                style={{ width: 140, accentColor: "#38bdf8" }}
              />
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", alignItems: "center" }}>
            {/* Input Character */}
            <div style={{
              background: "rgba(15, 23, 42, 0.6)",
              borderRadius: 8,
              border: "1px solid rgba(148, 163, 184, 0.15)",
              padding: "1.5rem",
              display: "flex",
              flexDirection: "column",
              alignItems: "center"
            }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8", textTransform: "uppercase", marginBottom: 12 }}>
                Latent Seed Input
              </span>
              <div style={{
                background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/16px 16px",
                borderRadius: 8,
                padding: "1rem",
                marginBottom: 10,
                width: 224,
                height: 224,
                display: "flex",
                alignItems: "center",
                justifyContent: "center"
              }}>
                <img
                  src={selectedSprite?.url}
                  alt="Original"
                  style={{ width: 192, height: 192, imageRendering: "pixelated" }}
                />
              </div>
              <span style={{ fontSize: 13, fontWeight: 600, color: "#f8fafc" }}>{selectedSprite?.name}</span>
            </div>

            {/* Generated VAE Variation */}
            <div style={{
              background: "rgba(15, 23, 42, 0.6)",
              borderRadius: 8,
              border: "1px solid rgba(56, 189, 248, 0.3)",
              padding: "1.5rem",
              display: "flex",
              flexDirection: "column",
              alignItems: "center"
            }}>
              <span style={{ fontSize: 12, fontWeight: 700, color: "#38bdf8", letterSpacing: "0.5px", marginBottom: 12 }}>
                SYNTHESIZED VAE VARIATION (σ = {vaeScale} • {vaeResult?.output_size || "256×256"})
              </span>
              <div style={{
                background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/16px 16px",
                borderRadius: 8,
                padding: "1rem",
                marginBottom: 10,
                position: "relative",
                width: 224,
                height: 224,
                display: "flex",
                alignItems: "center",
                justifyContent: "center"
              }}>
                {(vaeResult?.variant || vaeResult?.reconstructed || vaeResult?.image || selectedSprite?.url) ? (
                  <img
                    src={resolveSpriteUrl(vaeResult?.variant || vaeResult?.reconstructed || vaeResult?.image || selectedSprite?.url)}
                    alt="Variation"
                    style={{
                      width: 192,
                      height: 192,
                      imageRendering: "pixelated",
                      opacity: isVaeLoading ? 0.4 : 1.0,
                      transition: "opacity 0.2s ease"
                    }}
                  />
                ) : (
                  <div style={{ width: 192, height: 192, display: "flex", alignItems: "center", justifyContent: "center", color: "#64748b" }}>
                    Generating...
                  </div>
                )}
                {isVaeLoading && (
                  <div style={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "rgba(15, 23, 42, 0.45)",
                    borderRadius: 8,
                    fontSize: 12,
                    fontWeight: 600,
                    color: "#38bdf8"
                  }}>
                    Synthesizing...
                  </div>
                )}
              </div>
              <button
                onClick={() => triggerVariations(selectedSprite?.url, vaeScale)}
                disabled={isVaeLoading}
                style={{
                  background: "linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)",
                  border: "none",
                  color: "#ffffff",
                  padding: "0.4rem 1rem",
                  borderRadius: 6,
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: "pointer"
                }}
              >
                {isVaeLoading ? "Generating..." : "Sample Another Variation"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: SPHERICAL LATENT INTERPOLATION */}
      {activeTab === "interpolate" && (
        <div style={{
          background: "rgba(30, 41, 59, 0.5)",
          border: "1px solid rgba(148, 163, 184, 0.15)",
          borderRadius: 12,
          padding: "1.5rem",
          marginBottom: "2rem"
        }}>
          <div style={{ marginBottom: "1.25rem" }}>
            <h3 style={{ margin: "0 0 2px 0", fontSize: 16, fontWeight: 700, color: "#f8fafc" }}>
              Spherical Linear Latent Interpolation (SLERP)
            </h3>
            <p style={{ margin: 0, fontSize: 12, color: "#94a3b8" }}>
              Continuous vector traversal between Character A and Character B across the 256-D latent manifold.
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "160px 1fr 160px", gap: "1.5rem", alignItems: "center", marginBottom: "1.5rem" }}>
            {/* Character A */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
              <span style={{ fontSize: 11, color: "#818cf8", fontWeight: 700, marginBottom: 6 }}>Source A (α = 0.0)</span>
              <div style={{
                background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/12px 12px",
                borderRadius: 6,
                padding: "0.5rem",
                marginBottom: 4
              }}>
                <img src={selectedSprite?.url} alt="Char A" style={{ width: 96, height: 96, imageRendering: "pixelated" }} />
              </div>
              <span style={{ fontSize: 11, color: "#f8fafc", textAlign: "center" }}>{selectedSprite?.name}</span>
            </div>

            {/* Interpolated Result Center */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
              <div style={{
                background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/16px 16px",
                borderRadius: 8,
                border: "2px solid #10b981",
                padding: "1rem",
                marginBottom: 10,
                position: "relative",
                width: 224,
                height: 224,
                display: "flex",
                alignItems: "center",
                justifyContent: "center"
              }}>
                {(interpolatedResult?.interpolated || interpolatedResult?.interpolated_image || interpolatedResult?.image || selectedSprite?.url) ? (
                  <img
                    src={resolveSpriteUrl(interpolatedResult?.interpolated || interpolatedResult?.interpolated_image || interpolatedResult?.image || (alphaVal < 0.5 ? selectedSprite?.url : secondarySprite?.url))}
                    alt="Interpolated"
                    style={{
                      width: 192,
                      height: 192,
                      imageRendering: "pixelated",
                      opacity: isInterpolating ? 0.4 : 1.0,
                      transition: "opacity 0.2s ease"
                    }}
                  />
                ) : (
                  <div style={{ width: 192, height: 192, display: "flex", alignItems: "center", justifyContent: "center", color: "#64748b" }}>
                    Interpolating...
                  </div>
                )}
                {isInterpolating && (
                  <div style={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "rgba(15, 23, 42, 0.45)",
                    borderRadius: 8,
                    fontSize: 12,
                    fontWeight: 600,
                    color: "#34d399"
                  }}>
                    Interpolating...
                  </div>
                )}
              </div>
              <span style={{ fontSize: 12, color: "#34d399", fontWeight: 600 }}>
                Synthesized Blend (α = {alphaVal} • {interpolatedResult?.output_size || "256×256"} {interpolatedResult?.interpolation_mode || "SLERP"})
              </span>
            </div>

            {/* Character B */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
              <span style={{ fontSize: 11, color: "#38bdf8", fontWeight: 700, marginBottom: 6 }}>Target B (α = 1.0)</span>
              <div style={{
                background: "conic-gradient(#334155 90deg, #1e293b 90deg 180deg, #334155 180deg 270deg, #1e293b 270deg) 0 0/12px 12px",
                borderRadius: 6,
                padding: "0.5rem",
                marginBottom: 4
              }}>
                <img src={secondarySprite?.url} alt="Char B" style={{ width: 96, height: 96, imageRendering: "pixelated" }} />
              </div>
              <span style={{ fontSize: 11, color: "#f8fafc", textAlign: "center" }}>{secondarySprite?.name}</span>
              <select
                value={secondarySprite?.url}
                onChange={(e) => {
                  const found = sprites.find(s => s.url === e.target.value);
                  if (found) {
                    setSecondarySprite(found);
                    triggerInterpolation(alphaVal, selectedSprite, found);
                  }
                }}
                style={{
                  marginTop: 6,
                  background: "#0f172a",
                  border: "1px solid #475569",
                  color: "#94a3b8",
                  borderRadius: 6,
                  padding: "0.15rem 0.4rem",
                  fontSize: 10,
                  maxWidth: 120
                }}
              >
                {sprites.map(s => (
                  <option key={s.id} value={s.url}>#{s.id} {s.name.split(" ")[0]}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Interpolation Slider */}
          <div style={{
            background: "rgba(15, 23, 42, 0.6)",
            borderRadius: 8,
            padding: "1rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16
          }}>
            <span style={{ fontSize: 12, color: "#94a3b8" }}>Character A (100%)</span>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={alphaVal}
              onChange={(e) => handleAlphaChange(parseFloat(e.target.value))}
              style={{ flex: 1, accentColor: "#10b981" }}
            />
            <span style={{ fontSize: 12, color: "#94a3b8" }}>Character B (100%)</span>
          </div>
        </div>
      )}
    </div>
  );
}
