import { useState } from "react";

const API = "http://localhost:8000";
const CATEGORIES = ["stone", "grass", "wood", "brick", "dirt"];

export default function TextureStudio() {
  const [category, setCategory] = useState("stone");
  const [count, setCount] = useState(4);
  const [textures, setTextures] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ganStatus, setGanStatus] = useState(null);

  async function handleGenerate() {
    setLoading(true);
    setError(null);
    setTextures([]);
    try {
      const form = new FormData();
      form.append("category", category);
      form.append("count", count);
      const res = await fetch(`${API}/gan/generate_texture`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Generation failed");
      setTextures(data.textures || []);
      setGanStatus(data.generator_status);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleDownload(b64, idx) {
    const a = document.createElement("a");
    a.href = b64;
    a.download = `${category}_texture_${idx + 1}.png`;
    a.click();
  }

  return (
    <div style={{ padding: "2rem", maxWidth: 900, margin: "0 auto" }}>
      <h2 style={{ marginBottom: 4 }}>🎨 Texture Studio</h2>
      <p style={{ color: "#888", marginBottom: "1.5rem", fontSize: 14 }}>
        GAN-powered surface texture generation. Select a material category and generate
        tileable texture candidates for your game environment.
      </p>

      <div style={{ display: "flex", gap: "1rem", alignItems: "flex-end", marginBottom: "1.5rem", flexWrap: "wrap" }}>
        <div>
          <label style={{ display: "block", fontSize: 13, marginBottom: 4, color: "#aaa" }}>
            Material Category
          </label>
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            style={{ padding: "0.5rem 1rem", borderRadius: 6, border: "1px solid #444", background: "#1a1a2e", color: "#fff", fontSize: 14 }}
          >
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>
            ))}
          </select>
        </div>

        <div>
          <label style={{ display: "block", fontSize: 13, marginBottom: 4, color: "#aaa" }}>
            Count (1–8)
          </label>
          <input
            type="number"
            min={1}
            max={8}
            value={count}
            onChange={(e) => setCount(Math.max(1, Math.min(8, parseInt(e.target.value) || 1)))}
            style={{ padding: "0.5rem", width: 70, borderRadius: 6, border: "1px solid #444", background: "#1a1a2e", color: "#fff", fontSize: 14 }}
          />
        </div>

        <button
          onClick={handleGenerate}
          disabled={loading}
          style={{
            padding: "0.55rem 1.5rem", borderRadius: 6, border: "none",
            background: loading ? "#555" : "#6c63ff", color: "#fff",
            cursor: loading ? "not-allowed" : "pointer", fontSize: 14, fontWeight: 600
          }}
        >
          {loading ? "Generating…" : "Generate Textures"}
        </button>
      </div>

      {ganStatus && ganStatus !== "Loaded" && (
        <div style={{ background: "#2a1a1a", border: "1px solid #c0392b", borderRadius: 6, padding: "0.75rem 1rem", marginBottom: "1rem", fontSize: 13, color: "#e74c3c" }}>
          GAN Status: {ganStatus}. Train the model first — see <code>gan/Notebook/GAN_Texture_Studio.ipynb</code>.
        </div>
      )}

      {error && (
        <div style={{ background: "#2a1a1a", border: "1px solid #c0392b", borderRadius: 6, padding: "0.75rem 1rem", marginBottom: "1rem", fontSize: 13, color: "#e74c3c" }}>
          {error}
        </div>
      )}

      {textures.length > 0 && (
        <>
          <div style={{ fontSize: 13, color: "#aaa", marginBottom: "0.75rem" }}>
            {textures.length} texture{textures.length > 1 ? "s" : ""} generated for <strong>{category}</strong>.
            Seamless wrap-and-blend applied. Inspect tiling before use in production.
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "1rem" }}>
            {textures.map((b64, i) => (
              <div key={i} style={{ border: "1px solid #333", borderRadius: 8, overflow: "hidden", background: "#111" }}>
                <img
                  src={b64}
                  alt={`${category} texture ${i + 1}`}
                  style={{ width: "100%", display: "block", imageRendering: "auto" }}
                />
                <div style={{ padding: "0.5rem", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 12, color: "#888" }}>{category} #{i + 1}</span>
                  <button
                    onClick={() => handleDownload(b64, i)}
                    style={{ fontSize: 11, padding: "0.25rem 0.6rem", borderRadius: 4, border: "1px solid #555", background: "transparent", color: "#aaa", cursor: "pointer" }}
                  >
                    ↓ PNG
                  </button>
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: "1rem", padding: "0.75rem 1rem", background: "#111", border: "1px solid #333", borderRadius: 6, fontSize: 12, color: "#666" }}>
            <strong style={{ color: "#888" }}>Tiling check:</strong> To verify seamlessness, arrange 4 copies in a 2×2 grid and inspect the seam boundaries.
            Textures with visible seams should not be used in production without further processing.
          </div>
        </>
      )}
    </div>
  );
}
