import { Link } from 'react-router-dom';
import { FRONT_FACING_SPRITES_25 } from '../services/api';

export default function Dashboard() {
  const showcaseCharacters = FRONT_FACING_SPRITES_25.slice(0, 10);

  const modules = [
    {
      id: "transformer",
      name: "Transformer",
      subtitle: "Character Concept Planner",
      icon: "🎯",
      path: "/concept",
      badge: "Seq2Seq Flan-T5",
      badgeColor: "#6366f1",
      description: "Fine-tuned on 2,000 prompt-JSON concept pairs to synthesize structured game specifications, character backstories, environments, and equipment archetypes.",
      metric: "99.8% Schema Compliance",
    },
    {
      id: "ae",
      name: "AE",
      subtitle: "Outlier Screening & Reconstruction",
      icon: "🔍",
      path: "/ae",
      badge: "Autoencoder 280K",
      badgeColor: "#10b981",
      description: "Deterministic high-fidelity pixel-art reconstruction with automated P95 outlier screening, MSE/PSNR/SSIM evaluation, and silhouette validation.",
      metric: "P95 Threshold ≤ 0.001313",
    },
    {
      id: "vae",
      name: "VAE",
      subtitle: "Latent Manifold & Similarity",
      icon: "🧬",
      path: "/vae",
      badge: "256-D Continuous Manifold",
      badgeColor: "#0ea5e9",
      description: "Continuous 256-dimensional latent space exploration, continuous spherical interpolation (SLERP), and real-time cosine similarity retrieval.",
      metric: "282,511 Indexed Sprites",
    },
    {
      id: "dcgan",
      name: "DCGAN",
      subtitle: "Procedural Character Synthesis",
      icon: "⚔️",
      path: "/dcgan",
      badge: "Discrete 50-Epoch + cDCGAN v3",
      badgeColor: "#f59e0b",
      description: "Trained on 42,212 front-facing characters across 50 epochs for procedural generation, paired with cDCGAN v3 palette conditioning and Quality Filter v3.",
      metric: "36-Sprite 6×6 Canvas",
    },
    {
      id: "diffusion",
      name: "Diffusion",
      subtitle: "Multi-Pose Walk Cycle Studio",
      icon: "🏃",
      path: "/diffusion",
      badge: "50-Epoch DDIM Reverse",
      badgeColor: "#ec4899",
      description: "Synthesizes kinematic multi-pose sequences and complete 8-frame walk cycle animations from single front-facing character sprites via reverse diffusion.",
      metric: "8-Frame Loop Continuity",
    },
  ];

  return (
    <div style={{ padding: "2rem", maxWidth: 1200, margin: "0 auto", color: "#f1f5f9" }}>
      
      {/* Hero Section */}
      <div style={{
        background: "linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: 16,
        padding: "2.5rem 2rem",
        marginBottom: "2.5rem",
        textAlign: "center",
        boxShadow: "0 20px 40px rgba(0,0,0,0.4)"
      }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: "rgba(99, 102, 241, 0.15)", border: "1px solid rgba(99, 102, 241, 0.3)", padding: "4px 12px", borderRadius: 20, marginBottom: "1rem" }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#6366f1" }}></span>
          <span style={{ fontSize: 12, fontWeight: 700, color: "#a5b4fc", letterSpacing: "1px", textTransform: "uppercase" }}>
            Neural Pixel Art Studio &amp; Latent Platform
          </span>
        </div>

        <h1 style={{
          fontSize: 42,
          fontWeight: 800,
          letterSpacing: "-1px",
          margin: "0 0 0.75rem 0",
          background: "linear-gradient(135deg, #ffffff 30%, #94a3b8 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent"
        }}>
          GAMEFORGE AI
        </h1>

        <p style={{
          fontSize: 18,
          color: "#cbd5e1",
          fontWeight: 500,
          margin: "0 0 1rem 0"
        }}>
          Create, Explore &amp; Generate Game-Ready Pixel Art with Generative AI
        </p>

        <p style={{
          fontSize: 14,
          color: "#94a3b8",
          maxWidth: 720,
          margin: "0 auto 1.75rem auto",
          lineHeight: 1.6
        }}>
          An end-to-end generative pipeline uniting Seq2Seq Transformers for semantic game specification, Autoencoders for quality assurance screening, Variational Autoencoders for continuous latent retrieval, Discrete DCGAN for procedural character synthesis, and Diffusion models for kinematic walk-cycle animation.
        </p>

        <div style={{ display: "flex", justifyContent: "center", gap: "1rem", flexWrap: "wrap" }}>
          <Link
            to="/concept"
            style={{
              background: "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
              color: "#ffffff",
              padding: "0.75rem 1.5rem",
              borderRadius: 8,
              fontSize: 14,
              fontWeight: 600,
              textDecoration: "none",
              boxShadow: "0 4px 14px rgba(99, 102, 241, 0.4)",
              transition: "transform 0.15s ease"
            }}
          >
            Start Concept Planner 🎯
          </Link>
          <Link
            to="/dcgan"
            style={{
              background: "rgba(30, 41, 59, 0.8)",
              border: "1px solid rgba(148, 163, 184, 0.2)",
              color: "#f8fafc",
              padding: "0.75rem 1.5rem",
              borderRadius: 8,
              fontSize: 14,
              fontWeight: 600,
              textDecoration: "none",
              transition: "all 0.15s ease"
            }}
          >
            Explore DCGAN Studio ⚔️
          </Link>
        </div>
      </div>

      {/* Front-Facing Character Showcase Strip */}
      <div style={{
        background: "rgba(30, 41, 59, 0.5)",
        border: "1px solid rgba(148, 163, 184, 0.15)",
        borderRadius: 14,
        padding: "1.5rem",
        marginBottom: "2.5rem"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem", flexWrap: "wrap", gap: "0.5rem" }}>
          <div>
            <h3 style={{ margin: "0 0 2px 0", fontSize: 16, fontWeight: 700, color: "#f8fafc" }}>
              Front-Facing Character Asset Showcase
            </h3>
            <p style={{ margin: 0, fontSize: 12, color: "#94a3b8" }}>
              Sampled from our verified front-facing character dataset (64×64 RGBA crisp nearest-neighbor pixel art).
            </p>
          </div>
          <span style={{ fontSize: 11, color: "#60a5fa", background: "rgba(59, 130, 246, 0.15)", padding: "3px 8px", borderRadius: 6, fontWeight: 600 }}>
            LPC Front-Facing Dataset
          </span>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(95px, 1fr))",
          gap: "0.75rem"
        }}>
          {showcaseCharacters.map((char) => (
            <div
              key={char.id}
              style={{
                background: "conic-gradient(#1e293b 90deg, #0f172a 90deg 180deg, #1e293b 180deg 270deg, #0f172a 270deg) 0 0/14px 14px",
                border: "1px solid rgba(148, 163, 184, 0.2)",
                borderRadius: 8,
                padding: "0.6rem",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                transition: "transform 0.15s ease, border-color 0.15s ease"
              }}
            >
              <img
                src={char.url}
                alt={char.name}
                style={{
                  width: 56,
                  height: 56,
                  objectFit: "contain",
                  imageRendering: "pixelated",
                  filter: "drop-shadow(0 4px 8px rgba(0,0,0,0.5))"
                }}
              />
              <span style={{ fontSize: 11, fontWeight: 600, color: "#f8fafc", marginTop: 6, textAlign: "center", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", width: "100%" }}>
                {char.name}
              </span>
              <span style={{ fontSize: 9, color: "#94a3b8" }}>
                {char.category}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Module Overview: 5 Primary Neural Architectures */}
      <div>
        <div style={{ marginBottom: "1.25rem" }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 4px 0", color: "#f8fafc" }}>
            Five Specialized Generative AI Modules
          </h2>
          <p style={{ margin: 0, fontSize: 13, color: "#94a3b8" }}>
            Select any module to inspect neural inference, latent exploration manifolds, and quantitative benchmarks.
          </p>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: "1.25rem"
        }}>
          {modules.map((mod) => (
            <Link
              key={mod.id}
              to={mod.path}
              style={{
                background: "rgba(30, 41, 59, 0.6)",
                border: "1px solid rgba(148, 163, 184, 0.15)",
                borderRadius: 12,
                padding: "1.5rem",
                textDecoration: "none",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                transition: "all 0.2s ease",
                position: "relative",
                overflow: "hidden"
              }}
            >
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 28 }}>{mod.icon}</span>
                    <div>
                      <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#f8fafc" }}>{mod.name}</h3>
                      <span style={{ fontSize: 12, color: "#94a3b8" }}>{mod.subtitle}</span>
                    </div>
                  </div>
                  <span style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: "2px 8px",
                    borderRadius: 10,
                    background: `${mod.badgeColor}22`,
                    color: mod.badgeColor,
                    border: `1px solid ${mod.badgeColor}44`,
                    textTransform: "uppercase"
                  }}>
                    {mod.badge}
                  </span>
                </div>

                <p style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.5, margin: "0 0 1rem 0" }}>
                  {mod.description}
                </p>
              </div>

              <div style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                paddingTop: "0.85rem",
                borderTop: "1px solid rgba(148, 163, 184, 0.1)",
                fontSize: 12
              }}>
                <span style={{ color: "#94a3b8" }}>
                  Target: <strong style={{ color: mod.badgeColor }}>{mod.metric}</strong>
                </span>
                <span style={{ color: "#818cf8", fontWeight: 600 }}>Launch Module →</span>
              </div>
            </Link>
          ))}
        </div>
      </div>

    </div>
  );
}
