import { NavLink, Link } from 'react-router-dom';
import './Sidebar.css';

export default function Sidebar() {
  const modules = [
    {
      name: "Transformer",
      desc: "Concept Planner",
      icon: "🎯",
      path: "/concept"
    },
    {
      name: "AE",
      desc: "Outlier QA & Reconstruction",
      icon: "🔍",
      path: "/ae"
    },
    {
      name: "VAE",
      desc: "Latent Manifold & Similarity",
      icon: "🧬",
      path: "/vae"
    },
    {
      name: "DCGAN",
      desc: "Character Sprite Generator",
      icon: "⚔️",
      path: "/dcgan"
    },
    {
      name: "Diffusion",
      desc: "Multi-Pose Walk Cycle",
      icon: "🏃",
      path: "/diffusion"
    }
  ];

  return (
    <aside className="sidebar" style={{ width: 260, backgroundColor: "#0f172a", borderRight: "1px solid rgba(148, 163, 184, 0.15)", display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Brand Header linking to Dashboard */}
      <Link to="/" style={{ textDecoration: "none", color: "inherit" }}>
        <div style={{
          height: 64,
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "0 1.5rem",
          borderBottom: "1px solid rgba(148, 163, 184, 0.15)",
          background: "rgba(15, 23, 42, 0.6)"
        }}>
          <span style={{ fontSize: 24 }}>🎮</span>
          <div>
            <h2 style={{ fontSize: 16, fontWeight: 800, margin: 0, letterSpacing: "1px", color: "#f8fafc" }}>
              GAMEFORGE AI
            </h2>
            <span style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.5px" }}>
              Generative AI Platform
            </span>
          </div>
        </div>
      </Link>

      {/* Primary 5 Navigation Modules */}
      <nav style={{ padding: "1.25rem 0.75rem", display: "flex", flexDirection: "column", gap: 6, flex: 1, overflowY: "auto" }}>
        <div style={{
          fontSize: 11,
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.8px",
          color: "#64748b",
          padding: "0.25rem 0.75rem 0.5rem"
        }}>
          Neural Modules
        </div>

        {modules.map((mod, index) => (
          <NavLink
            key={mod.name}
            to={mod.path}
            className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
            style={({ isActive }) => ({
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "0.65rem 0.85rem",
              borderRadius: 8,
              textDecoration: "none",
              color: isActive ? "#ffffff" : "#cbd5e1",
              background: isActive ? "rgba(99, 102, 241, 0.2)" : "transparent",
              border: isActive ? "1px solid rgba(99, 102, 241, 0.4)" : "1px solid transparent",
              transition: "all 0.15s ease"
            })}
          >
            <span style={{ fontSize: 18 }}>{mod.icon}</span>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>{mod.name}</span>
                <span style={{ fontSize: 9, color: "#64748b", background: "rgba(148, 163, 184, 0.1)", padding: "1px 4px", borderRadius: 4 }}>
                  M{index + 1}
                </span>
              </div>
              <span style={{ fontSize: 11, color: "#94a3b8" }}>{mod.desc}</span>
            </div>
          </NavLink>
        ))}
      </nav>

      {/* Footer Info */}
      <div style={{
        padding: "1rem 1.25rem",
        borderTop: "1px solid rgba(148, 163, 184, 0.12)",
        background: "rgba(15, 23, 42, 0.4)",
        fontSize: 11,
        color: "#64748b"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#10b981", display: "inline-block" }}></span>
          <span style={{ color: "#94a3b8", fontWeight: 600 }}>5 Models Online</span>
        </div>
        <div>Pipeline Active &amp; Verified</div>
      </div>
    </aside>
  );
}
