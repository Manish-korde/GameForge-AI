import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { checkAEStatus } from '../services/api';
import './TopBar.css';

export default function TopBar() {
  const [sysStatus, setSysStatus] = useState({ online: true, text: 'Neural Services Online', allLoaded: true });

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await checkAEStatus();
        if (typeof res === 'string') {
          setSysStatus({ online: !res.includes('Offline'), text: res, allLoaded: res.includes('Loaded') });
        } else if (res) {
          setSysStatus(res);
        }
      } catch (e) {
        setSysStatus({ online: false, text: 'Backend Initializing...', allLoaded: false });
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const badgeColor = !sysStatus.online ? '#ef4444' : sysStatus.allLoaded ? '#10b981' : '#f59e0b';
  const badgeLabel = !sysStatus.online ? 'Backend Offline' : sysStatus.allLoaded ? 'All 5 Neural Engines Active' : 'Initializing Engines...';

  return (
    <header className="topbar" style={{
      height: 64,
      background: "rgba(15, 23, 42, 0.8)",
      backdropFilter: "blur(12px)",
      borderBottom: "1px solid rgba(148, 163, 184, 0.12)",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "0 1.5rem"
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <Link to="/" style={{ textDecoration: "none", color: "inherit", display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 20 }}>⚔️</span>
          <strong style={{ fontSize: 16, letterSpacing: "1px", color: "#f8fafc", fontWeight: 800 }}>GAMEFORGE AI</strong>
        </Link>
      </div>

      <div style={{ fontSize: 13, color: "#94a3b8", display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ color: "#6366f1" }}>◆</span>
        <span>AI Game Asset Synthesis &amp; Latent Exploration Engine</span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{
          background: "rgba(15, 23, 42, 0.6)",
          border: `1px solid ${badgeColor}44`,
          color: badgeColor,
          padding: "0.3rem 0.8rem",
          borderRadius: 20,
          fontSize: 12,
          fontWeight: 600,
          display: "flex",
          alignItems: "center",
          gap: 6
        }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: badgeColor, display: "inline-block" }}></span>
          {badgeLabel}
        </div>
      </div>
    </header>
  );
}
