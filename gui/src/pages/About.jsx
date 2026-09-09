import { useState, useEffect } from 'react';
import { fetchEthicalGuidelines } from '../services/api';

const About = () => {
  const [ethicalData, setEthicalData] = useState(null);

  useEffect(() => {
    fetchEthicalGuidelines().then(setEthicalData);
  }, []);

  return (
    <div className="about-page max-w-4xl">
      <div className="page-header mb-6">
        <h2>Responsible AI & Ethical Deployment Hub 🛡️</h2>
        <p className="text-lg text-secondary">
          GameForge AI is built on strict Responsible AI principles: content safety boundary filtering, creator attribution, dataset licensing compliance, and human-in-the-loop oversight.
        </p>
      </div>

      {/* 4 Pillars of Ethical AI */}
      <div className="grid grid-cols-2 gap-md mb-6">
        <div className="card bg-surface-elevated">
          <h4 className="text-indigo-400 font-bold text-base mb-2">1. Prompt Safety & Boundary Filtering 🚫</h4>
          <p className="text-xs text-secondary leading-relaxed">
            All user prompts undergo real-time boundary validation in our backend to block harmful, explicit, or trademark-infringing content before running Transformer concept generation.
          </p>
        </div>

        <div className="card bg-surface-elevated">
          <h4 className="text-green-400 font-bold text-base mb-2">2. Creator Attribution & Transparency 📜</h4>
          <p className="text-xs text-secondary leading-relaxed">
            Every dataset source is fully credited. Retrieved visual assets and synthesized specifications attach explicit creator attribution metadata and license tags (`CC-BY-NC-SA 4.0`).
          </p>
        </div>

        <div className="card bg-surface-elevated">
          <h4 className="text-amber-400 font-bold text-base mb-2">3. Human-in-the-Loop Co-Pilot 🤝</h4>
          <p className="text-xs text-secondary leading-relaxed">
            AI is positioned strictly as an assistive workflow co-pilot for indie game developers. Human designers retain 100% control to approve, modify, or reject AI outputs.
          </p>
        </div>

        <div className="card bg-surface-elevated">
          <h4 className="text-blue-400 font-bold text-base mb-2">4. Non-Commercial Research Use ⚖️</h4>
          <p className="text-xs text-secondary leading-relaxed">
            Model training and benchmarking strictly honor Creative Commons research boundaries, providing controlled scientific evaluation without infringing on commercial creators.
          </p>
        </div>
      </div>

      {/* Dataset Licensing & Attribution Table */}
      <div className="card mb-6">
        <h3 className="mb-4">Dataset Attribution & Licensing Manifest</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left border-collapse">
            <thead>
              <tr className="border-b border-border text-xs uppercase text-secondary">
                <th className="py-3 px-4">Dataset Source</th>
                <th className="py-3 px-4">Population Size</th>
                <th className="py-3 px-4">License Type</th>
                <th className="py-3 px-4">Pipeline Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border text-xs">
              {ethicalData?.datasets ? (
                ethicalData.datasets.map((d, i) => (
                  <tr key={i}>
                    <td className="py-3 px-4 font-semibold text-white">{d.name}</td>
                    <td className="py-3 px-4 font-mono text-indigo-400">{d.count.toLocaleString()} items</td>
                    <td className="py-3 px-4"><span className="badge badge-genre">{d.license}</span></td>
                    <td className="py-3 px-4 text-secondary">{d.role}</td>
                  </tr>
                ))
              ) : (
                <>
                  <tr className="border-b border-border">
                    <td className="py-3 px-4 font-semibold text-white">evilsocket/alucard-sprites</td>
                    <td className="py-3 px-4 font-mono text-indigo-400">282,511 RGBA Sprites</td>
                    <td className="py-3 px-4"><span className="badge badge-genre">CC-BY-NC-SA 4.0</span></td>
                    <td className="py-3 px-4 text-secondary">32-bit Sprite VAE/AE & Vector Search</td>
                  </tr>
                  <tr className="border-b border-border">
                    <td className="py-3 px-4 font-semibold text-white">GEM/viggo</td>
                    <td className="py-3 px-4 font-mono text-indigo-400">6,900 Samples</td>
                    <td className="py-3 px-4"><span className="badge badge-genre">CC-BY 4.0</span></td>
                    <td className="py-3 px-4 text-secondary">Dialogue & Intent NLP Transformer</td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-semibold text-white">Fraser/pico-8-games</td>
                    <td className="py-3 px-4 font-mono text-indigo-400">10,967 Cartridges</td>
                    <td className="py-3 px-4"><span className="badge badge-genre">CC-BY-NC-SA 4.0</span></td>
                    <td className="py-3 px-4 text-secondary">Retro 8x8 Tilemaps & Cart Mechanics</td>
                  </tr>
                </>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Family Overview */}
      <div className="card">
        <h3 className="mb-4">Model Family Roles & Responsible Use</h3>
        <div className="flex flex-col gap-md text-xs text-secondary">
          <div className="p-3 bg-surface-elevated rounded border border-border">
            <strong className="text-white text-sm">1. Transformer (Flan-T5-Small)</strong>
            <p className="mt-1">Translates natural language text into structured game spec JSON. Enforces safety boundary checks to reject prohibited or infringing requests.</p>
          </div>
          <div className="p-3 bg-surface-elevated rounded border border-border">
            <strong className="text-white text-sm">2. Autoencoder (AE 280K)</strong>
            <p className="mt-1">High-precision deterministic reconstruction baseline (`Median MSE: 0.000782`). Surfacing the highest-error 5% of assets exceeding the P95 MSE threshold (`0.001313`) for automated QA review.</p>
          </div>
          <div className="p-3 bg-surface-elevated rounded border border-border">
            <strong className="text-white text-sm">3. Variational Autoencoder (VAE 280K)</strong>
            <p className="mt-1">Continuous Gaussian latent space representation. Powers sub-millisecond similarity search (&lt; 1ms across 25,000 vectors) and smooth real-sprite morphing without replacing human artists.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
