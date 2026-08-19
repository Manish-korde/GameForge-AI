const About = () => {
  return (
    <div className="about-page max-w-3xl">
      <h2>GameForge AI</h2>
      <p className="text-lg text-secondary mb-8">
        A multi-model Generative AI pipeline for 2D game asset generation and processing.
      </p>

      <div className="card mb-6">
        <h3>Models</h3>
        <div className="mt-4 flex flex-col gap-md">
          <div>
            <strong>Transformer</strong>
            <p className="text-sm text-secondary">Game understanding and structured generation.</p>
          </div>
          <div>
            <strong>Autoencoder</strong>
            <p className="text-sm text-secondary">Reconstruction and denoising.</p>
          </div>
          <div>
            <strong>VAE</strong>
            <p className="text-sm text-secondary">Latent-space variation.</p>
          </div>
          <div>
            <strong>GAN</strong>
            <p className="text-sm text-secondary">Texture/material generation and refinement.</p>
          </div>
          <div>
            <strong>Diffusion</strong>
            <p className="text-sm text-secondary">Visual generation.</p>
          </div>
        </div>
      </div>

      <div className="card mb-6">
        <h3>Dataset & Responsible AI</h3>
        <table className="w-full text-sm text-left mt-4 border-collapse">
          <thead>
            <tr className="border-b border-border text-secondary">
              <th className="py-2">Dataset</th>
              <th className="py-2">License</th>
              <th className="py-2">Purpose</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-border">
              <td className="py-2">evilsocket/alucard-sprites (282k+ images)</td>
              <td className="py-2">Research / Non-Commercial</td>
              <td className="py-2">Autoencoder training</td>
            </tr>
            <tr className="border-b border-border">
              <td className="py-2">OpenGameArt-CC0 (Curated Subset)</td>
              <td className="py-2">CC0</td>
              <td className="py-2">Sprites & Props generation</td>
            </tr>
            <tr>
              <td className="py-2">VastTextures / PolyHaven</td>
              <td className="py-2">CC0 / Open</td>
              <td className="py-2">GAN Textures</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default About;
