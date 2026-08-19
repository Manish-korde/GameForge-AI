import { Link } from 'react-router-dom';
import './Pages.css';

const Dashboard = () => {
  return (
    <div className="dashboard">
      <div className="hero-section flex flex-col items-center justify-center gap-md">
        <h1>Turn Game Ideas Into Game Assets</h1>
        <p className="subtitle">
          A multi-model Generative AI pipeline for designing,
          generating, reconstructing, varying, and refining 2D game assets.
        </p>
        <div className="flex gap-md mt-4">
          <Link to="/create" className="btn btn-primary">Create New Asset</Link>
          <Link to="/about" className="btn btn-secondary">Explore AI Pipeline</Link>
        </div>
      </div>

      <div className="pipeline-viz-container mt-8">
        <h3 className="mb-4">AI Pipeline</h3>
        <div className="pipeline-viz flex items-center justify-between">
          <div className="pipeline-stage">Prompt</div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage active">Transformer</div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage active">Diffusion</div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">VAE / AE / GAN</div>
          <div className="pipeline-arrow">→</div>
          <div className="pipeline-stage">Asset Library</div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
