import { NavLink } from 'react-router-dom';
import './Sidebar.css';

const Sidebar = () => {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>GameForge AI</h2>
      </div>
      <nav className="sidebar-nav">
        <NavLink to="/" end className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>Dashboard</NavLink>
        
        <div className="nav-section">Create</div>
        <NavLink to="/create" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>Game Concept</NavLink>
        <div className="nav-link disabled">Asset Generator</div>
        <div className="nav-link disabled">NPC / Lore</div>

        <div className="nav-section">AI Pipeline</div>
        <div className="nav-link disabled">Transformer</div>
        <div className="nav-link disabled">Diffusion</div>
        <div className="nav-link disabled">VAE</div>
        <div className="nav-link disabled">Autoencoder</div>
        <div className="nav-link disabled">GAN</div>

        <div className="nav-separator"></div>

        <NavLink to="/library" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>Asset Library</NavLink>
        <NavLink to="/experiments" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>Experiments</NavLink>
        <NavLink to="/about" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>About Project</NavLink>
      </nav>
    </aside>
  );
};

export default Sidebar;
