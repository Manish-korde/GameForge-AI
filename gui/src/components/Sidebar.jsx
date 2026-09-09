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
        
        <div className="nav-section">Create & Explore</div>
        <NavLink to="/concept" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>🎯 Game Concept (T5)</NavLink>
        <NavLink to="/create" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>🎨 Asset Studio & VAE</NavLink>

        <div className="nav-section">AI Pipeline</div>
        <NavLink to="/concept" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>⚡ Transformer Planner</NavLink>
        <NavLink to="/create" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>🧬 VAE Latent Engine</NavLink>
        <NavLink to="/create" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>🔍 Autoencoder Outliers</NavLink>


        <div className="nav-separator"></div>

        <NavLink to="/library" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>Asset Library</NavLink>
        <NavLink to="/experiments" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>Experiments</NavLink>
        <NavLink to="/about" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>About Project</NavLink>
      </nav>
    </aside>
  );
};

export default Sidebar;
