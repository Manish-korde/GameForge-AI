import { Routes, Route } from 'react-router-dom';
import AppShell from './components/AppShell';
import Dashboard from './pages/Dashboard';
import ConceptPlanner from './pages/ConceptPlanner';
import AEStudio from './pages/AEStudio';
import VAEStudio from './pages/VAEStudio';
import DCGANStudio from './pages/DCGANStudio';
import ActionStudio from './pages/ActionStudio';
import AssetLibrary from './pages/AssetLibrary';
import Experiments from './pages/Experiments';
import About from './pages/About';

function App() {
  return (
    <Routes>
      <Route path="/" element={<AppShell />}>
        {/* Landing Page */}
        <Route index element={<Dashboard />} />
        
        {/* Exactly 5 Primary Generative AI Modules */}
        <Route path="concept" element={<ConceptPlanner />} />
        <Route path="ae" element={<AEStudio />} />
        <Route path="vae" element={<VAEStudio />} />
        <Route path="dcgan" element={<DCGANStudio />} />
        <Route path="diffusion" element={<ActionStudio />} />
        
        {/* Backward-Compatible Route Aliases & Secondary Tools */}
        <Route path="actions" element={<ActionStudio />} />
        <Route path="create" element={<AEStudio />} />
        <Route path="library" element={<AssetLibrary />} />
        <Route path="experiments" element={<Experiments />} />
        <Route path="about" element={<About />} />
      </Route>
    </Routes>
  );
}

export default App;
