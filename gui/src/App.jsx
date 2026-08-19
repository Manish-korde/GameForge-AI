import { Routes, Route } from 'react-router-dom';
import AppShell from './components/AppShell';
import Dashboard from './pages/Dashboard';
import CreateAsset from './pages/CreateAsset';
import AssetLibrary from './pages/AssetLibrary';
import Experiments from './pages/Experiments';
import About from './pages/About';

function App() {
  return (
    <Routes>
      <Route path="/" element={<AppShell />}>
        <Route index element={<Dashboard />} />
        <Route path="create" element={<CreateAsset />} />
        <Route path="library" element={<AssetLibrary />} />
        <Route path="experiments" element={<Experiments />} />
        <Route path="about" element={<About />} />
      </Route>
    </Routes>
  );
}

export default App;
