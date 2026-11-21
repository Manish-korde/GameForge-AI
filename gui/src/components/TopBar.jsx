import { useState, useEffect } from 'react';
import { checkAEStatus } from '../services/api';
import './TopBar.css';

const TopBar = () => {
  const [aeStatus, setAeStatus] = useState('Checking...');

  useEffect(() => {
    const fetchStatus = async () => {
      const status = await checkAEStatus();
      setAeStatus(status);
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="topbar flex items-center justify-between">
      <div className="topbar-left">
        <strong>GameForge AI</strong>
      </div>
      <div className="topbar-center">
        Current Project: Untitled Game
      </div>
      <div className="topbar-right flex items-center gap-md">
        <div className="status-indicator" style={{ fontSize: '0.85rem', color: (aeStatus.includes('Autoencoder: Loaded') && aeStatus.includes('VAE: Loaded')) ? '#4ade80' : (aeStatus.includes('Loading') || aeStatus.includes('Checking')) ? '#f59e0b' : '#f87171' }}>
          {aeStatus}
        </div>
        <div className="status-indicator">
          <span className="dot live" style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981', display: 'inline-block', marginRight: 6 }}></span>
          Live Neural Models
        </div>
      </div>
    </header>
  );
};

export default TopBar;
