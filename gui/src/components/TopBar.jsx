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
        <div className="status-indicator" style={{ fontSize: '0.85rem', color: aeStatus.includes('Loaded') ? '#4ade80' : '#f87171' }}>
          {aeStatus}
        </div>
        <div className="status-indicator">
          <span className="dot demo"></span> Demo Mode
        </div>
      </div>
    </header>
  );
};

export default TopBar;
