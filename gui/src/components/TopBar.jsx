import { useState, useEffect } from 'react';
import { checkAEStatus } from '../services/api';
import './TopBar.css';

const TopBar = () => {
  const [sysStatus, setSysStatus] = useState({ online: false, text: 'Connecting...', allLoaded: false });

  useEffect(() => {
    const fetchStatus = async () => {
      const res = await checkAEStatus();
      if (typeof res === 'string') {
        setSysStatus({ online: !res.includes('Offline'), text: res, allLoaded: res.includes('Loaded') });
      } else {
        setSysStatus(res);
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const badgeColor = !sysStatus.online ? '#ef4444' : sysStatus.allLoaded ? '#10b981' : '#f59e0b';
  const badgeLabel = !sysStatus.online ? 'Backend Offline' : sysStatus.allLoaded ? 'Live Neural Models Loaded' : 'Neural Models Initializing...';

  return (
    <header className="topbar flex items-center justify-between">
      <div className="topbar-left">
        <strong>GameForge AI</strong>
      </div>
      <div className="topbar-center">
        Current Project: Untitled Game
      </div>
      <div className="topbar-right flex items-center gap-md">
        <div className="status-indicator" style={{ fontSize: '0.85rem', color: badgeColor }}>
          {sysStatus.text}
        </div>
        <div className="status-indicator" style={{ color: badgeColor, display: 'flex', alignItems: 'center' }}>
          <span className="dot" style={{ width: 8, height: 8, borderRadius: '50%', background: badgeColor, display: 'inline-block', marginRight: 6 }}></span>
          {badgeLabel}
        </div>
      </div>
    </header>
  );
};

export default TopBar;
