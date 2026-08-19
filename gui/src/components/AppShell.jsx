import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import './AppShell.css';

const AppShell = () => {
  return (
    <div className="app-layout">
      <Sidebar />
      <div className="main-container">
        <TopBar />
        <main className="content-area">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default AppShell;
