import { NavLink } from 'react-router-dom';

const NAV_ITEMS = [
  { path: '/', icon: '📊', label: 'Dashboard' },
  { path: '/characters', icon: '🎭', label: 'Character Studio' },
  { path: '/camera', icon: '📷', label: 'Camera Setup' },
  { path: '/recording', icon: '⏺️', label: 'Recording Studio' },
  { path: '/render', icon: '🔄', label: 'Render Queue' },
  { path: '/exports', icon: '📦', label: 'Export Library' },
  { path: '/settings', icon: '⚙️', label: 'Settings' },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="sidebar-logo-icon">🎬</span>
        <span className="sidebar-logo-text">ReelMotion AI</span>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) =>
              `sidebar-link${isActive ? ' active' : ''}`
            }
          >
            <span className="sidebar-link-icon">{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">v0.1.0</div>
    </aside>
  );
}
