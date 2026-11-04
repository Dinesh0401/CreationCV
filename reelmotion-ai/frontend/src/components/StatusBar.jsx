import { useLocation } from 'react-router-dom';

const PAGE_TITLES = {
  '/': 'Dashboard',
  '/characters': 'Character Studio',
  '/camera': 'Camera Setup',
  '/recording': 'Recording Studio',
  '/render': 'Render Queue',
  '/exports': 'Export Library',
  '/settings': 'Settings',
};

export default function StatusBar({ wsState, hardwareStats }) {
  const location = useLocation();
  const pageTitle = PAGE_TITLES[location.pathname] || 'ReelMotion AI';

  const isConnected = wsState === 'connected';
  const isConnecting = wsState === 'connecting';

  // Extract GPU VRAM from hardware stats if available
  const gpu = hardwareStats?.gpu;
  const vramUsed = gpu?.vram_used_gb;
  const vramTotal = gpu?.vram_total_gb;
  const vramPercent =
    vramUsed != null && vramTotal != null && vramTotal > 0
      ? Math.round((vramUsed / vramTotal) * 100)
      : null;

  const getBarColor = (pct) => {
    if (pct < 60) return 'var(--status-ok)';
    if (pct < 85) return 'var(--status-warn)';
    return 'var(--status-error)';
  };

  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="topbar-title">{pageTitle}</span>
      </div>

      <div className="topbar-center">ReelMotion AI Studio</div>

      <div className="topbar-right">
        {vramPercent != null && (
          <div className="vram-mini">
            <span>VRAM</span>
            <div className="vram-mini-bar">
              <div
                className="vram-mini-fill"
                style={{
                  width: `${vramPercent}%`,
                  background: getBarColor(vramPercent),
                }}
              />
            </div>
            <span>{vramPercent}%</span>
          </div>
        )}

        <div className="connection-indicator">
          <span
            className={`connection-dot ${
              isConnected
                ? 'connected'
                : isConnecting
                ? 'connecting'
                : 'disconnected'
            }`}
          />
          <span>
            {isConnected
              ? 'Connected'
              : isConnecting
              ? 'Connecting…'
              : 'Disconnected'}
          </span>
        </div>
      </div>
    </header>
  );
}
