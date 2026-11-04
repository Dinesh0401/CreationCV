import React, { useEffect, useState } from 'react';
import useApi from '../hooks/useApi';

export default function Settings() {
  const { request } = useApi();
  const [health, setHealth] = useState(null);
  const [hw, setHw] = useState(null);

  useEffect(() => {
    request('/api/health').then((res) => setHealth(res.data));
    request('/api/hardware').then((res) => setHw(res.data));
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2>System Settings & Diagnostics</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
          ReelMotion AI configuration, directory paths, and local runtime profile.
        </p>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <span className="card-title">⚙️ Core Architecture</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div className="hw-card-row">
              <span className="hw-card-label">Execution Model</span>
              <span className="hw-card-val">Local-Only (Zero Cloud/API)</span>
            </div>
            <div className="hw-card-row">
              <span className="hw-card-label">Backend Host</span>
              <span className="hw-card-val">127.0.0.1:8000</span>
            </div>
            <div className="hw-card-row">
              <span className="hw-card-label">Frontend Host</span>
              <span className="hw-card-val">localhost:5173</span>
            </div>
            <div className="hw-card-row">
              <span className="hw-card-label">Target Aspect Ratio</span>
              <span className="hw-card-val">9:16 (1080×1920)</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">📁 Storage Directories</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div className="hw-card-row">
              <span className="hw-card-label">Models</span>
              <span className="hw-card-val">./models</span>
            </div>
            <div className="hw-card-row">
              <span className="hw-card-label">Characters</span>
              <span className="hw-card-val">./characters</span>
            </div>
            <div className="hw-card-row">
              <span className="hw-card-label">Recordings</span>
              <span className="hw-card-val">./recordings</span>
            </div>
            <div className="hw-card-row">
              <span className="hw-card-label">Exports</span>
              <span className="hw-card-val">./exports</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

