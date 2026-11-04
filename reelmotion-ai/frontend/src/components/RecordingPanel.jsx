import React from 'react';

export default function RecordingPanel() {
  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div className="card-header">
        <span className="card-title">⏺️ Recording Controls</span>
        <span className="badge badge-info">Phase 2</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
        <button className="btn btn-danger btn-lg" disabled>
          🔴 Start Recording
        </button>
      </div>
      <div style={{ textAlign: 'center', color: 'var(--text-tertiary)', fontSize: '0.8rem' }}>
        MediaStream recording with audio sync will be activated in Phase 2.
      </div>
    </div>
  );
}

