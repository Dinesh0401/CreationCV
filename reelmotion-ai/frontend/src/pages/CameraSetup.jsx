import React from 'react';
import CameraStudio from '../components/CameraStudio';

export default function CameraSetup() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>Camera Setup & Calibration</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
            Configure webcam, lighting, and framing before recording.
          </p>
        </div>
        <span className="badge badge-warn">Phase 2 Target</span>
      </div>

      <div className="grid-2">
        <CameraStudio />

        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <h4>Capture Devices</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <label className="hw-card-label">Webcam Device</label>
            <select
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border-default)',
                color: 'var(--text-primary)',
                padding: '8px',
                borderRadius: 'var(--radius-md)',
              }}
              disabled
            >
              <option>Default Camera (Browser MediaStream)</option>
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <label className="hw-card-label">Microphone Input</label>
            <select
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border-default)',
                color: 'var(--text-primary)',
                padding: '8px',
                borderRadius: 'var(--radius-md)',
              }}
              disabled
            >
              <option>Default Microphone</option>
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <label className="hw-card-label">Recording Resolution</label>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button className="btn btn-secondary btn-sm" disabled>720p (HD)</button>
              <button className="btn btn-primary btn-sm" disabled>1080p (FHD)</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
