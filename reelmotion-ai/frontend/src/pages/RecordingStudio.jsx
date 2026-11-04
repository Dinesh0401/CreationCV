import React from 'react';
import CameraStudio from '../components/CameraStudio';
import RecordingPanel from '../components/RecordingPanel';

export default function RecordingStudio() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>Live Recording Studio</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
            Record Reels with synchronized audio, landmarks extraction, and background preservation.
          </p>
        </div>
        <span className="badge badge-accent">Vertical 9:16</span>
      </div>

      <div className="grid-2">
        <CameraStudio />
        <RecordingPanel />
      </div>
    </div>
  );
}

