import React from 'react';
import ExportPanel from '../components/ExportPanel';

export default function ExportLibrary() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>Export Library</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
            Browse and download final rendered 1080×1920 MP4 Reels.
          </p>
        </div>
        <span className="badge badge-accent">Phase 8</span>
      </div>

      <ExportPanel />

      <div className="card" style={{ padding: '36px', textAlign: 'center', color: 'var(--text-tertiary)' }}>
        No exported Reels yet. Recorded and rendered videos will appear here.
      </div>
    </div>
  );
}
