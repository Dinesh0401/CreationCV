import React from 'react';

export default function ExportPanel() {
  return (
    <div className="card" style={{ padding: '24px' }}>
      <div className="card-header">
        <span className="card-title">📦 Video Export Settings</span>
        <span className="badge badge-accent">1080×1920 (9:16)</span>
      </div>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
        Export controls and FFmpeg encoding configurations (Phase 8).
      </p>
    </div>
  );
}
