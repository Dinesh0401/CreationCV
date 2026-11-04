import React from 'react';

export default function CameraStudio() {
  return (
    <div className="card" style={{ padding: '24px', textAlign: 'center' }}>
      <div style={{ fontSize: '2rem', marginBottom: '12px' }}>📷</div>
      <h3>Live Camera View</h3>
      <p style={{ color: 'var(--text-secondary)', marginTop: '8px' }}>
        Browser MediaStream camera preview will be enabled in Phase 2.
      </p>
    </div>
  );
}

