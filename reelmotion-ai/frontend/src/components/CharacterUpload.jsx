import React from 'react';

export default function CharacterUpload() {
  return (
    <div
      className="card"
      style={{
        border: '2px dashed var(--border-strong)',
        padding: '36px',
        textAlign: 'center',
        cursor: 'pointer',
      }}
    >
      <div style={{ fontSize: '2.5rem', marginBottom: '8px' }}>🎭</div>
      <h4>Drag & Drop Character References</h4>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '6px' }}>
        Accepts PNG, JPG, WebP. Multi-image character validation in Phase 4.
      </p>
    </div>
  );
}
