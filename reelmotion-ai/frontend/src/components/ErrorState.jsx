import React from 'react';

export default function ErrorState({
  title = 'Something went wrong',
  message,
  retryAction,
  icon = '⚠️',
}) {
  return (
    <div
      className="card"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '36px',
        textAlign: 'center',
        gap: '12px',
        maxWidth: '560px',
        margin: '24px auto',
        border: '1px solid var(--status-error-muted)',
      }}
    >
      <div style={{ fontSize: '2.5rem' }}>{icon}</div>
      <h3 style={{ color: 'var(--text-primary)', fontSize: '1.1rem', fontWeight: 600 }}>{title}</h3>
      {message && (
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.5 }}>
          {message}
        </p>
      )}
      {retryAction && (
        <button
          className="btn btn-secondary btn-sm"
          onClick={retryAction}
          style={{ marginTop: '8px' }}
        >
          🔄 Try Again
        </button>
      )}
    </div>
  );
}

