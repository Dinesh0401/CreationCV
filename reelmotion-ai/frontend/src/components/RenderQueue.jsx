import React from 'react';

export default function RenderQueue({ jobs = [] }) {
  if (jobs.length === 0) {
    return (
      <div className="card" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        <p>No active jobs in the render queue.</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {jobs.map((job) => (
        <div key={job.id} className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <strong>Job #{job.id.slice(0, 8)}</strong>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)' }}>{job.type} • {job.state}</div>
          </div>
          <span className={`badge badge-${job.state === 'completed' ? 'ok' : 'info'}`}>
            {job.state}
          </span>
        </div>
      ))}
    </div>
  );
}
