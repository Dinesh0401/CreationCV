import React from 'react';
import { useNavigate } from 'react-router-dom';

/**
 * RenderQueue Component
 * Renders list of rendering jobs with progress bar (0-100%), status pill,
 * elapsed time / timestamps, and cancel button.
 */
export default function RenderQueue({
  jobs = [],
  onCancelJob = () => {},
  onRetryJob = () => {},
  loading = false,
}) {
  const navigate = useNavigate();

  // Helper to format ISO date string
  const formatTimestamp = (isoStr) => {
    if (!isoStr) return 'N/A';
    try {
      const d = new Date(isoStr);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return isoStr;
    }
  };

  // Helper to format elapsed time in seconds
  const getElapsedSeconds = (startedAt, completedAt) => {
    if (!startedAt) return null;
    const start = new Date(startedAt).getTime();
    const end = completedAt ? new Date(completedAt).getTime() : Date.now();
    const sec = Math.max(0, Math.floor((end - start) / 1000));
    const mins = Math.floor(sec / 60);
    const remainingSec = sec % 60;
    return `${mins}m ${remainingSec}s`;
  };

  if (jobs.length === 0 && !loading) {
    return (
      <div className="card" style={{ padding: '48px 24px', textAlign: 'center' }}>
        <div style={{ fontSize: '2.5rem', marginBottom: '12px', opacity: 0.5 }}>⚡</div>
        <h4 style={{ color: 'var(--text-secondary)', marginBottom: '8px' }}>Render Queue is Clear</h4>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', maxWidth: '380px', margin: '0 auto 16px' }}>
          No active or pending rendering jobs. Record a video in the Recording Studio to generate a character replacement reel.
        </p>
        <button className="btn btn-primary" onClick={() => navigate('/recording')}>
          Go to Recording Studio →
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {jobs.map((job) => {
        const isRunning = job.state === 'running';
        const isQueued = job.state === 'queued';
        const isCompleted = job.state === 'completed';
        const isFailed = job.state === 'failed';
        const isCancelled = job.state === 'cancelled';

        const canCancel = isQueued || isRunning;
        const progress = Math.min(100, Math.max(0, Math.round(job.progress || (isCompleted ? 100 : 0))));
        const elapsed = getElapsedSeconds(job.started_at, job.completed_at);
        const characterName = job.metadata?.character_name || job.metadata?.character_id || 'AI Character';

        return (
          <div
            key={job.id}
            className="card"
            style={{
              padding: '18px 20px',
              borderLeft: isRunning
                ? '4px solid var(--accent)'
                : isCompleted
                ? '4px solid var(--status-ok)'
                : isFailed
                ? '4px solid var(--status-error)'
                : isQueued
                ? '4px solid var(--status-warn)'
                : '4px solid var(--border-default)',
            }}
          >
            {/* Top Row: Job Header & Status Pill */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ fontSize: '1.2rem' }}>
                  {isRunning ? '⚙️' : isCompleted ? '✅' : isFailed ? '❌' : '⏳'}
                </span>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.92rem', color: 'var(--text-primary)' }}>
                    Job #{job.id.slice(0, 8)} — {characterName}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                    Type: {job.type?.toUpperCase() || 'RENDER'} • Target: 1080×1920 (9:16)
                  </div>
                </div>
              </div>

              {/* Status Pill Badge */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span
                  className={`badge ${
                    isCompleted
                      ? 'badge-ok'
                      : isRunning
                      ? 'badge-accent'
                      : isQueued
                      ? 'badge-warn'
                      : isFailed
                      ? 'badge-error'
                      : 'badge-info'
                  }`}
                  style={{
                    boxShadow: isRunning ? '0 0 10px rgba(59, 130, 246, 0.4)' : 'none',
                    animation: isRunning ? 'pulse-dot 1.5s infinite' : 'none',
                  }}
                >
                  {job.state.toUpperCase()}
                </span>
              </div>
            </div>

            {/* Middle Row: Progress Bar (0 - 100%) */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                <span>
                  {job.message ||
                    (isRunning
                      ? 'Processing pose tracking & diffusion synthesis...'
                      : isQueued
                      ? 'Waiting in queue for GPU worker allocation...'
                      : isCompleted
                      ? 'Render pipeline complete. 1080x1920 MP4 exported.'
                      : job.error || 'Job ended')}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{progress}%</span>
              </div>

              <div
                style={{
                  height: '8px',
                  background: 'var(--bg-surface)',
                  borderRadius: '4px',
                  overflow: 'hidden',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: `${progress}%`,
                    background: isFailed
                      ? 'var(--status-error)'
                      : isCompleted
                      ? 'var(--status-ok)'
                      : 'linear-gradient(90deg, var(--accent), var(--highlight))',
                    borderRadius: '4px',
                    transition: 'width 300ms ease',
                  }}
                />
              </div>
            </div>

            {/* Bottom Row: Metadata & Actions */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '12px',
                fontSize: '0.78rem',
                color: 'var(--text-tertiary)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontFamily: 'var(--font-mono)' }}>
                <span>Created: {formatTimestamp(job.created_at)}</span>
                {elapsed && <span>Elapsed: {elapsed}</span>}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {canCancel && (
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => onCancelJob(job.id)}
                    title="Cancel this render job"
                  >
                    ⏹️ Cancel Job
                  </button>
                )}

                {isCompleted && (
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={() => navigate('/exports')}
                    title="View completed Reel in library"
                  >
                    🎬 View in Export Library →
                  </button>
                )}

                {(isFailed || isCancelled) && (
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => onRetryJob(job)}
                    title="Retry this render job"
                  >
                    🔄 Retry
                  </button>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
