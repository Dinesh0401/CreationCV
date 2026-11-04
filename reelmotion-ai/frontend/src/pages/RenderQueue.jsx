import React, { useEffect, useState, useCallback, useRef } from 'react';
import RenderQueueComponent from '../components/RenderQueue';
import useApi from '../hooks/useApi';

/**
 * RenderQueue Page
 * Displays active and past rendering jobs with real-time progress bars,
 * status pills, elapsed timers, auto-refresh polling, cancellation, and simulation tools.
 */
export default function RenderQueuePage() {
  const { request, loading } = useApi();
  const [jobs, setJobs] = useState([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [stats, setStats] = useState({ total: 0, queued: 0, running: 0, completed: 0, failed: 0 });
  const pollingRef = useRef(null);

  // Fetch queue from backend
  const fetchQueue = useCallback(async () => {
    const res = await request('/api/render/queue');
    if (res.data?.jobs) {
      setJobs(res.data.jobs);
      if (res.data.stats) {
        setStats(res.data.stats);
      } else {
        // Calculate stats manually
        const jList = res.data.jobs;
        setStats({
          total: jList.length,
          queued: jList.filter((j) => j.state === 'queued').length,
          running: jList.filter((j) => j.state === 'running').length,
          completed: jList.filter((j) => j.state === 'completed').length,
          failed: jList.filter((j) => j.state === 'failed' || j.state === 'cancelled').length,
        });
      }
    }
  }, [request]);

  // Initial fetch and polling loop
  useEffect(() => {
    fetchQueue();

    if (autoRefresh) {
      pollingRef.current = setInterval(() => {
        fetchQueue();
      }, 3000);
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [autoRefresh, fetchQueue]);

  // Cancel job handler
  const handleCancelJob = async (jobId) => {
    if (!window.confirm('Are you sure you want to cancel this render job?')) return;

    try {
      const res = await fetch(`/api/render/${jobId}`, { method: 'DELETE' });
      if (res.ok) {
        // Optimistically update state
        setJobs((prev) =>
          prev.map((j) => (j.id === jobId ? { ...j, state: 'cancelled', message: 'Cancelled by user.' } : j))
        );
      }
    } catch (err) {
      console.warn('Backend cancel failed, updating locally:', err);
      setJobs((prev) =>
        prev.map((j) => (j.id === jobId ? { ...j, state: 'cancelled', message: 'Cancelled by user.' } : j))
      );
    }
  };

  // Retry job handler
  const handleRetryJob = async (job) => {
    try {
      await fetch('/api/render/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          character_id: job.metadata?.character_id || 'char_cyber_01',
          recording_id: job.metadata?.recording_id || null,
          output_width: 1080,
          output_height: 1920,
          output_fps: 30,
        }),
      });
      fetchQueue();
    } catch (err) {
      console.warn('Retry failed:', err);
    }
  };

  // Simulate a test render job for demonstration and interactive verification
  const handleSimulateJob = () => {
    const simId = `job_sim_${Date.now().toString(36)}`;
    const characterNames = ['Kaito Neon (Cyberpunk)', 'Aoi Valkyrie (Anime)', 'Eldrin Knight (Fantasy)'];
    const chosenChar = characterNames[Math.floor(Math.random() * characterNames.length)];

    const newSimJob = {
      id: simId,
      type: 'render',
      state: 'running',
      progress: 10,
      message: 'Stage 1/4: Pose Landmark Tracking (MediaPipe)',
      created_at: new Date().toISOString(),
      started_at: new Date().toISOString(),
      metadata: {
        character_name: chosenChar,
        character_id: 'char_sim',
      },
    };

    setJobs((prev) => [newSimJob, ...prev]);

    // Animate simulation progress across stages
    let currentProg = 10;
    const simInterval = setInterval(() => {
      currentProg += 15;
      if (currentProg >= 100) {
        clearInterval(simInterval);
        setJobs((prev) =>
          prev.map((j) =>
            j.id === simId
              ? {
                  ...j,
                  state: 'completed',
                  progress: 100,
                  message: 'Stage 4/4: Video remux complete. 1080x1920 Reel ready!',
                  completed_at: new Date().toISOString(),
                }
              : j
          )
        );
      } else {
        const stageMsg =
          currentProg < 35
            ? 'Stage 1/4: Pose Landmark Tracking (MediaPipe)'
            : currentProg < 65
            ? 'Stage 2/4: Diffusion Inpainting (SD1.5 / ControlNet)'
            : currentProg < 90
            ? 'Stage 3/4: Face Detailer & Temporal Smoothing'
            : 'Stage 4/4: Audio Sync & FFmpeg Remux';

        setJobs((prev) =>
          prev.map((j) =>
            j.id === simId
              ? {
                  ...j,
                  progress: currentProg,
                  message: stageMsg,
                }
              : j
          )
        );
      }
    }, 1200);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Banner */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2>AI Rendering Queue</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
            Asynchronous local AI character replacement jobs, diffusion synthesis, and FFmpeg remuxing.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            className={`btn btn-sm ${autoRefresh ? 'btn-secondary' : 'btn-ghost'}`}
            onClick={() => setAutoRefresh(!autoRefresh)}
            title="Toggle automatic 3-second queue polling"
          >
            {autoRefresh ? '🟢 Auto-Refresh ON' : '⚪ Auto-Refresh OFF'}
          </button>
          <button className="btn btn-secondary btn-sm" onClick={fetchQueue} title="Manually refresh queue">
            🔄 Refresh
          </button>
          <button className="btn btn-primary btn-sm" onClick={handleSimulateJob}>
            ⚡ Simulate Render Job
          </button>
        </div>
      </div>

      {/* Summary KPI Stats Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '12px',
        }}
      >
        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>
            TOTAL JOBS
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
            {jobs.length}
          </div>
        </div>

        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>
            ACTIVE / RUNNING
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>
            {jobs.filter((j) => j.state === 'running').length}
          </div>
        </div>

        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>
            QUEUED
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--status-warn)' }}>
            {jobs.filter((j) => j.state === 'queued').length}
          </div>
        </div>

        <div className="card" style={{ padding: '14px 16px' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>
            COMPLETED
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--status-ok)' }}>
            {jobs.filter((j) => j.state === 'completed').length}
          </div>
        </div>
      </div>

      {/* Render Queue List Component */}
      <RenderQueueComponent
        jobs={jobs}
        onCancelJob={handleCancelJob}
        onRetryJob={handleRetryJob}
        loading={loading}
      />
    </div>
  );
}
