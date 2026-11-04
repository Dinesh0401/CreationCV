import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import HardwareCard from '../components/HardwareCard';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorState from '../components/ErrorState';
import useApi from '../hooks/useApi';

export default function Dashboard({ wsState, liveTelemetry }) {
  const navigate = useNavigate();
  const { request, loading } = useApi();

  const [healthData, setHealthData] = useState(null);
  const [hardwareData, setHardwareData] = useState(null);
  const [compatData, setCompatData] = useState(null);
  const [fetchError, setFetchError] = useState(null);

  const fetchDashboardData = async () => {
    setFetchError(null);
    const [hRes, hwRes, compRes] = await Promise.all([
      request('/api/health'),
      request('/api/hardware'),
      request('/api/hardware/compatibility'),
    ]);

    if (hRes.error || hwRes.error) {
      setFetchError(
        'Backend server is not running or unreachable. Start it with: python -m uvicorn backend.main:app'
      );
      return;
    }

    setHealthData(hRes.data);
    setHardwareData(hwRes.data);
    setCompatData(compRes.data);
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  if (loading && !hardwareData) {
    return <LoadingSpinner message="Detecting system hardware and inspecting environment..." size="lg" />;
  }

  if (fetchError && !hardwareData) {
    return (
      <ErrorState
        title="Backend Offline"
        message={fetchError}
        retryAction={fetchDashboardData}
        icon="🔌"
      />
    );
  }

  const gpu = hardwareData?.gpu || {};
  const cpu = hardwareData?.cpu || {};
  const ram = hardwareData?.ram || {};
  const sw = hardwareData?.software || {};
  const tier = hardwareData?.tier || 'unknown';

  // Live overrides if WebSocket provides them
  const liveRamUsed = liveTelemetry?.ram?.used_mb ?? ram.used_mb;
  const liveRamPct = liveTelemetry?.ram?.percent ?? ram.percent_used;
  const liveCpuPct = liveTelemetry?.cpu_percent ?? 0;
  const liveGpuUsed = liveTelemetry?.gpu?.vram_used_mb ?? gpu.vram_used_mb;

  const gpuPercent =
    gpu.vram_total_mb > 0 ? (liveGpuUsed / gpu.vram_total_mb) * 100 : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* ── System Status Banner ── */}
      <div className="status-grid">
        <div className="card">
          <div className="card-subtitle">Server Status</div>
          <div className="card-value" style={{ color: 'var(--status-ok)', fontSize: '1.2rem' }}>
            ● Online
          </div>
          <div className="card-detail">Uptime: {healthData?.uptime_seconds || 0}s</div>
        </div>

        <div className="card">
          <div className="card-subtitle">WebSocket Telemetry</div>
          <div
            className="card-value"
            style={{
              color: wsState === 'connected' ? 'var(--status-ok)' : 'var(--status-warn)',
              fontSize: '1.2rem',
            }}
          >
            {wsState === 'connected' ? '● Connected' : '○ Standby'}
          </div>
          <div className="card-detail">Channel: hardware, jobs</div>
        </div>

        <div className="card">
          <div className="card-subtitle">Hardware Tier</div>
          <div className="card-value" style={{ color: 'var(--accent)', fontSize: '1.2rem' }}>
            {tier.toUpperCase()}
          </div>
          <div className="card-detail">
            {tier === 'low_vram'
              ? '4-6GB: Optimized inference'
              : tier === 'mid_vram'
              ? '8-12GB: Standard inference'
              : tier === 'high_vram'
              ? '16GB+: Full inference'
              : 'CPU Only mode'}
          </div>
        </div>

        <div className="card">
          <div className="card-subtitle">App Version</div>
          <div className="card-value" style={{ fontSize: '1.2rem' }}>
            v{healthData?.version || '0.1.0'}
          </div>
          <div className="card-detail">ReelMotion Studio Core</div>
        </div>
      </div>

      {/* ── Quick Actions ── */}
      <div>
        <div className="section-heading">Quick Actions</div>
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          <button className="btn btn-primary" onClick={() => navigate('/characters')}>
            🎭 Upload Character
          </button>
          <button className="btn btn-secondary" onClick={() => navigate('/camera')}>
            📷 Camera Setup
          </button>
          <button className="btn btn-secondary" onClick={() => navigate('/recording')}>
            ⏺️ Recording Studio
          </button>
          <button className="btn btn-ghost" onClick={() => navigate('/render')}>
            🔄 View Queue
          </button>
        </div>
      </div>

      {/* ── Real Hardware Monitoring ── */}
      <div>
        <div className="section-heading">System Hardware Introspection</div>
        <div className="grid-3">
          {/* GPU */}
          <HardwareCard
            title="GPU"
            icon="🎮"
            primaryValue={gpu.detected ? gpu.name : 'No GPU Detected'}
            subtitle={
              gpu.detected
                ? `${(liveGpuUsed / 1024).toFixed(1)} / ${(gpu.vram_total_mb / 1024).toFixed(1)} GB VRAM`
                : 'Dedicated GPU not available'
            }
            usagePercent={gpu.detected ? gpuPercent : null}
            status={gpu.detected ? (gpuPercent > 85 ? 'error' : gpuPercent > 60 ? 'warn' : 'ok') : 'error'}
            rows={[
              { label: 'Driver', value: gpu.driver_version || 'N/A' },
              { label: 'Detection Method', value: gpu.detection_method || 'none' },
              {
                label: 'Free VRAM',
                value: gpu.detected
                  ? `${Math.max(0, gpu.vram_total_mb - liveGpuUsed)} MB`
                  : '0 MB',
              },
            ]}
          />

          {/* CPU */}
          <HardwareCard
            title="Processor"
            icon="💻"
            primaryValue={cpu.name}
            subtitle={`${cpu.physical_cores} Physical / ${cpu.logical_cores} Logical Cores`}
            usagePercent={liveCpuPct}
            status="ok"
            rows={[
              { label: 'Architecture', value: cpu.architecture || 'x86_64' },
              { label: 'Max Frequency', value: cpu.max_frequency_mhz ? `${cpu.max_frequency_mhz} MHz` : 'Dynamic' },
              { label: 'Active Utilization', value: `${Math.round(liveCpuPct)}%` },
            ]}
          />

          {/* RAM */}
          <HardwareCard
            title="System RAM"
            icon="🧠"
            primaryValue={`${(ram.total_mb / 1024).toFixed(1)} GB Total`}
            subtitle={`${(liveRamUsed / 1024).toFixed(1)} GB in use`}
            usagePercent={liveRamPct}
            status={liveRamPct > 85 ? 'error' : liveRamPct > 65 ? 'warn' : 'ok'}
            rows={[
              { label: 'Available Memory', value: `${(ram.available_mb / 1024).toFixed(1)} GB` },
              { label: 'Memory Percent', value: `${Math.round(liveRamPct)}%` },
            ]}
          />
        </div>
      </div>

      {/* ── Environment & Compatibility ── */}
      <div>
        <div className="section-heading">Environment & AI Engine Readiness</div>
        <div className="grid-2">
          {/* Software environment */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">📦 Installed Runtime & Tools</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div className="hw-card-row">
                <span className="hw-card-label">Python Version</span>
                <span className="hw-card-val">{sw.python_version || 'N/A'}</span>
              </div>
              <div className="hw-card-row">
                <span className="hw-card-label">PyTorch Installed</span>
                <span className="hw-card-val">
                  {sw.torch_installed ? `Yes (${sw.torch_version})` : 'Not installed'}
                </span>
              </div>
              <div className="hw-card-row">
                <span className="hw-card-label">CUDA Acceleration</span>
                <span className="hw-card-val">
                  {sw.cuda_available ? (
                    <span className="badge badge-ok">CUDA Active ({sw.cuda_version})</span>
                  ) : (
                    <span className="badge badge-warn">CPU Only (Phase 1 OK)</span>
                  )}
                </span>
              </div>
              <div className="hw-card-row">
                <span className="hw-card-label">FFmpeg Video Engine</span>
                <span className="hw-card-val">
                  {sw.ffmpeg_available ? (
                    <span className="badge badge-ok">Available</span>
                  ) : (
                    <span className="badge badge-error">Missing (Needed for Phase 7)</span>
                  )}
                </span>
              </div>
            </div>
          </div>

          {/* Model compatibility profile */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">🎯 Model Capability Profile</span>
              <span className="badge badge-accent">{tier}</span>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginBottom: '12px' }}>
              {compatData?.preset ? (
                <>
                  Max Resolution: <strong>{compatData.preset.max_resolution}px</strong> | Max
                  Frames: <strong>{compatData.preset.max_frames}</strong> | Precision:{' '}
                  <strong>{compatData.preset.precision}</strong>
                </>
              ) : (
                'System will automatically tune generation parameters for your VRAM.'
              )}
            </p>
            <div className="section-heading" style={{ fontSize: '0.72rem' }}>
              Recommended Models for this hardware
            </div>
            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {compatData?.compatible_models?.map((m) => (
                <li
                  key={m.id}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '0.8rem',
                    padding: '4px 8px',
                    background: 'var(--bg-surface)',
                    borderRadius: '4px',
                  }}
                >
                  <span>{m.name}</span>
                  <span className="badge badge-ok">Compatible</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

