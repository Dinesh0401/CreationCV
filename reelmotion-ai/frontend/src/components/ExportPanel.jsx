import React, { useState } from 'react';

/**
 * ExportPanel Component
 * Video export presets, FFmpeg encoding parameters, bitrate controls,
 * audio settings, and color grading LUT selection.
 */
export default function ExportPanel({ onSettingsChange = () => {} }) {
  const [preset, setPreset] = useState('reels_standard');
  const [videoCodec, setVideoCodec] = useState('libx264');
  const [bitrateMb, setBitrateMb] = useState(14);
  const [fps, setFps] = useState(30);
  const [audioCodec, setAudioCodec] = useState('aac');
  const [includeWatermark, setIncludeWatermark] = useState(false);
  const [lutGrade, setLutGrade] = useState('cyberpunk');
  const [isSaved, setIsSaved] = useState(false);

  const presets = {
    reels_standard: {
      name: 'Instagram / TikTok Reels (1080×1920 30FPS H.264)',
      codec: 'libx264',
      bitrate: 14,
      fps: 30,
      audio: 'aac',
    },
    reels_ultra: {
      name: 'Ultra Smooth (1080×1920 60FPS HEVC/H.265)',
      codec: 'libx265',
      bitrate: 22,
      fps: 60,
      audio: 'aac',
    },
    web_optimized: {
      name: 'Fast Web Streaming (1080×1920 VP9 WebM)',
      codec: 'libvpx-vp9',
      bitrate: 10,
      fps: 30,
      audio: 'opus',
    },
  };

  const handlePresetSelect = (key) => {
    setPreset(key);
    const p = presets[key];
    if (p) {
      setVideoCodec(p.codec);
      setBitrateMb(p.bitrate);
      setFps(p.fps);
      setAudioCodec(p.audio);
    }
  };

  const handleSave = () => {
    setIsSaved(true);
    onSettingsChange({
      preset,
      videoCodec,
      bitrateMb,
      fps,
      audioCodec,
      includeWatermark,
      lutGrade,
    });
    setTimeout(() => setIsSaved(false), 2000);
  };

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div className="card-header">
        <span className="card-title">
          <span>📦</span>
          <span>Reel Export & FFmpeg Render Settings</span>
        </span>
        <span className="badge badge-accent">1080×1920 (9:16)</span>
      </div>

      {/* Preset Buttons */}
      <div className="control-group">
        <label className="control-label">Export Profile Preset</label>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '8px' }}>
          {Object.entries(presets).map(([key, p]) => (
            <button
              key={key}
              className={`btn btn-sm ${preset === key ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => handlePresetSelect(key)}
              style={{ justifyContent: 'flex-start', padding: '8px 12px', textAlign: 'left' }}
            >
              {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* Parameters Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '14px' }}>
        {/* Video Codec */}
        <div className="control-group">
          <label className="control-label">Video Encoder</label>
          <select value={videoCodec} onChange={(e) => setVideoCodec(e.target.value)}>
            <option value="libx264">H.264 / AVC (Broadest Compatibility)</option>
            <option value="libx265">H.265 / HEVC (High Efficiency)</option>
            <option value="h264_nvenc">NVIDIA NVENC H.264 (Hardware)</option>
            <option value="libvpx-vp9">VP9 (WebM Standard)</option>
          </select>
        </div>

        {/* Bitrate */}
        <div className="control-group">
          <label className="control-label">
            Target Bitrate: <span style={{ fontFamily: 'var(--font-mono)' }}>{bitrateMb} Mbps</span>
          </label>
          <input
            type="range"
            min="6"
            max="32"
            step="2"
            value={bitrateMb}
            onChange={(e) => setBitrateMb(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>

        {/* Audio Codec */}
        <div className="control-group">
          <label className="control-label">Audio Codec</label>
          <select value={audioCodec} onChange={(e) => setAudioCodec(e.target.value)}>
            <option value="aac">AAC 192kbps (Stereo)</option>
            <option value="opus">Opus 160kbps (Low-Latency)</option>
            <option value="mp3">MP3 256kbps</option>
          </select>
        </div>

        {/* Color LUT Grade */}
        <div className="control-group">
          <label className="control-label">Color Grading Filter</label>
          <select value={lutGrade} onChange={(e) => setLutGrade(e.target.value)}>
            <option value="cyberpunk">Cyberpunk Neon Contrast</option>
            <option value="golden">Golden Hour Sunlight</option>
            <option value="vivid">Vivid High Saturation</option>
            <option value="cinematic">Cinematic 35mm Film</option>
            <option value="natural">Natural Rec.709 Pass-through</option>
          </select>
        </div>
      </div>

      {/* Watermark toggle */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '8px', borderTop: '1px solid var(--border-subtle)' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={includeWatermark}
            onChange={(e) => setIncludeWatermark(e.target.checked)}
          />
          <span>Embed "Created with ReelMotion AI" corner mark</span>
        </label>

        <button className="btn btn-primary btn-sm" onClick={handleSave}>
          {isSaved ? '✅ Settings Applied!' : 'Save Export Preset'}
        </button>
      </div>
    </div>
  );
}
