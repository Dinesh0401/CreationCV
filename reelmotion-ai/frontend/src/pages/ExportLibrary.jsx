import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ExportPanel from '../components/ExportPanel';
import useApi from '../hooks/useApi';

/**
 * ExportLibrary Page
 * Browse finished 1080x1920 MP4 Reels, in-browser portrait video player,
 * direct MP4 download, and character re-render options.
 */
export default function ExportLibrary() {
  const navigate = useNavigate();
  const { get, post } = useApi();

  const [reels, setReels] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('all');

  // In-browser video player modal state
  const [selectedReel, setSelectedReel] = useState(null);

  // Re-render modal state
  const [reRenderTarget, setReRenderTarget] = useState(null);
  const [reRenderCharacter, setReRenderCharacter] = useState('char_cyber_01');
  const [reRenderRes, setReRenderRes] = useState('1080p');
  const [reRenderFps, setReRenderFps] = useState(30);

  // Default initial showcase reels
  const defaultReels = [
    {
      id: 'reel_kaito_neon_01',
      title: 'Neon Cyberpunk Dance Reel',
      character_name: 'Kaito Neon',
      character_id: 'char_cyber_01',
      duration_seconds: 14.8,
      resolution: [1080, 1920],
      fps: 30,
      file_size_mb: 22.4,
      created_at: new Date(Date.now() - 3600000 * 2).toISOString(),
      thumbnail_icon: '🤖',
      tags: ['Cyberpunk', '1080×1920', '30 FPS'],
      video_src: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
    },
    {
      id: 'reel_aoi_valkyrie_02',
      title: 'Aoi Valkyrie Martial Reel',
      character_name: 'Aoi Valkyrie',
      character_id: 'char_anime_02',
      duration_seconds: 19.2,
      resolution: [1080, 1920],
      fps: 60,
      file_size_mb: 31.8,
      created_at: new Date(Date.now() - 3600000 * 5).toISOString(),
      thumbnail_icon: '⚔️',
      tags: ['Anime', '1080×1920', '60 FPS'],
      video_src: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4',
    },
    {
      id: 'reel_eldrin_knight_03',
      title: 'Eldrin Knight Battle Stance',
      character_name: 'Eldrin Knight',
      character_id: 'char_fantasy_03',
      duration_seconds: 11.5,
      resolution: [1080, 1920],
      fps: 30,
      file_size_mb: 18.0,
      created_at: new Date(Date.now() - 3600000 * 18).toISOString(),
      thumbnail_icon: '🛡️',
      tags: ['Fantasy', 'Cinematic', '30 FPS'],
      video_src: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4',
    },
  ];

  // Fetch exports from backend
  useEffect(() => {
    get('/api/exports').then((res) => {
      if (res.data && Array.isArray(res.data) && res.data.length > 0) {
        setReels(res.data);
      } else {
        setReels(defaultReels);
      }
    });
  }, [get]);

  // Filtered reels list
  const filteredReels = reels.filter((r) => {
    const matchesSearch =
      r.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.character_name.toLowerCase().includes(searchQuery.toLowerCase());

    if (!matchesSearch) return false;
    if (activeFilter === 'all') return true;
    return r.tags?.some((t) => t.toLowerCase().includes(activeFilter.toLowerCase()));
  });

  // Direct MP4 Download trigger
  const handleDownload = (reel, e) => {
    if (e) e.stopPropagation();

    // Trigger download
    const link = document.createElement('a');
    link.href = reel.video_src || reel.download_url || '#';
    link.download = `${reel.title.replace(/\s+/g, '_')}_1080x1920.mp4`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Submit Re-render to queue
  const handleStartReRender = async () => {
    if (!reRenderTarget) return;

    try {
      await fetch('/api/render/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          character_id: reRenderCharacter,
          recording_id: reRenderTarget.id,
          output_width: 1080,
          output_height: 1920,
          output_fps: reRenderFps,
          output_format: 'mp4',
        }),
      });
      setReRenderTarget(null);
      navigate('/render');
    } catch (err) {
      console.warn('Re-render dispatch error, redirecting to queue:', err);
      setReRenderTarget(null);
      navigate('/render');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Banner */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2>Export Library</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
            Browse, play, download, and re-render finished 1080×1920 MP4 vertical Reels.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span className="badge badge-accent">{reels.length} Finished Reels</span>
          <button className="btn btn-primary btn-sm" onClick={() => navigate('/recording')}>
            + Record New Reel
          </button>
        </div>
      </div>

      {/* Export Settings Panel Component */}
      <ExportPanel />

      {/* Filter and Search Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        {/* Search */}
        <div style={{ position: 'relative', minWidth: '260px' }}>
          <input
            type="text"
            placeholder="Search by title or character..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{ width: '100%', paddingLeft: '32px' }}
          />
          <span style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', opacity: 0.5 }}>
            🔍
          </span>
        </div>

        {/* Filter Pills */}
        <div style={{ display: 'flex', gap: '6px' }}>
          {['all', 'cyberpunk', 'anime', 'fantasy'].map((f) => (
            <button
              key={f}
              className={`btn btn-sm ${activeFilter === f ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setActiveFilter(f)}
              style={{ textTransform: 'capitalize' }}
            >
              {f === 'all' ? 'All Reels' : f}
            </button>
          ))}
        </div>
      </div>

      {/* Reel Grid */}
      {filteredReels.length === 0 ? (
        <div className="card" style={{ padding: '48px 24px', textAlign: 'center' }}>
          <div style={{ fontSize: '2.5rem', marginBottom: '12px', opacity: 0.5 }}>🎬</div>
          <h4 style={{ color: 'var(--text-secondary)', marginBottom: '8px' }}>No Reels Matching Filter</h4>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Try clearing your search query or record a new Reel.
          </p>
        </div>
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
            gap: '20px',
          }}
        >
          {filteredReels.map((reel) => (
            <div
              key={reel.id}
              className="card"
              style={{
                display: 'flex',
                flexDirection: 'column',
                padding: '0',
                overflow: 'hidden',
                transition: 'transform var(--transition-fast), border-color var(--transition-fast)',
                cursor: 'pointer',
              }}
              onClick={() => setSelectedReel(reel)}
            >
              {/* Thumbnail Container (Portrait 9:16 Ratio Box) */}
              <div
                style={{
                  position: 'relative',
                  width: '100%',
                  aspectRatio: '9 / 16',
                  maxHeight: '340px',
                  background: '#07070c',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'hidden',
                }}
              >
                {/* Simulated Reel Poster Gradient & Icon */}
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: 'radial-gradient(circle at center, rgba(59, 130, 246, 0.25) 0%, #07070c 80%)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                  }}
                >
                  <span style={{ fontSize: '3.5rem' }}>{reel.thumbnail_icon || '👤'}</span>
                  <span
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.72rem',
                      color: 'var(--accent)',
                      background: 'rgba(0,0,0,0.6)',
                      padding: '2px 8px',
                      borderRadius: '4px',
                    }}
                  >
                    1080×1920 (9:16)
                  </span>
                </div>

                {/* Play overlay button */}
                <div
                  style={{
                    position: 'absolute',
                    width: '48px',
                    height: '48px',
                    borderRadius: '50%',
                    background: 'rgba(255,255,255,0.9)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '1.4rem',
                    color: '#000',
                    boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
                  }}
                >
                  ▶
                </div>

                {/* Duration Badge */}
                <div
                  style={{
                    position: 'absolute',
                    bottom: '8px',
                    right: '8px',
                    background: 'rgba(0,0,0,0.85)',
                    padding: '3px 6px',
                    borderRadius: '4px',
                    fontSize: '0.7rem',
                    fontFamily: 'var(--font-mono)',
                    color: '#fff',
                  }}
                >
                  {reel.duration_seconds}s
                </div>
              </div>

              {/* Card Body */}
              <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: '8px', flex: 1 }}>
                <div>
                  <h4 style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {reel.title}
                  </h4>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-tertiary)', marginTop: '2px' }}>
                    Character: <strong>{reel.character_name}</strong>
                  </div>
                </div>

                {/* Tags */}
                <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                  {reel.tags?.map((tag, idx) => (
                    <span
                      key={idx}
                      className="badge badge-accent"
                      style={{ fontSize: '0.65rem', padding: '2px 6px' }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '0.72rem',
                    color: 'var(--text-tertiary)',
                    marginTop: 'auto',
                    paddingTop: '8px',
                    borderTop: '1px solid var(--border-subtle)',
                  }}
                >
                  <span>{reel.file_size_mb} MB</span>
                  <span>{new Date(reel.created_at).toLocaleDateString()}</span>
                </div>

                {/* Card Actions */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '6px' }}>
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={(e) => handleDownload(reel, e)}
                  >
                    ⬇️ Download
                  </button>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      setReRenderTarget(reel);
                    }}
                  >
                    🔄 Re-Render
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* In-Browser Video Player Modal */}
      {selectedReel && (
        <div className="modal-backdrop" onClick={() => setSelectedReel(null)}>
          <div
            className="modal-content"
            style={{ maxWidth: '640px', padding: '24px' }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div>
                <h3 style={{ fontSize: '1.1rem' }}>{selectedReel.title}</h3>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Character: {selectedReel.character_name} • 1080×1920 (9:16 Reel)
                </div>
              </div>
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setSelectedReel(null)}
                style={{ fontSize: '1.2rem', padding: '4px 8px' }}
              >
                ✕
              </button>
            </div>

            {/* In-Browser Video Player in Vertical 9:16 Frame */}
            <div
              className="reel-player-container"
              style={{
                width: '100%',
                maxHeight: '480px',
                aspectRatio: '9 / 16',
                background: '#000',
              }}
            >
              <video
                src={selectedReel.video_src || selectedReel.download_url}
                controls
                autoPlay
                playsInline
                loop
                className="reel-player-video"
              />
            </div>

            {/* Reel Specs & Modal Actions */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginTop: '18px',
                flexWrap: 'wrap',
                gap: '12px',
              }}
            >
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-tertiary)' }}>
                {selectedReel.duration_seconds}s • {selectedReel.fps} FPS • {selectedReel.file_size_mb} MB • MP4 (H.264)
              </div>

              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    setReRenderTarget(selectedReel);
                    setSelectedReel(null);
                  }}
                >
                  🔄 Re-Render Character
                </button>
                <button className="btn btn-primary" onClick={() => handleDownload(selectedReel)}>
                  ⬇️ Download MP4
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Re-Render Configuration Modal */}
      {reRenderTarget && (
        <div className="modal-backdrop" onClick={() => setReRenderTarget(null)}>
          <div
            className="modal-content"
            style={{ maxWidth: '480px', padding: '24px' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '1.1rem' }}>🔄 Re-Render with Different AI Character</h3>
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setReRenderTarget(null)}
              >
                ✕
              </button>
            </div>

            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '16px' }}>
              Re-synthesize reel <strong>"{reRenderTarget.title}"</strong> using a different AI character avatar or upgraded resolution.
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div className="control-group">
                <label className="control-label">Select Replacement Character</label>
                <select
                  value={reRenderCharacter}
                  onChange={(e) => setReRenderCharacter(e.target.value)}
                >
                  <option value="char_cyber_01">Kaito Neon (Cyberpunk Glow)</option>
                  <option value="char_anime_02">Aoi Valkyrie (Anime Action)</option>
                  <option value="char_fantasy_03">Eldrin Knight (Cinematic Plate)</option>
                  <option value="char_scifi_04">Vortex Android (Polished Chrome)</option>
                </select>
              </div>

              <div className="control-group">
                <label className="control-label">Target Resolution</label>
                <select value={reRenderRes} onChange={(e) => setReRenderRes(e.target.value)}>
                  <option value="1080p">1080×1920 (9:16 FHD Reel)</option>
                  <option value="720p">720×1280 (9:16 Fast Render)</option>
                </select>
              </div>

              <div className="control-group">
                <label className="control-label">Framerate</label>
                <select value={reRenderFps} onChange={(e) => setReRenderFps(Number(e.target.value))}>
                  <option value={30}>30 FPS (Standard)</option>
                  <option value={60}>60 FPS (Ultra Smooth)</option>
                </select>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '8px' }}>
                <button className="btn btn-ghost" onClick={() => setReRenderTarget(null)}>
                  Cancel
                </button>
                <button className="btn btn-primary" onClick={handleStartReRender}>
                  🚀 Queue Re-Render Job
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
