import React, { useState, useEffect, useCallback } from 'react';
import CharacterUpload from '../components/CharacterUpload';
import useApi from '../hooks/useApi';

const VIEW_KEYS = [
  { key: 'front', label: 'Front (Full Body)', icon: '🧍' },
  { key: 'side', label: 'Side Profile', icon: '🚶' },
  { key: 'back', label: 'Back View', icon: '👤' },
  { key: 'face_closeup', label: 'Face Close-Up', icon: '🙂' },
  { key: 'outfit_details', label: 'Outfit Details', icon: '👔' },
];

export default function CharacterStudio() {
  const { request } = useApi();
  const [characters, setCharacters] = useState([]);
  const [selectedCharacter, setSelectedCharacter] = useState(null);
  const [characterDetail, setCharacterDetail] = useState(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newCharName, setNewCharName] = useState('');
  const [newCharTags, setNewCharTags] = useState('');
  const [activeUploadAngle, setActiveUploadAngle] = useState('front');
  const [feedbackMsg, setFeedbackMsg] = useState(null);

  const fetchCharacters = useCallback(async () => {
    const res = await request('/api/characters');
    if (res.data) {
      const list = Array.isArray(res.data) ? res.data : res.data.characters || [];
      setCharacters(list);

      // Auto-select first character or maintain selection
      setSelectedCharacter((prev) => {
        if (prev) {
          const match = list.find((c) => c.id === prev.id);
          return match || (list.length > 0 ? list[0] : null);
        }
        return list.length > 0 ? list[0] : null;
      });
    }
  }, [request]);

  useEffect(() => {
    fetchCharacters();
  }, [fetchCharacters]);

  const fetchDetail = useCallback(
    async (charId) => {
      if (!charId) {
        setCharacterDetail(null);
        return;
      }
      setLoadingDetail(true);
      const res = await request(`/api/characters/${charId}`);
      if (res.data) {
        setCharacterDetail(res.data);
      }
      setLoadingDetail(false);
    },
    [request]
  );

  useEffect(() => {
    if (selectedCharacter) {
      fetchDetail(selectedCharacter.id);
    } else {
      setCharacterDetail(null);
    }
  }, [selectedCharacter, fetchDetail]);

  const handleCreateCharacter = async (e) => {
    e.preventDefault();
    if (!newCharName.trim()) return;

    const tagsArray = newCharTags
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);

    const res = await request('/api/characters', {
      method: 'POST',
      body: { name: newCharName.trim(), tags: tagsArray },
    });

    if (res.data) {
      setShowCreateModal(false);
      setNewCharName('');
      setNewCharTags('');
      setFeedbackMsg(`Character "${res.data.name}" created!`);
      setTimeout(() => setFeedbackMsg(null), 3000);
      await fetchCharacters();
      setSelectedCharacter({ id: res.data.id, name: res.data.name });
    }
  };

  const handleDeleteCharacter = async (charId, name) => {
    if (!window.confirm(`Are you sure you want to delete character "${name}" and all references?`)) {
      return;
    }

    const res = await request(`/api/characters/${charId}`, { method: 'DELETE' });
    if (!res.error) {
      setFeedbackMsg(`Character "${name}" deleted.`);
      setTimeout(() => setFeedbackMsg(null), 3000);
      await fetchCharacters();
    }
  };

  const handleDeleteReference = async (viewType) => {
    if (!selectedCharacter) return;
    if (!window.confirm(`Remove reference image for ${viewType}?`)) return;

    const res = await request(`/api/characters/${selectedCharacter.id}/references/${viewType}`, {
      method: 'DELETE',
    });
    if (!res.error) {
      await fetchDetail(selectedCharacter.id);
      await fetchCharacters();
    }
  };

  const handleUploadSuccess = async () => {
    if (selectedCharacter) {
      await fetchDetail(selectedCharacter.id);
      await fetchCharacters();
    }
  };

  const readinessScore = characterDetail?.readiness?.readiness_score ?? 0;
  const isReady = characterDetail?.readiness?.is_ready ?? false;
  const singleImageMode = characterDetail?.readiness?.single_image_mode ?? false;
  const recommendations = characterDetail?.readiness?.recommendations || [];
  const warnings = characterDetail?.readiness?.warnings || [];
  const consistencyScore = characterDetail?.readiness?.consistency_score;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>Character Studio</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
            Manage character identities, multi-angle reference photos, and AI readiness validation.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => setShowCreateModal(true)}
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <span>➕</span>
            <span>New Character</span>
          </button>
          <span className="badge badge-info">Phase 4 Active</span>
        </div>
      </div>

      {feedbackMsg && (
        <div
          style={{
            padding: '10px 16px',
            borderRadius: 'var(--radius-sm)',
            backgroundColor: 'var(--accent-muted)',
            border: '1px solid var(--accent)',
            color: 'var(--accent)',
            fontSize: '0.85rem',
          }}
        >
          {feedbackMsg}
        </div>
      )}

      {/* Main Studio Grid: Left Sidebar Character Gallery, Right Details & Workspace */}
      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '20px' }}>
        {/* Left Column: Character Gallery */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <span className="section-heading" style={{ margin: 0 }}>
              Characters ({characters.length})
            </span>
          </div>

          {characters.length === 0 ? (
            <div
              className="card"
              style={{
                padding: '24px',
                textAlign: 'center',
                color: 'var(--text-tertiary)',
                fontSize: '0.85rem',
              }}
            >
              No character profiles yet. Click "+ New Character" to begin.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {characters.map((c) => {
                const isSelected = selectedCharacter?.id === c.id;
                const score = c.readiness_score || 0;
                const badgeColor =
                  score >= 70 ? 'ok' : score >= 50 ? 'info' : score > 0 ? 'warn' : 'error';

                return (
                  <div
                    key={c.id}
                    className="card"
                    onClick={() => setSelectedCharacter(c)}
                    style={{
                      padding: '12px',
                      cursor: 'pointer',
                      border: isSelected ? '1px solid var(--accent)' : '1px solid var(--border-default)',
                      backgroundColor: isSelected ? 'var(--bg-card-hover)' : 'var(--bg-card)',
                      transition: 'all var(--transition-fast)',
                    }}
                  >
                    <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                      <div
                        style={{
                          width: '46px',
                          height: '56px',
                          borderRadius: 'var(--radius-sm)',
                          backgroundColor: 'var(--bg-surface)',
                          border: '1px solid var(--border-subtle)',
                          overflow: 'hidden',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                        }}
                      >
                        {c.thumbnail_url ? (
                          <img
                            src={c.thumbnail_url}
                            alt={c.name}
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                          />
                        ) : (
                          <span style={{ fontSize: '1.4rem' }}>🎭</span>
                        )}
                      </div>

                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div
                          style={{
                            fontWeight: 600,
                            fontSize: '0.92rem',
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          }}
                        >
                          {c.name}
                        </div>
                        <div
                          style={{
                            fontSize: '0.78rem',
                            color: 'var(--text-tertiary)',
                            marginTop: '2px',
                          }}
                        >
                          {c.reference_count || 0} angle{c.reference_count === 1 ? '' : 's'}
                        </div>
                        <div style={{ marginTop: '4px', display: 'flex', gap: '6px' }}>
                          <span
                            className={`badge badge-${badgeColor}`}
                            style={{ fontSize: '0.68rem', padding: '1px 6px' }}
                          >
                            {Math.round(score)}% {c.is_ready ? 'Ready' : 'Incomplete'}
                          </span>
                          {c.single_image_mode && (
                            <span
                              className="badge badge-info"
                              style={{ fontSize: '0.68rem', padding: '1px 6px' }}
                            >
                              Single Image
                            </span>
                          )}
                        </div>
                      </div>

                      <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteCharacter(c.id, c.name);
                        }}
                        title="Delete Character"
                        style={{ padding: '4px 6px', color: 'var(--text-muted)' }}
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right Column: Character Studio Workspace */}
        {selectedCharacter ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {/* Character Header & Readiness Overview */}
            <div className="card" style={{ padding: '18px' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  flexWrap: 'wrap',
                  gap: '12px',
                }}
              >
                <div>
                  <h3 style={{ margin: 0 }}>{characterDetail?.name || selectedCharacter.name}</h3>
                  <div
                    style={{
                      fontSize: '0.8rem',
                      color: 'var(--text-tertiary)',
                      marginTop: '4px',
                    }}
                  >
                    ID: <code>{selectedCharacter.id}</code> | Created:{' '}
                    {characterDetail?.created_at
                      ? new Date(characterDetail.created_at).toLocaleDateString()
                      : 'Recently'}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <span
                    className={`badge badge-${isReady ? 'ok' : readinessScore > 0 ? 'warn' : 'error'}`}
                    style={{ fontSize: '0.85rem', padding: '4px 10px' }}
                  >
                    {isReady ? '✅ Studio Ready' : '⚠️ More References Needed'}
                  </span>
                  {singleImageMode && (
                    <span className="badge badge-info" style={{ fontSize: '0.85rem' }}>
                      Single Image Mode
                    </span>
                  )}
                </div>
              </div>

              {/* Progress Bar */}
              <div style={{ marginTop: '14px' }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '0.8rem',
                    marginBottom: '4px',
                  }}
                >
                  <span style={{ color: 'var(--text-secondary)' }}>Character Readiness Score</span>
                  <span style={{ fontWeight: 600 }}>{Math.round(readinessScore)}%</span>
                </div>
                <div
                  style={{
                    height: '8px',
                    backgroundColor: 'var(--bg-surface)',
                    borderRadius: '4px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      height: '100%',
                      width: `${Math.min(100, Math.max(0, readinessScore))}%`,
                      backgroundColor: isReady
                        ? 'var(--status-ok)'
                        : readinessScore >= 40
                        ? 'var(--status-warn)'
                        : 'var(--status-error)',
                      transition: 'width var(--transition-normal)',
                    }}
                  />
                </div>
              </div>

              {/* Consistency & Recommendation hints */}
              <div
                style={{
                  marginTop: '16px',
                  padding: '12px',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                  {consistencyScore != null && (
                    <div style={{ fontSize: '0.82rem' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Multi-Angle Consistency: </span>
                      <strong
                        style={{
                          color: consistencyScore >= 60 ? 'var(--status-ok)' : 'var(--status-warn)',
                        }}
                      >
                        {consistencyScore}%
                      </strong>
                    </div>
                  )}
                </div>

                {recommendations.length > 0 && (
                  <div style={{ marginTop: '8px' }}>
                    <div
                      style={{
                        fontSize: '0.78rem',
                        fontWeight: 600,
                        color: 'var(--cyan)',
                        marginBottom: '4px',
                      }}
                    >
                      💡 Studio Recommendations:
                    </div>
                    <ul
                      style={{
                        margin: 0,
                        paddingLeft: '18px',
                        fontSize: '0.8rem',
                        color: 'var(--text-secondary)',
                      }}
                    >
                      {recommendations.map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            {/* Reference Angles Slot Gallery */}
            <div>
              <div className="section-heading">Multi-View Reference Angles</div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                  gap: '12px',
                }}
              >
                {VIEW_KEYS.map((v) => {
                  const hasView = characterDetail?.views?.[v.key];
                  const imgUrl = characterDetail?.view_urls?.[v.key];
                  const isCurrentUpload = activeUploadAngle === v.key;

                  return (
                    <div
                      key={v.key}
                      className="card"
                      style={{
                        padding: '10px',
                        textAlign: 'center',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        minHeight: '170px',
                        border: isCurrentUpload
                          ? '1px solid var(--accent)'
                          : '1px solid var(--border-default)',
                        backgroundColor: 'var(--bg-card)',
                      }}
                    >
                      <div
                        style={{
                          fontSize: '0.78rem',
                          fontWeight: 600,
                          color: 'var(--text-secondary)',
                          marginBottom: '6px',
                        }}
                      >
                        {v.icon} {v.label}
                      </div>

                      <div
                        style={{
                          width: '100%',
                          height: '95px',
                          borderRadius: 'var(--radius-sm)',
                          backgroundColor: 'var(--bg-surface)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          overflow: 'hidden',
                          position: 'relative',
                        }}
                      >
                        {imgUrl ? (
                          <img
                            src={imgUrl}
                            alt={v.label}
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                          />
                        ) : (
                          <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                            Empty Slot
                          </div>
                        )}
                      </div>

                      <div style={{ marginTop: '8px', width: '100%', display: 'flex', gap: '4px' }}>
                        <button
                          type="button"
                          className={`btn btn-sm ${hasView ? 'btn-secondary' : 'btn-primary'}`}
                          style={{ flex: 1, fontSize: '0.72rem', padding: '3px 6px' }}
                          onClick={() => setActiveUploadAngle(v.key)}
                        >
                          {hasView ? 'Replace' : 'Upload'}
                        </button>
                        {hasView && (
                          <button
                            type="button"
                            className="btn btn-ghost btn-sm"
                            style={{ fontSize: '0.72rem', padding: '3px 6px' }}
                            onClick={() => handleDeleteReference(v.key)}
                            title="Remove view"
                          >
                            ❌
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Reference Upload Section */}
            <div className="card" style={{ padding: '18px' }}>
              <div className="section-heading" style={{ marginBottom: '12px' }}>
                Upload & Validate Reference Image
              </div>
              <CharacterUpload
                characterId={selectedCharacter.id}
                defaultViewType={activeUploadAngle}
                onUploadSuccess={handleUploadSuccess}
              />
            </div>
          </div>
        ) : (
          <div
            className="card"
            style={{
              padding: '48px',
              textAlign: 'center',
              color: 'var(--text-tertiary)',
            }}
          >
            <h3>No Character Selected</h3>
            <p style={{ fontSize: '0.85rem', marginTop: '8px' }}>
              Select a character on the left or create a new profile to upload references.
            </p>
            <button
              type="button"
              className="btn btn-primary"
              style={{ marginTop: '16px' }}
              onClick={() => setShowCreateModal(true)}
            >
              ➕ Create Your First Character
            </button>
          </div>
        )}
      </div>

      {/* New Character Modal */}
      {showCreateModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'var(--bg-overlay)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
          onClick={() => setShowCreateModal(false)}
        >
          <div
            className="card"
            style={{
              width: '420px',
              maxWidth: '90vw',
              padding: '24px',
              backgroundColor: 'var(--bg-card)',
              border: '1px solid var(--border-strong)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ margin: '0 0 16px 0' }}>Create Character Profile</h3>
            <form onSubmit={handleCreateCharacter} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                  Character Name *
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Cyberpunk Hero"
                  value={newCharName}
                  onChange={(e) => setNewCharName(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-default)',
                    backgroundColor: 'var(--bg-input)',
                    color: 'var(--text-primary)',
                  }}
                  autoFocus
                />
              </div>

              <div>
                <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
                  Tags (comma separated)
                </label>
                <input
                  type="text"
                  placeholder="e.g. scifi, neon, hero"
                  value={newCharTags}
                  onChange={(e) => setNewCharTags(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-default)',
                    backgroundColor: 'var(--bg-input)',
                    color: 'var(--text-primary)',
                  }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '8px' }}>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowCreateModal(false)}
                >
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={!newCharName.trim()}>
                  Create Profile
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
