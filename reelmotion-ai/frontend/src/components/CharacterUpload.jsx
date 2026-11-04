import React, { useState, useRef } from 'react';

const VIEW_OPTIONS = [
  { id: 'front', label: 'Front (Full Body)', icon: '🧍' },
  { id: 'side', label: 'Side (Profile)', icon: '🚶' },
  { id: 'back', label: 'Back View', icon: '👤' },
  { id: 'face_closeup', label: 'Face Close-Up', icon: '🙂' },
  { id: 'outfit_details', label: 'Outfit Details', icon: '👔' },
];

export default function CharacterUpload({ characterId, onUploadSuccess, defaultViewType = 'front' }) {
  const [selectedView, setSelectedView] = useState(defaultViewType);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [recentReport, setRecentReport] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (!characterId) {
      setUploadError('Please select or create a character before uploading references.');
      return;
    }
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFiles(Array.from(files));
    }
  };

  const handleFileChange = (e) => {
    if (!characterId) {
      setUploadError('Please select or create a character before uploading references.');
      return;
    }
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFiles(Array.from(files));
    }
  };

  const uploadSingleFile = async (file, viewType) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('view_type', viewType);

    const res = await fetch(`/api/characters/${characterId}/upload-reference`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const errText = await res.text();
      let msg = errText;
      try {
        const parsed = JSON.parse(errText);
        msg = parsed.detail || errText;
      } catch {
        // use errText as is
      }
      throw new Error(msg);
    }

    return await res.json();
  };

  const handleFiles = async (files) => {
    setUploadError(null);
    setUploading(true);

    try {
      // If multiple files dropped, auto-assign view types if possible
      let currentViewIndex = VIEW_OPTIONS.findIndex((v) => v.id === selectedView);
      if (currentViewIndex === -1) currentViewIndex = 0;

      let lastResult = null;
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const assignedView =
          files.length === 1
            ? selectedView
            : VIEW_OPTIONS[(currentViewIndex + i) % VIEW_OPTIONS.length].id;

        const result = await uploadSingleFile(file, assignedView);
        lastResult = result;
      }

      if (lastResult) {
        setRecentReport(lastResult);
        setPreviewUrl(lastResult.image_url);
        if (onUploadSuccess) {
          onUploadSuccess(lastResult);
        }
      }
    } catch (err) {
      setUploadError(err.message || 'Reference upload failed.');
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* View Angle Selector */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
          Select Reference Angle / View:
        </label>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {VIEW_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              className={`btn btn-sm ${selectedView === opt.id ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setSelectedView(opt.id)}
              style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
            >
              <span>{opt.icon}</span>
              <span>{opt.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Drop Zone */}
      <div
        className="card"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => {
          if (!uploading && fileInputRef.current) {
            fileInputRef.current.click();
          }
        }}
        style={{
          border: isDragging ? '2px dashed var(--accent)' : '2px dashed var(--border-strong)',
          backgroundColor: isDragging ? 'var(--accent-muted)' : 'var(--bg-card)',
          padding: '28px',
          textAlign: 'center',
          cursor: uploading ? 'wait' : 'pointer',
          transition: 'all var(--transition-normal)',
          position: 'relative',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.webp"
          multiple
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        <div style={{ fontSize: '2.4rem', marginBottom: '8px' }}>
          {uploading ? '⏳' : '📤'}
        </div>
        <h4 style={{ margin: '0 0 6px 0', fontSize: '1.05rem' }}>
          {uploading
            ? 'Analyzing Image Quality...'
            : `Upload ${VIEW_OPTIONS.find((v) => v.id === selectedView)?.label} Reference`}
        </h4>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', margin: 0 }}>
          Drag & drop single or multiple PNG, JPG, WebP images, or click to browse.
          <br />
          <span style={{ color: 'var(--text-tertiary)', fontSize: '0.78rem' }}>
            Recommended: ≥ 512×512 resolution, sharp focus, clean solid background.
          </span>
        </p>

        {uploadError && (
          <div
            style={{
              marginTop: '12px',
              padding: '8px 12px',
              backgroundColor: 'var(--status-error-muted)',
              border: '1px solid var(--status-error)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--status-error)',
              fontSize: '0.82rem',
            }}
          >
            ⚠️ {uploadError}
          </div>
        )}
      </div>

      {/* Instant Validation Feedback Card */}
      {recentReport && recentReport.quality && (
        <div className="card" style={{ padding: '16px', background: 'var(--bg-surface)' }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '12px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '1.2rem' }}>✅</span>
              <div>
                <h5 style={{ margin: 0 }}>Instant AI Validation Feedback</h5>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-tertiary)' }}>
                  View: <strong>{recentReport.view_type}</strong> | File: {recentReport.filename}
                </span>
              </div>
            </div>
            <span
              className={`badge badge-${
                recentReport.quality.quality_score >= 70
                  ? 'ok'
                  : recentReport.quality.quality_score >= 50
                  ? 'warn'
                  : 'error'
              }`}
            >
              Quality {recentReport.quality.quality_score}/100
            </span>
          </div>

          <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
            {previewUrl && (
              <img
                src={previewUrl}
                alt="Uploaded reference"
                style={{
                  width: '90px',
                  height: '110px',
                  objectFit: 'cover',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-default)',
                }}
              />
            )}

            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                  gap: '8px',
                }}
              >
                {/* Resolution */}
                <div
                  style={{
                    padding: '8px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-card)',
                    fontSize: '0.8rem',
                  }}
                >
                  <div style={{ color: 'var(--text-tertiary)' }}>Resolution</div>
                  <div style={{ fontWeight: 600, marginTop: '2px' }}>
                    {recentReport.quality.width} × {recentReport.quality.height}
                  </div>
                  <span
                    className={`badge badge-${
                      recentReport.quality.is_optimal_resolution
                        ? 'ok'
                        : recentReport.quality.resolution_ok
                        ? 'warn'
                        : 'error'
                    }`}
                    style={{ marginTop: '4px', fontSize: '0.7rem' }}
                  >
                    {recentReport.quality.is_optimal_resolution
                      ? 'Optimal'
                      : recentReport.quality.resolution_ok
                      ? 'Acceptable'
                      : 'Low Res'}
                  </span>
                </div>

                {/* Sharpness */}
                <div
                  style={{
                    padding: '8px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-card)',
                    fontSize: '0.8rem',
                  }}
                >
                  <div style={{ color: 'var(--text-tertiary)' }}>Sharpness</div>
                  <div style={{ fontWeight: 600, marginTop: '2px' }}>
                    {recentReport.quality.sharpness_score}%
                  </div>
                  <span
                    className={`badge badge-${recentReport.quality.is_sharp ? 'ok' : 'error'}`}
                    style={{ marginTop: '4px', fontSize: '0.7rem' }}
                  >
                    {recentReport.quality.is_sharp ? 'Sharp Focus' : 'Blurry'}
                  </span>
                </div>

                {/* Face Visibility */}
                <div
                  style={{
                    padding: '8px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-card)',
                    fontSize: '0.8rem',
                  }}
                >
                  <div style={{ color: 'var(--text-tertiary)' }}>Face Visibility</div>
                  <div style={{ fontWeight: 600, marginTop: '2px' }}>
                    {recentReport.quality.face_detected ? 'Detected' : 'Not Clear'}
                  </div>
                  <span
                    className={`badge badge-${recentReport.quality.face_detected ? 'ok' : 'warn'}`}
                    style={{ marginTop: '4px', fontSize: '0.7rem' }}
                  >
                    {recentReport.quality.face_detected
                      ? `${Math.round(recentReport.quality.face_confidence * 100)}% Conf`
                      : 'Partial/None'}
                  </span>
                </div>

                {/* Full Body */}
                <div
                  style={{
                    padding: '8px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-card)',
                    fontSize: '0.8rem',
                  }}
                >
                  <div style={{ color: 'var(--text-tertiary)' }}>Body Completeness</div>
                  <div style={{ fontWeight: 600, marginTop: '2px' }}>
                    {recentReport.quality.is_full_body ? 'Full Body' : 'Partial'}
                  </div>
                  <span
                    className={`badge badge-${recentReport.quality.is_full_body ? 'ok' : 'info'}`}
                    style={{ marginTop: '4px', fontSize: '0.7rem' }}
                  >
                    {recentReport.quality.is_full_body ? 'All Limbs' : 'Upper / Crop'}
                  </span>
                </div>

                {/* Background */}
                <div
                  style={{
                    padding: '8px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-card)',
                    fontSize: '0.8rem',
                  }}
                >
                  <div style={{ color: 'var(--text-tertiary)' }}>Background</div>
                  <div style={{ fontWeight: 600, marginTop: '2px' }}>
                    {recentReport.quality.is_clean_background ? 'Clean' : 'Complex'}
                  </div>
                  <span
                    className={`badge badge-${
                      recentReport.quality.is_clean_background ? 'ok' : 'warn'
                    }`}
                    style={{ marginTop: '4px', fontSize: '0.7rem' }}
                  >
                    {recentReport.quality.is_clean_background ? 'Studio Ready' : 'Textured'}
                  </span>
                </div>
              </div>

              {/* Issues / Recommendations for image */}
              {recentReport.quality.issues && recentReport.quality.issues.length > 0 && (
                <div style={{ marginTop: '6px', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  <div style={{ fontWeight: 600, color: 'var(--status-warn)', marginBottom: '4px' }}>
                    Quality Notices:
                  </div>
                  <ul style={{ paddingLeft: '18px', margin: 0 }}>
                    {recentReport.quality.issues.map((iss, idx) => (
                      <li key={idx}>{iss}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
