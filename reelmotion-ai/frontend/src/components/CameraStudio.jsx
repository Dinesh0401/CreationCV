import React, { useEffect, useRef, useState, useCallback } from 'react';

/**
 * CameraStudio Component
 * Real MediaStream webcam & microphone capture with 9:16 portrait overlay,
 * full-body skeletal framing guide, and real-time Web Audio API VU meter.
 */
export default function CameraStudio({
  onStreamReady,
  onStreamStop,
  showControls = true,
  defaultFramingGuide = true,
  defaultPortraitGuide = true,
  preferredResolution = '720p',
  preferredFps = 30,
  isRecording = false,
  className = '',
}) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animFrameRef = useRef(null);

  // Device lists
  const [videoDevices, setVideoDevices] = useState([]);
  const [audioDevices, setAudioDevices] = useState([]);
  const [selectedVideoId, setSelectedVideoId] = useState('');
  const [selectedAudioId, setSelectedAudioId] = useState('');

  // UI state
  const [showFramingGuide, setShowFramingGuide] = useState(defaultFramingGuide);
  const [showPortraitGuide, setShowPortraitGuide] = useState(defaultPortraitGuide);
  const [isMirrored, setIsMirrored] = useState(true);
  const [audioMuted, setAudioMuted] = useState(false);
  const [error, setError] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [streamMeta, setStreamMeta] = useState({ width: 0, height: 0, fps: 0, label: '' });

  // VU Meter state
  const [audioLevel, setAudioLevel] = useState(0); // 0 - 100
  const [audioPeak, setAudioPeak] = useState(0); // 0 - 100
  const [audioDb, setAudioDb] = useState(-60); // dB

  // Resolution map
  const resolutionMap = {
    '480p': { width: 854, height: 480 },
    '720p': { width: 1280, height: 720 },
    '1080p': { width: 1920, height: 1080 },
  };

  // Enumerate devices
  const updateDeviceList = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const vDevs = devices.filter((d) => d.kind === 'videoinput');
      const aDevs = devices.filter((d) => d.kind === 'audioinput');

      setVideoDevices(vDevs);
      setAudioDevices(aDevs);

      // Auto-select first devices if not selected
      if (!selectedVideoId && vDevs.length > 0) {
        setSelectedVideoId(vDevs[0].deviceId);
      }
      if (!selectedAudioId && aDevs.length > 0) {
        setSelectedAudioId(aDevs[0].deviceId);
      }
    } catch (err) {
      console.warn('Error enumerating devices:', err);
    }
  }, [selectedVideoId, selectedAudioId]);

  // Clean up Web Audio VU meter
  const stopAudioAnalyser = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (audioContextRef.current) {
      try {
        if (audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close();
        }
      } catch (e) {
        // ignore
      }
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    setAudioLevel(0);
    setAudioPeak(0);
    setAudioDb(-60);
  }, []);

  // Setup Web Audio Analyser for VU Meter
  const setupAudioAnalyser = useCallback((stream) => {
    stopAudioAnalyser();

    const audioTracks = stream.getAudioTracks();
    if (audioTracks.length === 0) return;

    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;

      const audioCtx = new AudioCtx();
      audioContextRef.current = audioCtx;

      // Ensure AudioContext is running if suspended
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.6;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      let currentPeak = 0;

      const updateVU = () => {
        if (!analyserRef.current) return;

        analyserRef.current.getByteTimeDomainData(dataArray);

        // Compute Root Mean Square (RMS) volume
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          const val = (dataArray[i] - 128) / 128;
          sum += val * val;
        }
        const rms = Math.sqrt(sum / bufferLength);

        // Calculate Decibels roughly (-60dB to 0dB)
        const db = rms > 0 ? Math.max(-60, Math.round(20 * Math.log10(rms))) : -60;
        setAudioDb(db);

        // Map to percentage 0 - 100 with non-linear boost for speech sensitivity
        const pct = Math.min(100, Math.round(Math.pow(rms, 0.6) * 120));
        setAudioLevel(pct);

        if (pct > currentPeak) {
          currentPeak = pct;
        } else {
          currentPeak = Math.max(0, currentPeak - 1.5);
        }
        setAudioPeak(Math.round(currentPeak));

        animFrameRef.current = requestAnimationFrame(updateVU);
      };

      animFrameRef.current = requestAnimationFrame(updateVU);
    } catch (err) {
      console.warn('AudioContext setup error:', err);
    }
  }, [stopAudioAnalyser]);

  // Start or switch media stream
  const startStream = useCallback(async () => {
    setError(null);

    // Stop existing tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    stopAudioAnalyser();

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('MediaDevices getUserMedia is not supported in this browser.');
      return;
    }

    const resConfig = resolutionMap[preferredResolution] || resolutionMap['720p'];

    const videoConstraints = {
      width: { ideal: resConfig.width },
      height: { ideal: resConfig.height },
      frameRate: { ideal: preferredFps },
    };

    if (selectedVideoId) {
      videoConstraints.deviceId = { exact: selectedVideoId };
    }

    const audioConstraints = selectedAudioId
      ? { deviceId: { exact: selectedAudioId }, echoCancellation: true, noiseSuppression: true }
      : { echoCancellation: true, noiseSuppression: true };

    let mediaStream = null;

    try {
      // First attempt: Request both Video + Audio
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: audioConstraints,
      });
    } catch (err) {
      console.warn('getUserMedia with audio failed, attempting video-only fallback:', err);
      try {
        // Fallback: Video only
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: videoConstraints,
          audio: false,
        });
      } catch (videoErr) {
        console.error('getUserMedia failed completely:', videoErr);
        setError(
          videoErr.name === 'NotAllowedError'
            ? 'Camera/Microphone permission was denied. Please allow access in browser settings.'
            : videoErr.name === 'NotFoundError'
            ? 'No camera or microphone found on this device.'
            : `Failed to access camera: ${videoErr.message || videoErr.name}`
        );
        setCameraActive(false);
        if (onStreamStop) onStreamStop();
        return;
      }
    }

    streamRef.current = mediaStream;

    // Attach stream to video element
    if (videoRef.current) {
      videoRef.current.srcObject = mediaStream;
      videoRef.current.onloadedmetadata = () => {
        const vTrack = mediaStream.getVideoTracks()[0];
        const settings = vTrack ? vTrack.getSettings() : {};
        const meta = {
          width: videoRef.current.videoWidth || settings.width || resConfig.width,
          height: videoRef.current.videoHeight || settings.height || resConfig.height,
          fps: settings.frameRate || preferredFps,
          label: vTrack ? vTrack.label : 'Webcam',
        };
        setStreamMeta(meta);
        setCameraActive(true);

        if (onStreamReady) {
          onStreamReady(mediaStream, meta);
        }
      };
    }

    // Setup VU Meter
    setupAudioAnalyser(mediaStream);

    // Refresh devices list with granted permissions (labels will now be visible)
    updateDeviceList();
  }, [
    selectedVideoId,
    selectedAudioId,
    preferredResolution,
    preferredFps,
    stopAudioAnalyser,
    setupAudioAnalyser,
    updateDeviceList,
    onStreamReady,
    onStreamStop,
  ]);

  // Initialize on mount
  useEffect(() => {
    updateDeviceList();
    startStream();

    const handleDeviceChange = () => {
      updateDeviceList();
    };

    if (navigator.mediaDevices?.addEventListener) {
      navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);
    }

    return () => {
      if (navigator.mediaDevices?.removeEventListener) {
        navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
      }
      stopAudioAnalyser();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (onStreamStop) onStreamStop();
    };
  }, []);

  // Re-start stream when selected device changes or preferred resolution/fps change
  useEffect(() => {
    if (selectedVideoId || selectedAudioId) {
      startStream();
    }
  }, [selectedVideoId, selectedAudioId, preferredResolution, preferredFps]);

  // Toggle mic track mute
  const toggleMute = () => {
    if (streamRef.current) {
      const audioTracks = streamRef.current.getAudioTracks();
      audioTracks.forEach((track) => {
        track.enabled = audioMuted;
      });
      setAudioMuted(!audioMuted);
    }
  };

  return (
    <div className={`camera-studio-container ${className}`}>
      {/* Top Header Bar */}
      {showControls && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '10px 16px',
            borderBottom: '1px solid var(--border-default)',
            background: 'var(--bg-topbar)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '1.1rem' }}>📹</span>
            <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Studio Camera</span>
            {cameraActive && (
              <span className="badge badge-ok" style={{ fontSize: '0.7rem' }}>
                LIVE {streamMeta.width}×{streamMeta.height} @ {Math.round(streamMeta.fps)}FPS
              </span>
            )}
          </div>

          {/* Quick HUD Toggles */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              className={`btn btn-sm ${showPortraitGuide ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setShowPortraitGuide(!showPortraitGuide)}
              title="Toggle 9:16 Portrait Reel Mask"
            >
              📱 9:16 Mask
            </button>
            <button
              className={`btn btn-sm ${showFramingGuide ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setShowFramingGuide(!showFramingGuide)}
              title="Toggle Full-body Pose Framing Guide"
            >
              🧍 Framing Guide
            </button>
            <button
              className={`btn btn-sm ${isMirrored ? 'btn-secondary' : 'btn-ghost'}`}
              onClick={() => setIsMirrored(!isMirrored)}
              title="Mirror Webcam Preview"
            >
              🪞 {isMirrored ? 'Mirrored' : 'Normal'}
            </button>
            <button
              className="btn btn-sm btn-ghost"
              onClick={startStream}
              title="Reconnect Camera"
            >
              🔄
            </button>
          </div>
        </div>
      )}

      {/* Main Viewport Container */}
      <div className="camera-viewport">
        {/* Live Video Element */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`camera-video-element ${isMirrored ? 'mirrored' : ''}`}
        />

        {/* Recording Active Pill Indicator */}
        {isRecording && (
          <div className="recording-live-pill">
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#fff' }} />
            <span>REC 9:16</span>
          </div>
        )}

        {/* 9:16 Portrait Overlay Guide */}
        {showPortraitGuide && (
          <div className="portrait-mask-overlay">
            <div className="portrait-pillarbox-left" />
            <div className={`portrait-frame-box ${isRecording ? 'recording' : ''}`}>
              {/* Corner brackets */}
              <div className="corner-bracket corner-tl" />
              <div className="corner-bracket corner-tr" />
              <div className="corner-bracket corner-bl" />
              <div className="corner-bracket corner-br" />

              {/* Full-Body Skeletal Framing Guide Overlay */}
              {showFramingGuide && (
                <div className="framing-hud">
                  {/* Center Alignment Axis */}
                  <div className="framing-center-axis" />

                  {/* Head Oval Target */}
                  <div className="framing-head-oval" />

                  {/* Line 1: Head / Eyes */}
                  <div className="framing-line-group framing-line-head">
                    <span className="framing-tag">HEAD</span>
                    <div className="framing-line" />
                    <span className="framing-tag">EYES</span>
                  </div>

                  {/* Line 2: Torso / Chest */}
                  <div className="framing-line-group framing-line-chest">
                    <span className="framing-tag">CHEST / TORSO</span>
                    <div className="framing-line" />
                  </div>

                  {/* Line 3: Hands / Hip */}
                  <div className="framing-line-group framing-line-hands">
                    <span className="framing-tag">HANDS / WAIST</span>
                    <div className="framing-line" />
                  </div>

                  {/* Line 4: Legs / Feet Ground */}
                  <div className="framing-line-group framing-line-legs">
                    <span className="framing-tag">LEGS / FEET</span>
                    <div className="framing-line" />
                    <span className="framing-tag">GROUND</span>
                  </div>
                </div>
              )}
            </div>
            <div className="portrait-pillarbox-right" />
          </div>
        )}

        {/* Error State Banner */}
        {error && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background: 'rgba(10, 10, 15, 0.9)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '24px',
              textAlign: 'center',
              zIndex: 20,
            }}
          >
            <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>⚠️</div>
            <h4 style={{ color: 'var(--status-error)', marginBottom: '8px' }}>Camera Access Required</h4>
            <p style={{ color: 'var(--text-secondary)', maxWidth: '420px', fontSize: '0.85rem', marginBottom: '16px' }}>
              {error}
            </p>
            <button className="btn btn-primary" onClick={startStream}>
              Grant Permissions & Retry
            </button>
          </div>
        )}
      </div>

      {/* Bottom Bar: Device Switcher & Real-Time Audio VU Meter */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '10px',
          padding: '12px 16px',
          background: 'var(--bg-card)',
          borderTop: '1px solid var(--border-default)',
        }}
      >
        {/* Device Switcher Row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          {/* Webcam Switcher */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flex: 1, minWidth: '220px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)', whiteSpace: 'nowrap' }}>
              Webcam:
            </span>
            <select
              style={{ flex: 1, fontSize: '0.82rem', padding: '6px 10px' }}
              value={selectedVideoId}
              onChange={(e) => setSelectedVideoId(e.target.value)}
              disabled={videoDevices.length === 0}
            >
              {videoDevices.length > 0 ? (
                videoDevices.map((d, i) => (
                  <option key={d.deviceId || i} value={d.deviceId}>
                    {d.label || `Camera ${i + 1}`}
                  </option>
                ))
              ) : (
                <option value="">No Camera Detected</option>
              )}
            </select>
          </div>

          {/* Microphone Switcher */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flex: 1, minWidth: '220px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-tertiary)', whiteSpace: 'nowrap' }}>
              Mic:
            </span>
            <select
              style={{ flex: 1, fontSize: '0.82rem', padding: '6px 10px' }}
              value={selectedAudioId}
              onChange={(e) => setSelectedAudioId(e.target.value)}
              disabled={audioDevices.length === 0}
            >
              {audioDevices.length > 0 ? (
                audioDevices.map((d, i) => (
                  <option key={d.deviceId || i} value={d.deviceId}>
                    {d.label || `Microphone ${i + 1}`}
                  </option>
                ))
              ) : (
                <option value="">No Microphone Detected</option>
              )}
            </select>
            <button
              className={`btn btn-sm ${audioMuted ? 'btn-danger' : 'btn-secondary'}`}
              onClick={toggleMute}
              title={audioMuted ? 'Unmute Microphone' : 'Mute Microphone'}
            >
              {audioMuted ? '🔇' : '🎙️'}
            </button>
          </div>
        </div>

        {/* Real-time Audio VU Meter */}
        <div className="vu-meter-container">
          <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-secondary)', minWidth: '60px' }}>
            VU LEVEL
          </span>

          <div className="vu-meter-bar-outer">
            <div
              className="vu-meter-bar-inner"
              style={{
                width: `${audioMuted ? 0 : audioLevel}%`,
              }}
            />
            {audioPeak > 0 && !audioMuted && (
              <div
                className="vu-meter-peak"
                style={{
                  left: `${Math.min(99, audioPeak)}%`,
                }}
              />
            )}
          </div>

          <span className="vu-meter-value">
            {audioMuted ? 'MUTED' : `${audioDb} dB`}
          </span>

          <span
            className={`badge ${
              audioMuted
                ? 'badge-warn'
                : audioLevel > 85
                ? 'badge-error'
                : audioLevel > 15
                ? 'badge-ok'
                : 'badge-info'
            }`}
            style={{ fontSize: '0.68rem', padding: '2px 6px' }}
          >
            {audioMuted ? 'OFF' : audioLevel > 85 ? 'PEAK' : audioLevel > 15 ? 'OPTIMAL' : 'LOW'}
          </span>
        </div>
      </div>
    </div>
  );
}
