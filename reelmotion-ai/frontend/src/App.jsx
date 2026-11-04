import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import StatusBar from './components/StatusBar';
import useWebSocket from './hooks/useWebSocket';

import Dashboard from './pages/Dashboard';
import CharacterStudio from './pages/CharacterStudio';
import CameraSetup from './pages/CameraSetup';
import RecordingStudio from './pages/RecordingStudio';
import RenderQueuePage from './pages/RenderQueue';
import ExportLibrary from './pages/ExportLibrary';
import Settings from './pages/Settings';

export default function App() {
  const { state: wsState, hardwareStats } = useWebSocket();

  return (
    <div className="app-layout">
      <Sidebar />
      <div className="app-main-wrapper">
        <StatusBar wsState={wsState} hardwareStats={hardwareStats} />
        <main className="app-content">
          <Routes>
            <Route
              path="/"
              element={<Dashboard wsState={wsState} liveTelemetry={hardwareStats} />}
            />
            <Route path="/characters" element={<CharacterStudio />} />
            <Route path="/camera" element={<CameraSetup />} />
            <Route path="/recording" element={<RecordingStudio />} />
            <Route path="/render" element={<RenderQueuePage />} />
            <Route path="/exports" element={<ExportLibrary />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

