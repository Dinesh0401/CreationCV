import React, { useEffect, useState } from 'react';
import RenderQueueComponent from '../components/RenderQueue';
import useApi from '../hooks/useApi';

export default function RenderQueuePage() {
  const { request } = useApi();
  const [queue, setQueue] = useState([]);

  useEffect(() => {
    request('/api/render/queue').then((res) => {
      if (res.data?.jobs) {
        setQueue(res.data.jobs);
      }
    });
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>AI Rendering Queue</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
            Asynchronous local AI character replacement jobs.
          </p>
        </div>
        <span className="badge badge-info">{queue.length} Active Jobs</span>
      </div>

      <RenderQueueComponent jobs={queue} />
    </div>
  );
}
