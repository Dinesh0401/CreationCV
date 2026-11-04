import React, { useState, useEffect } from 'react';
import CharacterUpload from '../components/CharacterUpload';
import useApi from '../hooks/useApi';

export default function CharacterStudio() {
  const { request } = useApi();
  const [characters, setCharacters] = useState([]);

  useEffect(() => {
    request('/api/characters').then((res) => {
      if (res.data?.characters) {
        setCharacters(res.data.characters);
      }
    });
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2>Character Studio</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px' }}>
            Manage character references for AI transfer. Single or multi-angle references.
          </p>
        </div>
        <span className="badge badge-info">Phase 4 Ready</span>
      </div>

      <CharacterUpload />

      <div>
        <div className="section-heading">Imported Characters</div>
        {characters.length === 0 ? (
          <div className="card" style={{ padding: '32px', textAlign: 'center', color: 'var(--text-tertiary)' }}>
            No character profiles saved yet. Upload references above to create your first character.
          </div>
        ) : (
          <div className="grid-3">
            {characters.map((c) => (
              <div key={c.id} className="card">
                <h4>{c.name}</h4>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {c.reference_count} references
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
