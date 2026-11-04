import React from 'react';

export default function HardwareCard({
  title,
  icon,
  primaryValue,
  subtitle,
  detail,
  usagePercent,
  status = 'ok',
  rows = [],
}) {
  const getFillClass = (pct) => {
    if (pct < 60) return 'ok';
    if (pct < 85) return 'warn';
    return 'error';
  };

  return (
    <div className="card hw-card">
      <div className="card-header">
        <span className="card-title">
          {icon && <span className="card-title-icon">{icon}</span>}
          {title}
        </span>
        {status && (
          <span className={`badge badge-${status}`}>
            {status}
          </span>
        )}
      </div>

      {primaryValue && <div className="card-value">{primaryValue}</div>}
      {subtitle && <div className="card-subtitle">{subtitle}</div>}

      {usagePercent != null && (
        <div className="usage-bar-wrapper">
          <div className="usage-bar-labels">
            <span>Utilization</span>
            <span>{Math.round(usagePercent)}%</span>
          </div>
          <div className="usage-bar">
            <div
              className={`usage-bar-fill ${getFillClass(usagePercent)}`}
              style={{ width: `${Math.min(100, Math.max(0, usagePercent))}%` }}
            />
          </div>
        </div>
      )}

      {rows.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '8px' }}>
          {rows.map((r, idx) => (
            <div key={idx} className="hw-card-row">
              <span className="hw-card-label">{r.label}</span>
              <span className="hw-card-val">{r.value}</span>
            </div>
          ))}
        </div>
      )}

      {detail && <div className="card-detail">{detail}</div>}
    </div>
  );
}

