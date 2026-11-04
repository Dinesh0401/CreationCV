import { useState, useCallback } from 'react';

/**
 * Lightweight API hook for backend calls.
 * Uses the Vite proxy so base URL is just '/api'.
 */
export default function useApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const request = useCallback(async (endpoint, options = {}) => {
    const { method = 'GET', body, retries = 1 } = options;
    const url = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;

    setLoading(true);
    setError(null);

    let lastError = null;

    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const fetchOptions = {
          method,
          headers: {
            'Content-Type': 'application/json',
          },
        };

        if (body) {
          fetchOptions.body = JSON.stringify(body);
        }

        const response = await fetch(url, fetchOptions);

        if (!response.ok) {
          const errorText = await response.text().catch(() => 'Unknown error');
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const data = await response.json();
        setLoading(false);
        return { data, error: null };
      } catch (err) {
        lastError = err;
        // If we have retries left and it's a network error, wait then retry
        if (attempt < retries && (err.name === 'TypeError' || err.message.includes('fetch'))) {
          await new Promise((r) => setTimeout(r, 1000));
          continue;
        }
      }
    }

    // All attempts failed
    const friendlyMessage = getFriendlyError(lastError);
    setError(friendlyMessage);
    setLoading(false);
    return { data: null, error: friendlyMessage };
  }, []);

  const get = useCallback((endpoint) => request(endpoint), [request]);

  const post = useCallback(
    (endpoint, body) => request(endpoint, { method: 'POST', body }),
    [request]
  );

  const clearError = useCallback(() => setError(null), []);

  return { get, post, request, loading, error, clearError };
}

function getFriendlyError(err) {
  if (!err) return 'An unknown error occurred.';

  const msg = err.message || '';

  if (msg.includes('Failed to fetch') || msg.includes('NetworkError') || msg.includes('fetch')) {
    return 'Cannot connect to the backend server. Make sure it is running on http://localhost:8000';
  }
  if (msg.includes('404')) {
    return 'The requested resource was not found.';
  }
  if (msg.includes('500')) {
    return 'The server encountered an internal error.';
  }
  if (msg.includes('403')) {
    return 'Access denied.';
  }

  return msg;
}
