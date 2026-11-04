import { useEffect, useRef, useState, useCallback } from 'react';

const WS_URL = `ws://${window.location.host}/ws`;

const STATES = {
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  ERROR: 'error',
};

const MAX_BACKOFF = 30000;
const INITIAL_BACKOFF = 1000;

export default function useWebSocket() {
  const [state, setState] = useState(STATES.DISCONNECTED);
  const [lastMessage, setLastMessage] = useState(null);
  const [hardwareStats, setHardwareStats] = useState(null);
  const [jobUpdates, setJobUpdates] = useState(null);
  const [statusUpdates, setStatusUpdates] = useState(null);

  const wsRef = useRef(null);
  const backoffRef = useRef(INITIAL_BACKOFF);
  const reconnectTimerRef = useRef(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    // Clean up any existing connection
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.onmessage = null;
      if (wsRef.current.readyState < 2) {
        wsRef.current.close();
      }
    }

    setState(STATES.CONNECTING);

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setState(STATES.CONNECTED);
        backoffRef.current = INITIAL_BACKOFF;
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);

          // Route message by channel/type
          const channel = data.channel || data.type;
          if (channel === 'hardware') {
            setHardwareStats(data.payload || data.data || data);
          } else if (channel === 'jobs' || channel === 'job_update') {
            setJobUpdates(data.payload || data.data || data);
          } else if (channel === 'status') {
            setStatusUpdates(data.payload || data.data || data);
          }
        } catch {
          // Non-JSON message — ignore
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setState(STATES.DISCONNECTED);
        scheduleReconnect();
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
        setState(STATES.ERROR);
        // onclose will fire next and trigger reconnect
      };
    } catch {
      setState(STATES.ERROR);
      scheduleReconnect();
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (!mountedRef.current) return;
    clearTimeout(reconnectTimerRef.current);

    reconnectTimerRef.current = setTimeout(() => {
      if (mountedRef.current) {
        connect();
      }
    }, backoffRef.current);

    // Exponential backoff
    backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF);
  }, [connect]);

  const sendMessage = useCallback((data) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
      return true;
    }
    return false;
  }, []);

  const disconnect = useCallback(() => {
    clearTimeout(reconnectTimerRef.current);
    if (wsRef.current) {
      wsRef.current.onclose = null; // prevent reconnect
      wsRef.current.close();
    }
    setState(STATES.DISCONNECTED);
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [connect]);

  return {
    state,
    isConnected: state === STATES.CONNECTED,
    isConnecting: state === STATES.CONNECTING,
    lastMessage,
    hardwareStats,
    jobUpdates,
    statusUpdates,
    sendMessage,
    disconnect,
    reconnect: connect,
  };
}

export { STATES };
