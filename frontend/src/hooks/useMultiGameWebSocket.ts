// Enhanced multi-game WebSocket hook with state sync and robust reconnection
import { useCallback, useEffect, useRef, useState } from 'react';
import type { GameType } from '../lib/types';

export interface GameAction {
  action_type: string;
  timestamp?: number;
  game_data?: any;
  [key: string]: any;
}

export interface GameState {
  fighting?: {
    player_health: number;
    ai_health: number;
    rounds: [number, number];
    combo_count: number;
  };
  badminton?: {
    score: [number, number];
    rally_count: number;
    game_point: boolean;
  };
  racing?: {
    lap: number;
    position: number;
    lap_times: number[];
    total_distance: number;
  };
}

export const useMultiGameWebSocket = (sessionId: string = 'test-session-123', enabled: boolean = true) => {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [gameState, setGameState] = useState<GameState>({});
  const ws = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const possiblePorts = [8000, 8001, 8002];

  const cleanup = () => {
    if (ws.current) {
      try { ws.current.close(); } catch {}
      ws.current = null;
    }
    setConnected(false);
  };

  const connect = useCallback(() => {
    if (!enabled) return;
    if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) return;

    let portIdx = 0;

    const tryConnect = (port: number) => {
      try {
        const url = `ws://localhost:${port}/ws/multi-game/${sessionId}`;
        const sock = new WebSocket(url);
        ws.current = sock;

        const timeout = setTimeout(() => {
          try { sock.close(); } catch {}
        }, 4000);

        sock.onopen = () => {
          clearTimeout(timeout);
          setConnected(true);
          setConnectionError(null);
          reconnectAttempts.current = 0;
          // Notify backend session start
          sock.send(JSON.stringify({
            type: 'connection_established',
            session_id: sessionId,
            timestamp: Date.now(),
          }));
        };

        sock.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data);
            setLastMessage(msg);
            if (msg?.type === 'game_state_update' && msg?.game) {
              setGameState((prev) => ({ ...prev, [msg.game]: msg.state }));
            }
          } catch {}
        };

        sock.onerror = () => {
          setConnectionError('Connection error occurred');
        };

        sock.onclose = () => {
          setConnected(false);
          if (!enabled) return;
          if (reconnectAttempts.current < maxReconnectAttempts) {
            reconnectAttempts.current += 1;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
            setTimeout(() => {
              portIdx = (portIdx + 1) % possiblePorts.length;
              tryConnect(possiblePorts[portIdx]);
            }, delay);
          } else {
            setConnectionError('Max reconnection attempts reached');
          }
        };
      } catch (e) {
        // Try next port
        portIdx = (portIdx + 1) % possiblePorts.length;
        tryConnect(possiblePorts[portIdx]);
      }
    };

    tryConnect(possiblePorts[portIdx]);
  }, [enabled, sessionId]);

  useEffect(() => {
    if (enabled) connect();
    return () => cleanup();
  }, [connect, enabled]);

  const sendRaw = (payload: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(payload));
      return true;
    }
    return false;
  };

  const sendGameAction = useCallback((game: string, action: GameAction) => {
    return sendRaw({
      type: 'game_action',
      game,
      session_id: sessionId,
      ...action,
      timestamp: action.timestamp ?? Date.now(),
    });
  }, [sessionId]);

  const sendGameStateUpdate = useCallback((game: string, state: any) => {
    return sendRaw({
      type: 'game_state_update',
      game,
      session_id: sessionId,
      state,
      timestamp: Date.now(),
    });
  }, [sessionId]);

  const switchGame = useCallback((newGame: GameType) => {
    return sendRaw({ type: 'switch_game', new_game: newGame, session_id: sessionId, timestamp: Date.now() });
  }, [sessionId]);

  const getStatus = useCallback(() => {
    return sendRaw({ type: 'get_status', session_id: sessionId, timestamp: Date.now() });
  }, [sessionId]);

  const disconnect = useCallback(() => {
    cleanup();
  }, []);

  return {
    connected,
    lastMessage,
    connectionError,
    gameState,
    sendGameAction,
    sendGameStateUpdate,
    switchGame,
    getStatus,
    reconnectAttempts: reconnectAttempts.current,
    maxReconnectAttempts,
    socket: ws.current,
    disconnect,
    connect,
  };
};
