// frontend/src/hooks/useMultiGameWebSocket.ts - PORT FIX
import { useEffect, useRef, useState } from 'react';
import type { UnifiedPersonality, AIActionResponse, GameType } from '../lib/types';

export const useMultiGameWebSocket = (sessionId: string = 'test-session-123', enabled: boolean = true) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const openConnection = () => {
    // TRY DIFFERENT PORTS UNTIL WE FIND THE RIGHT ONE
    const possiblePorts = [8000, 8001, 8002];
    let currentPortIndex = 0;

    const tryNextPort = () => {
      currentPortIndex++;
      if (currentPortIndex < possiblePorts.length) {
        setTimeout(() => tryConnection(possiblePorts[currentPortIndex]), 800);
      } else {
        setConnectionError('Failed to connect to backend on any port (8000, 8001, 8002). Make sure backend is running.');
      }
    };

    const tryConnection = (port: number) => {
      const safePort = (typeof port === 'number' && isFinite(port)) ? port : 8000;
      let ws: WebSocket | null = null;
      try {
        const wsUrl = `ws://localhost:${safePort}/ws/multi-game/${sessionId}`;
        ws = new WebSocket(wsUrl);
      } catch (e) {
        tryNextPort();
        return;
      }
      if (!ws) { tryNextPort(); return; }
      wsRef.current = ws;

      const connectionTimeout = setTimeout(() => {
        ws.close();
        tryNextPort();
      }, 3000);

      ws.onopen = () => {
        clearTimeout(connectionTimeout);
        setConnected(true);
        setSocket(ws);
        setConnectionError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } catch {}
      };

      ws.onclose = (event) => {
        clearTimeout(connectionTimeout);
        setConnected(false);
        setSocket(null);
        if (event.code === 1006 && enabled) {
          tryNextPort();
        }
      };

      ws.onerror = () => {
        clearTimeout(connectionTimeout);
        tryNextPort();
      };
    };

    tryConnection(possiblePorts[currentPortIndex]);
  };

  useEffect(() => {
    if (!enabled) {
      if (wsRef.current) wsRef.current.close();
      setConnected(false);
      setSocket(null);
      return;
    }
    openConnection();
    return () => {
      wsRef.current?.close();
    };
  }, [sessionId, enabled]);

  const sendGameAction = (gameType: string, actionData: any) => {
    if (!socket || !connected) return;
    const message = {
      type: 'player_action',
      action: {
        session_id: sessionId,
        action_type: actionData.action_type || 'unknown',
        game_type: gameType,
        timestamp: Date.now() / 1000,
        success: actionData.success ?? true,
        ...actionData,
      },
    };
    socket.send(JSON.stringify(message));
  };

  const switchGame = (newGame: GameType) => {
    if (!socket || !connected) return;
    socket.send(JSON.stringify({ type: 'game_switch', new_game: newGame }));
  };

  const getStatus = () => {
    if (!socket || !connected) return;
    socket.send(JSON.stringify({ type: 'get_status' }));
  };

  const disconnect = () => {
    wsRef.current?.close();
  };

  const connect = () => {
    if (!connected && enabled) openConnection();
  };

  return { connected, lastMessage, connectionError, sendGameAction, switchGame, getStatus, socket, disconnect, connect };
};
