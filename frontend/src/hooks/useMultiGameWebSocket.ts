// frontend/src/hooks/useMultiGameWebSocket.ts - PORT FIX
import { useEffect, useRef, useState } from 'react';
import type { UnifiedPersonality, AIActionResponse, GameType } from '../lib/types';

export const useMultiGameWebSocket = (sessionId: string = 'test-session-123') => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  useEffect(() => {
    // TRY DIFFERENT PORTS UNTIL WE FIND THE RIGHT ONE
    const possiblePorts = [8000, 8001, 8002];
    let currentPortIndex = 0;
    
    const tryConnection = (port: number) => {
      const wsUrl = `ws://localhost:${port}/ws/multi-game/${sessionId}`;
      console.log(`🔗 Trying to connect to: ${wsUrl}`);
      
      const ws = new WebSocket(wsUrl);
      
      const connectionTimeout = setTimeout(() => {
        console.log(`⏰ Connection timeout for port ${port}`);
        ws.close();
        tryNextPort();
      }, 3000); // 3 second timeout
      
      ws.onopen = () => {
        clearTimeout(connectionTimeout);
        console.log(`✅ WebSocket Connected Successfully on port ${port}`);
        setConnected(true);
        setSocket(ws);
        setConnectionError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('📨 Received:', data);
          setLastMessage(data);
        } catch (e) {
          console.error('❌ Parse error:', e);
        }
      };

      ws.onclose = (event) => {
        clearTimeout(connectionTimeout);
        console.log(`🔌 WebSocket Disconnected from port ${port}. Code: ${event.code}`);
        setConnected(false);
        setSocket(null);
        
        if (event.code === 1006) { // Connection failed
          tryNextPort();
        }
      };

      ws.onerror = (error) => {
        clearTimeout(connectionTimeout);
        console.error(`❌ WebSocket Error on port ${port}:`, error);
        tryNextPort();
      };

      return ws;
    };

    const tryNextPort = () => {
      currentPortIndex++;
      if (currentPortIndex < possiblePorts.length) {
        console.log(`🔄 Trying next port: ${possiblePorts[currentPortIndex]}`);
        setTimeout(() => tryConnection(possiblePorts[currentPortIndex]), 1000);
      } else {
        setConnectionError('Failed to connect to backend on any port (8000, 8001, 8002). Make sure backend is running.');
      }
    };

    const ws = tryConnection(possiblePorts[currentPortIndex]);

    return () => {
      console.log('🧹 Cleaning up WebSocket');
      ws?.close();
    };
  }, [sessionId]);

  const sendGameAction = (gameType: string, actionData: any) => {
    if (!socket || !connected) {
      console.warn('⚠️ WebSocket not connected');
      return;
    }

    const message = {
      type: 'player_action',
      action: {
        session_id: sessionId,
        action_type: actionData.action_type || 'unknown',
        game_type: gameType,
        timestamp: Date.now() / 1000,
        success: actionData.success ?? true,
        ...actionData
      }
    };
    
    console.log('🎮 Sending action:', message);
    socket.send(JSON.stringify(message));
  };

  const switchGame = (newGame: GameType) => {
    if (!socket || !connected) {
      console.warn('⚠️ WebSocket not connected');
      return;
    }

    const message = {
      type: 'game_switch',
      new_game: newGame
    };
    
    console.log('🔄 Switching game:', message);
    socket.send(JSON.stringify(message));
  };

  const getStatus = () => {
    if (!socket || !connected) {
      console.warn('⚠️ WebSocket not connected');
      return;
    }

    socket.send(JSON.stringify({ type: 'get_status' }));
  };

  return {
    connected,
    lastMessage,
    connectionError,
    sendGameAction,
    switchGame,
    getStatus,
    socket
  };
};