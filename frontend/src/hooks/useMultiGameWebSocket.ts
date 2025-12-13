// hooks/useMultiGameWebSocket.ts - FIXED VERSION
import { useState, useEffect, useRef, useCallback } from 'react';

interface WebSocketMessage {
  type: string;
  data?: any;
  message?: string;
  timestamp?: string;
}

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

export const useMultiGameWebSocket = (sessionId: string, enabled: boolean = true) => {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [gameState, setGameState] = useState<GameState>({});
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    if (!enabled || !sessionId) {
      console.log('❌ WebSocket disabled or no session ID');
      return;
    }

    try {
      // Close existing connection
      if (wsRef.current) {
        wsRef.current.close();
      }

      // FIX: Use correct WebSocket URL format
      const wsUrl = `ws://localhost:8000/ws/multi-game/${sessionId}`;
      console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('✅ WebSocket connected successfully');
        setConnected(true);
        setConnectionError(null);
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('📨 WebSocket message received:', message.type);
          setLastMessage(message);

          // Handle specific message types
          if (message.type === 'game_update' && message.data) {
            // Dispatch custom event for components to listen to
            window.dispatchEvent(new CustomEvent('gameUpdate', { 
              detail: message.data 
            }));
            
            // Update game state if available
            if (message.game_state) {
              const gameType = message.current_game || 'fighting';
              setGameState((prev) => ({ ...prev, [gameType]: message.game_state }));
            }
          } else if (message.type === 'game_state_update' && message.game) {
            setGameState((prev) => ({ ...prev, [message.game]: message.state }));
          } else if (message.type === 'session_status') {
            console.log('📊 Session status:', message);
            if (message.current_game && message.insights?.game_state) {
              setGameState((prev) => ({ ...prev, [message.current_game]: message.insights.game_state }));
            }
          }
        } catch (error) {
          console.error('❌ Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setConnectionError('WebSocket connection error');
        setConnected(false);
      };

      ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        setConnected(false);
        wsRef.current = null;

        // Auto-reconnect logic
        if (enabled && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000);
          console.log(`🔄 Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        }
      };
    } catch (error) {
      console.error('❌ Failed to create WebSocket:', error);
      setConnectionError('Failed to create WebSocket connection');
    }
  }, [sessionId, enabled]);

  // Connect on mount and when enabled changes
  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect, enabled]);

  const sendGameAction = useCallback((gameType: string, actionData: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const message = {
        type: 'player_action',
        action: {
          game_type: gameType,
          ...actionData,
          timestamp: Date.now()
        }
      };
      
      console.log('📤 Sending game action:', message.type, gameType);
      wsRef.current.send(JSON.stringify(message));
      return true;
    } else {
      console.warn('⚠️ WebSocket not connected, cannot send action');
      return false;
    }
  }, []);

  const sendGameStateUpdate = useCallback((game: string, state: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const message = {
        type: 'game_state_update',
        game,
        session_id: sessionId,
        state,
        timestamp: Date.now(),
      };
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, [sessionId]);

  const switchGame = useCallback((newGame: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const message = {
        type: 'game_switch',
        new_game: newGame
      };
      
      console.log('🎮 Switching game:', newGame);
      wsRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  const getStatus = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }));
      return true;
    }
    return false;
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
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
    reconnectAttempts: reconnectAttemptsRef.current,
    maxReconnectAttempts,
    socket: wsRef.current,
    disconnect,
    connect,
  };
};
