import { useEffect, useRef, useState } from 'react';
import type { UnifiedPersonality, AIActionResponse, GameType } from '../lib/types';
import { getSessionId } from '../lib/analytics';

interface AnalysisUpdatePayload {
  updated_personality?: UnifiedPersonality;
  unified_personality?: UnifiedPersonality;
  ai_response?: AIActionResponse;
  game_state?: Record<string, unknown>;
}

interface IncomingMessageBase {
  type: string;
  [k: string]: unknown;
}

export const useMultiGameWebSocket = (sessionIdProp?: string) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [gameState, setGameState] = useState<Record<string, unknown>>({});
  const [personalityData, setPersonalityData] = useState<UnifiedPersonality | null>(null);
  const [aiResponse, setAiResponse] = useState<AIActionResponse | null>(null);
  const urlRef = useRef<string | null>(null);

  useEffect(() => {
    const sessionId = sessionIdProp || getSessionId();
    const url = `ws://localhost:8000/ws/multi-game/${sessionId}`;
    urlRef.current = url;
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      setSocket(ws);
    };

    ws.onmessage = (event) => {
      try {
        const data: IncomingMessageBase = JSON.parse(event.data);
        switch (data.type) {
          case 'connection_established':
            // No-op; connection state handled onopen
            break;
          case 'game_switched': {
            const state = (data as any).data?.game_state || (data as any).game_state || {};
            setGameState(state);
            break;
          }
          case 'analysis_update': {
            const payload: AnalysisUpdatePayload = (data as any).data || (data as any);
            if (payload.updated_personality || (payload as any).unified_personality) {
              setPersonalityData(payload.updated_personality || (payload as any).unified_personality || null);
            }
            if (payload.ai_response) setAiResponse(payload.ai_response);
            if (payload.game_state) setGameState(payload.game_state);
            break;
          }
          default:
            break;
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error('WebSocket message parse error', e);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setSocket(null);
    };

    return () => ws.close();
  }, [sessionIdProp]);

  const switchGame = (newGame: GameType) => {
    if (socket && connected) {
      socket.send(
        JSON.stringify({
          type: 'game_switch',
          new_game: newGame,
        })
      );
    }
  };

  const sendPlayerAction = (actionData: unknown) => {
    if (socket && connected) {
      socket.send(
        JSON.stringify({
          type: 'player_action',
          action: actionData,
        })
      );
    }
  };

  return { connected, switchGame, sendPlayerAction, gameState, personalityData, aiResponse } as const;
};
