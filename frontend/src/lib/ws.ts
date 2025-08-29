import { UnifiedPersonality, AIActionResponse } from './types';
import { getSessionId } from './analytics';

export type WSMessage =
  | { type: 'analysis_update'; unified_personality: UnifiedPersonality; ai_response: AIActionResponse }
  | { type: 'game_switched'; new_game: 'fighting' | 'badminton' | 'racing' }
  | { type: 'session_status'; status: string; [k: string]: unknown };

export const connectMultiGameWS = (onMessage: (msg: WSMessage) => void, sessionId?: string) => {
  const sid = sessionId || getSessionId();
  const ws = new WebSocket(`ws://localhost:8000/ws/multi-game/${sid}`);
  ws.onmessage = (event) => {
    try {
      const data: WSMessage = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error('WS parse error', e);
    }
  };
  return ws;
};

export const sendActionOverWS = (ws: WebSocket | null, action: unknown) => {
  try {
    ws?.send(JSON.stringify({ type: 'player_action', action }));
  } catch {}
};
