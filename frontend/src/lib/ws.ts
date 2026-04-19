import { UnifiedPersonality, AIActionResponse } from './types';
import { getSessionId } from './analytics';

export type WSMessage =
  | { type: 'analysis_update'; unified_personality: UnifiedPersonality; ai_response: AIActionResponse }
  | { type: 'game_switched'; new_game: 'fighting' | 'badminton' | 'racing' }
  | { type: 'session_status'; status: string; [k: string]: unknown };

/** Returns the WebSocket base URL (no trailing slash, no path). */
export const getWsBase = (): string => {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL as string;
  if (import.meta.env.VITE_API_URL) {
    return (import.meta.env.VITE_API_URL as string)
      .replace(/^http/, 'ws')
      .replace(/\/api\/v1\/?$/, '');
  }
  return 'ws://localhost:8000';
};

export const connectMultiGameWS = (onMessage: (msg: WSMessage) => void, sessionId?: string) => {
  const sid = sessionId || getSessionId();
  const WS_URL = import.meta.env.VITE_WS_URL || `ws://localhost:8000/ws/multi-game/${sid}`;
  const ws = new WebSocket(WS_URL);
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
