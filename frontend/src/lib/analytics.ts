import { BadmintonAction, FightingAction, GameAction, GameType, RacingAction, UnifiedPersonality, AIActionResponse } from './types';

const API_BASE = '/api/v1';

const BATCH_SIZE = 10;
let actionBuffer: GameAction[] = [];
let currentSessionId: string | null = null;

export const getSessionId = () => {
  if (!currentSessionId) {
    currentSessionId = `sess_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  }
  return currentSessionId;
};

export const resetSession = () => {
  currentSessionId = null;
  actionBuffer = [];
};

export const isCriticalAction = (action: Partial<GameAction>) => {
  // Fighting combo, badminton smash, racing overtake are critical
  // Use in a type-safe yet permissive way
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const anyAction: any = action;
  return (
    anyAction?.action_type === 'combo' ||
    anyAction?.shot_type === 'smash' ||
    anyAction?.overtaking_attempt === true
  );
};

const groupActionsForUniversal = (actions: GameAction[]) => {
  const fighting_actions: FightingAction[] = [];
  const badminton_actions: BadmintonAction[] = [];
  const racing_actions: RacingAction[] = [];
  for (const a of actions) {
    if (a.game_type === 'fighting') fighting_actions.push(a);
    else if (a.game_type === 'badminton') badminton_actions.push(a);
    else racing_actions.push(a);
  }
  return { fighting_actions, badminton_actions, racing_actions };
};

const postJSON = async <T>(url: string, body: unknown): Promise<T> => {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
};

export const flushActions = async () => {
  if (actionBuffer.length === 0) return null;
  const toSend = [...actionBuffer];
  actionBuffer = [];
  const grouped = groupActionsForUniversal(toSend);
  return postJSON<{ unified_personality: UnifiedPersonality; cross_game_insights: unknown; session_stats: unknown }>(
    `${API_BASE}/player/analyze-universal`,
    {
      session_id: getSessionId(),
      ...grouped,
    }
  ).catch((e) => {
    // Swallow errors to avoid breaking UX; could log to Sentry when connected
    console.error('analyze-universal failed', e);
    return null;
  });
};

export const addAction = async (action: GameAction) => {
  actionBuffer.push(action);
  if (actionBuffer.length >= BATCH_SIZE || isCriticalAction(action)) {
    return flushActions();
  }
  return null;
};

export const requestAIAction = async (game: GameType, game_state: Record<string, unknown>) => {
  return postJSON<AIActionResponse>(`${API_BASE}/ai/get-action/${game}`,
    {
      session_id: getSessionId(),
      current_game: game,
      game_state,
    }
  );
};
