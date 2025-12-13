// lib/analytics.ts - FIXED SESSION ID with proper API integration
import { BadmintonAction, FightingAction, GameAction, GameType, RacingAction, UnifiedPersonality, AIActionResponse } from './types';

const API_BASE = 'http://localhost:8000/api/v1';

// ✅ FIX: Use single consistent session ID format
let sessionIdCache: string | null = null;
let nextRetryAt = 0;

const BATCH_SIZE = 10;
let actionBuffer: GameAction[] = [];

export const getSessionId = (): string => {
  if (!sessionIdCache) {
    // Try to get from localStorage first
    const stored = localStorage.getItem('arena_session_id');
    
    // ✅ FIX: Clear old session IDs with wrong prefix
    if (stored && stored.startsWith('session-')) {
      console.log('🔄 Clearing old session ID format:', stored);
      localStorage.removeItem('arena_session_id');
    } else if (stored && stored.startsWith('arena-')) {
      sessionIdCache = stored;
      return sessionIdCache;
    }
    
    // Create new session with 'arena-' prefix
    sessionIdCache = `arena-${Date.now()}`;
    localStorage.setItem('arena_session_id', sessionIdCache);
    console.log('🆕 Created new session ID:', sessionIdCache);
  }
  return sessionIdCache;
};

// ✅ CRITICAL: Initialize session on app load
export const initializeSession = () => {
  // Force refresh session ID
  sessionIdCache = null;
  const sessionId = getSessionId();
  console.log('🎮 Session initialized:', sessionId);
  return sessionId;
};

export const resetSession = () => {
  sessionIdCache = null;
  localStorage.removeItem('arena_session_id');
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
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return res.json();
};

// Fetch real-time session analytics
export const fetchSessionAnalytics = async (sessionId: string) => {
  try {
    console.log('📊 Fetching analytics for:', sessionId);
    const response = await fetch(`${API_BASE}/analytics/${sessionId}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('✅ Analytics fetched:', data);
    return data;
  } catch (error) {
    console.error('❌ Failed to fetch analytics:', error);
    return null;
  }
};

// Fetch real-time personality profile
export const fetchPersonalityProfile = async (sessionId: string) => {
  try {
    console.log('🧠 Fetching personality for:', sessionId);
    const response = await fetch(`${API_BASE}/personality/${sessionId}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('✅ Personality fetched:', data);
    return data;
  } catch (error) {
    console.error('❌ Failed to fetch personality:', error);
    return null;
  }
};

// Universal analysis endpoint (for personality updates)
export const submitUniversalAnalysis = async (actionsData: any) => {
  try {
    const sessionId = getSessionId();
    const response = await fetch(`${API_BASE}/player/analyze-universal`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        ...actionsData
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ Analysis failed:', response.status, errorText);
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    console.log('✅ Universal analysis complete:', data);
    
    // Dispatch event for components to update
    window.dispatchEvent(new CustomEvent('analyticsUpdate', {
      detail: {
        session_stats: data.session_stats,
        personality: data.unified_personality
      }
    }));

    return data;
  } catch (error) {
    console.error('❌ Universal analysis error:', error);
    return null;
  }
};

export const flushActions = async () => {
  if (actionBuffer.length === 0) return null;
  if (Date.now() < nextRetryAt) return null;
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
    // Backoff to avoid spamming proxy errors when backend is offline
    nextRetryAt = Date.now() + 30000; // 30s cooldown
    console.warn('analyze-universal failed (backing off 30s)', e?.message || e);
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

// Request AI action from backend
export const requestAIAction = async (gameType: string, gameState: any) => {
  try {
    const sessionId = getSessionId();
    const response = await fetch(`${API_BASE}/ai/get-action/${gameType}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        current_game: gameType,
        game_state: gameState,
        cross_game_history: []
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('🤖 AI action received:', data);
    return data;
  } catch (error) {
    console.error('❌ AI action request failed:', error);
    return null;
  }
};