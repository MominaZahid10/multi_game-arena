// lib/gameActions.ts - FIXED CORS VERSION

// ✅ Use localhost instead of 127.0.0.1 to match frontend origin
const API_BASE = 'http://localhost:8000/api/v1';

import { getSessionId } from './analytics';
export type SendPlayerAction = (action: unknown) => void;

// Fighting game action submission  
export const postFightingAction = async (actionData: any) => {
  const sessionId = getSessionId();
  const url = `${API_BASE}/games/fighting/action?session_id=${sessionId}`;
  
  // Reduced logging for performance
  
  try {
    const fetchPromise = fetch(url, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        // ✅ Add explicit origin header
        'Origin': 'http://localhost:8080'
      },
      body: JSON.stringify({ action_data: actionData }),
      mode: 'cors', // ✅ Explicit CORS mode
      credentials: 'omit' // ✅ Don't send cookies (faster)
    });
    
    // ✅ Increased timeout to 15s for ML model inference
    const response = await Promise.race([
      fetchPromise,
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Request timeout after 15s')), 15000)
      )
    ]) as Response;

    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ Fighting action failed:', response.status, response.statusText, errorText);
      return null;
    }

    const data = await response.json();
    
    // ✅ Dispatch analytics update event
    if (data.session_stats || data.personality) {
      window.dispatchEvent(new CustomEvent('analyticsUpdate', {
        detail: {
          session_stats: data.session_stats,
          personality: data.personality
        }
      }));
    }

    // ✅ Dispatch AI action event (CRITICAL!)
    if (data.ai_action) {
      const event = new CustomEvent('aiActionUpdate', {
        detail: {
          gameType: 'fighting',
          ai_action: data.ai_action
        },
        bubbles: true,
        cancelable: true
      });
      
      window.dispatchEvent(event);
    }

    return data;
  } catch (error) {
    console.error('❌ Fighting action EXCEPTION:', error);
    return null;
  }
};

// Badminton game action submission
export const postBadmintonAction = async (actionData: any) => {
  try {
    const sessionId = getSessionId();
    
    console.log('🏸 Submitting badminton action:', actionData);
    
    const response = await fetch(`${API_BASE}/games/badminton/action?session_id=${sessionId}`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Origin': 'http://localhost:8080'
      },
      body: JSON.stringify({ action_data: actionData }),
      mode: 'cors'
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ Badminton action failed:', response.status, errorText);
      return null;
    }

    const data = await response.json();
    console.log('✅ Badminton action response:', data);
    
    // Dispatch analytics update event
    if (data.session_stats || data.personality) {
      window.dispatchEvent(new CustomEvent('analyticsUpdate', {
        detail: {
          session_stats: data.session_stats,
          personality: data.personality
        }
      }));
    }

    // ✅ CRITICAL: Dispatch AI action event for game to use
    if (data.ai_action) {
      console.log('🤖 Dispatching AI action:', data.ai_action);
      window.dispatchEvent(new CustomEvent('aiActionUpdate', {
        detail: {
          gameType: 'badminton',
          ai_action: data.ai_action
        }
      }));
    }

    return data;
  } catch (error) {
    console.error('❌ Badminton action error:', error);
    return null;
  }
};

// Racing game action submission
export const postRacingAction = async (actionData: any) => {
  try {
    const sessionId = getSessionId();
    
    console.log('🏎️ Submitting racing action:', actionData);
    
    const response = await fetch(`${API_BASE}/games/racing/action?session_id=${sessionId}`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Origin': 'http://localhost:8080'
      },
      body: JSON.stringify({ action_data: actionData }),
      mode: 'cors'
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ Racing action failed:', response.status, errorText);
      return null;
    }

    const data = await response.json();
    console.log('✅ Racing action response:', data);
    
    // Dispatch analytics update event
    if (data.session_stats || data.personality) {
      window.dispatchEvent(new CustomEvent('analyticsUpdate', {
        detail: {
          session_stats: data.session_stats,
          personality: data.personality
        }
      }));
    }

    // ✅ CRITICAL: Dispatch AI action event for game to use
    if (data.ai_action) {
      console.log('🤖 Dispatching AI action:', data.ai_action);
      window.dispatchEvent(new CustomEvent('aiActionUpdate', {
        detail: {
          gameType: 'racing',
          ai_action: data.ai_action
        }
      }));
    }

    return data;
  } catch (error) {
    console.error('❌ Racing action error:', error);
    return null;
  }
};


export const createGameActionHandlers = (sendPlayerAction: SendPlayerAction) => {
  const handleFightingAction = (
    moveType: 'attack' | 'block' | 'move' | 'combo',
    success: boolean,
    position: [number, number],
    damage: number = 0,
    combo: number = 0,
    playerHealth: number = 100,
    aiHealth: number = 100
  ) => {
    const actionData = {
      game_type: 'fighting' as const,
      action_type: moveType,
      move_type: moveType,
      success,
      position: { x: position[0], y: position[1] },
      damage_dealt: damage,
      combo_count: combo,
      timestamp: Date.now(),
      context: {
        player_health: playerHealth,
        ai_health: aiHealth,
        distance_to_opponent: Math.abs(position[0]) * 2,
      }
    };
    postFightingAction(actionData);
  };

  const handleBadmintonAction = (
    shotType: 'clear' | 'drop' | 'smash' | 'net' | 'drive',
    targetPos: [number, number],
    powerLevel: number,
    rallyCount: number,
    playerPosition: [number, number],
    playerScore: number = 0,
    aiScore: number = 0,
    success: boolean = true
  ) => {
    const actionData = {
      game_type: 'badminton' as const,
      action_type: shotType,
      shot_type: shotType,
      shuttlecock_target: targetPos,
      power_level: powerLevel,
      rally_position: rallyCount, // Corrected key to match Pydantic model
      court_position: playerPosition,
      success,
      timestamp: Date.now(),
      context: {
        score_player: playerScore,
        score_ai: aiScore,
        rally_count: rallyCount,
        court_position: { x: playerPosition[0], y: playerPosition[1] },
        shuttlecock_target: { x: targetPos[0], y: targetPos[1] }
      }
    };
    postBadmintonAction(actionData);
  };

  const handleRacingAction = (
    actionType: 'accelerate' | 'brake' | 'steer' | 'overtake',
    speed: number,
    position: [number, number],
    overtaking: boolean = false,
    crashed: boolean = false,
    lap: number = 1,
    racePosition: number = 2,
    success: boolean = true
  ) => {
    const actionData = {
      game_type: 'racing' as const,
      action_type: actionType,
      speed,
      position_on_track: { x: position[0], y: position[1] },
      overtaking_attempt: overtaking,
      crash_occurred: crashed,
      success,
      timestamp: Date.now(),
      context: {
        lap,
        position: racePosition,
        speed,
        track_position: { x: position[0], y: position[1] },
        is_overtaking: overtaking,
        has_crashed: crashed
      }
    };
    postRacingAction(actionData);
  };

  return { handleFightingAction, handleBadmintonAction, handleRacingAction } as const;
};