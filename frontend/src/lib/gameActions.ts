export type SendPlayerAction = (action: unknown) => void;

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
      player_position: position,
      damage_dealt: damage,
      combo_count: combo,
      timestamp: Date.now(),
      context: {
        player_health: playerHealth,
        ai_health: aiHealth,
        distance_to_opponent: Math.abs(position[0]) * 2,
      }
    };
    sendPlayerAction(actionData);
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
      target: targetPos,
      power_level: powerLevel,
      rally_count: rallyCount,
      player_position: playerPosition,
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
    sendPlayerAction(actionData);
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
      player_position: position,
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
    sendPlayerAction(actionData);
  };

  return { handleFightingAction, handleBadmintonAction, handleRacingAction } as const;
};
