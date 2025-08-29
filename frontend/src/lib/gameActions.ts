export type SendPlayerAction = (action: unknown) => void;

export const createGameActionHandlers = (sendPlayerAction: SendPlayerAction) => {
  const handleFightingAction = (
    moveType: 'attack' | 'block' | 'move' | 'combo',
    success: boolean,
    position: [number, number],
    damage: number = 0,
    combo: number = 0
  ) => {
    const actionData = {
      game_type: 'fighting' as const,
      action_type: moveType,
      move_type: moveType,
      success,
      player_position: position,
      damage,
      combo,
      timestamp: Date.now() / 1000,
    };
    sendPlayerAction(actionData);
  };

  const handleBadmintonAction = (
    shotType: 'clear' | 'drop' | 'smash' | 'net' | 'drive',
    targetPos: [number, number],
    powerLevel: number,
    rallyCount: number,
    playerPosition: [number, number]
  ) => {
    const actionData = {
      game_type: 'badminton' as const,
      action_type: shotType,
      shot_type: shotType,
      target: targetPos,
      power_level: powerLevel,
      rally_count: rallyCount,
      player_position: playerPosition,
      timestamp: Date.now() / 1000,
    };
    sendPlayerAction(actionData);
  };

  const handleRacingAction = (
    actionType: 'accelerate' | 'brake' | 'steer' | 'overtake',
    speed: number,
    position: [number, number],
    overtaking: boolean = false,
    crashed: boolean = false
  ) => {
    const actionData = {
      game_type: 'racing' as const,
      action_type: actionType,
      speed,
      player_position: position,
      overtaking_attempt: overtaking,
      crash_occurred: crashed,
      timestamp: Date.now() / 1000,
    };
    sendPlayerAction(actionData);
  };

  return { handleFightingAction, handleBadmintonAction, handleRacingAction } as const;
};
