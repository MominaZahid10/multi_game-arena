export type GameType = 'fighting' | 'badminton' | 'racing';

export interface UnifiedPersonality {
  aggression_level: number;
  risk_tolerance: number;
  analytical_thinking: number;
  patience_level: number;
  precision_focus: number;
  competitive_drive: number;
  strategic_thinking: number;
  adaptability: number;
  confidence_score: number;
  games_played: GameType[];
}

export interface FightingAction {
  game_type: 'fighting';
  action_type: 'attack' | 'block' | 'move' | 'combo';
  timestamp: number;
  success: boolean;
  move_type: 'attack' | 'block' | 'move';
  position: [number, number];
  damage_dealt?: number;
  combo_count: number;
  context: {
    player_health: number;
    ai_health: number;
    distance_to_opponent: number;
  };
}

export interface BadmintonAction {
  game_type: 'badminton';
  action_type: 'shot' | 'move' | 'serve';
  timestamp: number;
  success: boolean;
  shot_type: 'clear' | 'drop' | 'smash' | 'net' | 'drive';
  court_position: [number, number];
  shuttlecock_target: [number, number];
  power_level: number; // 0..1
  rally_position: number;
  context: {
    rally_count: number;
    court_side: 'left' | 'right';
    game_score: [number, number];
  };
}

export interface RacingAction {
  game_type: 'racing';
  action_type: 'accelerate' | 'brake' | 'steer' | 'overtake';
  timestamp: number;
  success: boolean;
  speed: number;
  position_on_track: [number, number];
  overtaking_attempt: boolean;
  crash_occurred: boolean;
  context: {
    lap_number: number;
    position_in_race: number;
    distance_to_finish: number;
  };
}

export type GameAction = FightingAction | BadmintonAction | RacingAction;

export interface AIActionResponse {
  current_game_action: string;
  confidence: number;
  strategy: string;
  cross_game_reasoning: string;
  personality_insights: Record<string, string>;
}

export interface AIInsight {
  type: 'strategy_change' | 'personality_update' | 'cross_game_pattern';
  message: string;
  confidence: number;
  timestamp: number;
}
