import React from 'react';
import { GameType } from '@/lib/types';

interface FightingScoreProps {
  playerHealth: number; // 0..100
  aiHealth: number; // 0..100
  rounds: [number, number];
}

interface BadmintonScoreProps {
  score: [number, number];
}

interface RacingScoreProps {
  lap: number;
  totalLaps: number;
  position: number;
  totalRacers: number;
}

type ScoreProps = (
  | ({ game: 'fighting' } & FightingScoreProps)
  | ({ game: 'badminton' } & BadmintonScoreProps)
  | ({ game: 'racing' } & RacingScoreProps)
) & { compact?: boolean };

const Bar = ({ value, color }: { value: number; color: string }) => (
  <div className="h-3 w-40 bg-black/40 rounded">
    <div
      className="h-3 rounded transition-all"
      style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: color }}
    />
  </div>
);

const ScoreBar: React.FC<ScoreProps> = (props) => {
  const base = 'pointer-events-auto select-none scorebar shadow-lg border border-white/10 rounded-xl px-4 py-2';

  if (props.game === 'fighting') {
    const { playerHealth, aiHealth, rounds } = props;
    return (
      <div className={`${base} bg-gradient-to-b from-[#1b1b28] to-[#10101a] text-white flex items-center gap-4`}> 
        <div className="font-gaming tracking-widest text-sm opacity-90">Player vs AI</div>
        <Bar value={playerHealth} color="#2dd4bf" />
        <div className="px-2 text-lg font-bold">{rounds[0]} : {rounds[1]}</div>
        <Bar value={aiHealth} color="#ef4444" />
      </div>
    );
  }
  if (props.game === 'badminton') {
    const { score } = props;
    return (
      <div className={`${base} bg-gradient-to-b from-[#122216] to-[#0d1710] text-white flex items-center gap-4`}> 
        <div className="font-gaming tracking-widest text-sm opacity-90">Player vs AI</div>
        <div className="text-xl font-extrabold">{score[0]} - {score[1]}</div>
        <div className="text-xs opacity-70">GAME SCORE</div>
      </div>
    );
  }
  const { lap, totalLaps, position, totalRacers } = props;
  return (
    <div className={`${base} bg-gradient-to-b from-[#1a1a1a] to-[#0f0f10] text-white flex items-center gap-4`}> 
      <div className="font-gaming tracking-widest text-sm opacity-90">Player vs AI</div>
      <div className="text-sm">Lap</div>
      <div className="text-xl font-extrabold">{lap}/{totalLaps}</div>
      <div className="h-5 w-px bg-white/20" />
      <div className="text-sm">Pos</div>
      <div className="text-xl font-extrabold">{position}/{totalRacers}</div>
    </div>
  );
};

export default ScoreBar;
