import React, { useEffect, useRef } from 'react';
import { requestAIAction } from '@/lib/analytics';

type Props = {
  sessionId: string;
  enabled: boolean;
  gameState: any;
  onAIMove: (order: { target?: [number, number]; swing?: { dir:[number,number,number]; power:number; spin?:[number,number,number] } } ) => void;
};

export default function BadmintonAIController({ sessionId, enabled, gameState, onAIMove }: Props) {
  const cooling = useRef(false);

  useEffect(() => {
    if (!enabled || cooling.current) return;
    cooling.current = true;
    const delay = 300 + Math.random() * 600;
    const t = setTimeout(async () => {
      try {
        const res = await requestAIAction('badminton', { session_id: sessionId, ...gameState });
        // Basic mapping: if backend suggests 'attack' or high confidence, target shuttle predicted landing
        const sx = gameState.shuttle?.pos?.[0] ?? 0;
        const sz = gameState.shuttle?.pos?.[2] ?? 0;
        const vx = gameState.shuttle?.vel?.[0] ?? 0;
        const vz = gameState.shuttle?.vel?.[2] ?? 0;
        const tx: [number, number] = [sx + vx * 0.6, sz + vz * 0.6];
        const order = { target: tx } as any;
        if (res?.confidence && res.confidence > 0.5) {
          // prepare a swing back toward player side
          const dir: [number,number,number] = [ -1, 0.6, (Math.random()-0.5)*0.6 ];
          order.swing = { dir, power: 0.8 };
        }
        onAIMove(order);
      } catch {
        // Fallback: still move toward estimated landing
        const sx = gameState.shuttle?.pos?.[0] ?? 0;
        const sz = gameState.shuttle?.pos?.[2] ?? 0;
        const vx = gameState.shuttle?.vel?.[0] ?? 0;
        const vz = gameState.shuttle?.vel?.[2] ?? 0;
        onAIMove({ target: [sx + vx * 0.6, sz + vz * 0.6] });
      } finally {
        cooling.current = false;
      }
    }, delay);
    return () => clearTimeout(t);
  }, [enabled, gameState, sessionId, onAIMove]);

  return null;
}
