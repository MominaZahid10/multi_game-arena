import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface BadmintonPlayerProps {
  position: [number, number, number];
  color: string;
  isPlayer?: boolean;
  paused?: boolean;
  followTarget?: [number, number, number];
  followVel?: [number, number, number];
  aiOrder?: any;
  onPlayerHit?: (dir: [number, number, number], power: number, spin?: [number, number, number]) => void;
  onPositionChange?: (pos: [number, number, number]) => void;
}

function safeArray3(val: any, fallback: [number, number, number] = [0,0,0], label = ''): [number, number, number] {
  if (Array.isArray(val) && val.length === 3 && val.every(v => typeof v === 'number' && !isNaN(v))) return val as [number, number, number];
  if (val && typeof val === 'object') {
    const x = typeof val.x === 'number' ? val.x : fallback[0];
    const y = typeof val.y === 'number' ? val.y : fallback[1];
    const z = typeof val.z === 'number' ? val.z : fallback[2];
    return [x, y, z];
  }
  if (label) console.error('Invalid 3D array for', label, val, 'using fallback', fallback);
  return fallback;
}

const BadmintonPlayer: React.FC<BadmintonPlayerProps> = ({
  position = [0, 0, 0],
  color = '#22D3EE',
  isPlayer = false,
  paused = false,
  followTarget,
  followVel,
  aiOrder,
  onPlayerHit,
  onPositionChange,
}) => {
  const meshRef = useRef<THREE.Mesh>(null);

  // Defensive normalization
  const normPosition = safeArray3(position, [0,0,0], 'BadmintonPlayer.position');
  const normFollowTarget = followTarget ? safeArray3(followTarget, [0,0,0], 'BadmintonPlayer.followTarget') : undefined;
  const normFollowVel = followVel ? safeArray3(followVel, [0,0,0], 'BadmintonPlayer.followVel') : undefined;

  // Simple movement logic for demonstration
  useFrame(() => {
    if (meshRef.current && onPositionChange) {
      const pos = meshRef.current.position;
      onPositionChange([pos.x, pos.y, pos.z]);
    }
  });

  try {
    if (import.meta.env.DEV) console.debug('BadmintonPlayer props:', JSON.stringify({ normPosition, color, isPlayer, paused, normFollowTarget, normFollowVel, aiOrder }));
  } catch (e) {
    if (import.meta.env.DEV) console.debug('BadmintonPlayer props (stringify failed):', { normPosition, color, isPlayer, paused, normFollowTarget, normFollowVel, aiOrder }, e);
  }
  return (
    <mesh ref={meshRef} position={normPosition} castShadow receiveShadow>
      <sphereGeometry args={[0.25, 32, 32]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
};

export default BadmintonPlayer;
