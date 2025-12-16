// src/components/GameArena.tsx

// Utility to ensure position is always [x, y, z] array
function toPositionArray(pos: any, fallback: [number, number, number] = [0, 0, 0]): [number, number, number] {
  let arr: [number, number, number];
  if (Array.isArray(pos) && pos.length === 3 && pos.every((v) => typeof v === 'number' && !isNaN(v))) {
    arr = pos as [number, number, number];
  } else if (pos && typeof pos === 'object') {
    const x = typeof pos.x === 'number' ? pos.x : fallback[0];
    const y = typeof pos.y === 'number' ? pos.y : fallback[1];
    const z = typeof pos.z === 'number' ? pos.z : fallback[2];
    arr = [x, y, z];
  } else {
    arr = fallback;
  }
  if (!Array.isArray(arr) || arr.length !== 3 || arr.some((v) => typeof v !== 'number' || isNaN(v))) {
    console.error('toPositionArray: Invalid input, returning fallback', pos, arr);
    arr = fallback;
  } else {
    if (import.meta.env.DEV) console.debug('toPositionArray:', pos, '=>', arr);
  }
  return arr;
}
import React, { useEffect, useRef, useState, Suspense, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Box, Sphere, Plane, Cylinder } from '@react-three/drei';
import BadmintonAIController from './BadmintonAIController';
import { Physics, RigidBody, CuboidCollider } from '@react-three/rapier';
import Player from './Player';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import Shuttlecock from './Shuttlecock';
import ScoreBar from './ScoreBar';
import { useMultiGameWebSocket } from '@/hooks/useMultiGameWebSocket';
import { fetchSessionAnalytics, fetchPersonalityProfile, getSessionId } from '../lib/analytics';
import { postFightingAction, postBadmintonAction, postRacingAction } from '../lib/gameActions';
import AnalyticsDashboard from './AnalyticsDashboard';


interface GameArenaProps {
  gameType: 'fighting' | 'badminton' | 'racing';
  onGameChange: (game: 'fighting' | 'badminton' | 'racing') => void;
  showAnalytics: boolean;
  onToggleAnalytics: () => void;
}

// =======================================================
// CORE COMPONENT DEFINITIONS (Moved to module scope to fix ReferenceError)
// =======================================================

type FightingAI = 'combo_attack' | 'defensive_counter' | 'rush_forward' | 'punch' | 'kick' | 'block' | null;

const FighterCharacter = ({ 
  position, 
  color, 
  isPlayer = false, 
  initialFacing = 1, 
  engaged = false, 
  paused = false, 
  opponentPosition, 
  onPositionChange, 
  aiCommand = null,
  aiTargetPosition = null,
  onPlayerAttack,
  playerCurrentHealth,
  aiCurrentHealth,
  postAction,
  onBlockStateChange,
}: any) => {
  const meshRef = useRef<THREE.Group>(null);
  
  // Body Parts Refs for Animation
  const torsoRef = useRef<THREE.Group>(null);
  const headRef = useRef<THREE.Group>(null);
  const leftArmGroup = useRef<THREE.Group>(null);
  const rightArmGroup = useRef<THREE.Group>(null);
  const leftLegGroup = useRef<THREE.Group>(null);
  const rightLegGroup = useRef<THREE.Group>(null);
  
  // State
  const [position2D, setPosition2D] = useState(position);
  const [isAttacking, setIsAttacking] = useState(false);
  const [isBlocking, setIsBlocking] = useState(false);
  const [isMoving, setIsMoving] = useState(false);
  const [isGrounded, setIsGrounded] = useState(true);
  
  // Physics & Logic Refs
  const physicsRef = useRef({
    position: [...position] as [number, number, number],
    velocity: { x: 0, y: 0, z: 0 },
    jumpCooldown: 0,
    facing: initialFacing
  });
  const inputRef = useRef({ x: 0, z: 0 });
  const attackCooldownRef = useRef(0);
  const lastAiActionTime = useRef(0);
  
  // Constants
  const MOVE_SPEED = 4.5;
  const JUMP_FORCE = 12;
  const GRAVITY = -30;
  const ATTACK_DURATION = 400;

  // --- PHYSICS LOOP ---
  useFrame((state, delta) => {
    if (paused) return;
    const phys = physicsRef.current;

    // 1. Movement Physics
    if (inputRef.current.x !== 0 || inputRef.current.z !== 0) {
      phys.velocity.x += inputRef.current.x * 20 * delta;
      phys.velocity.z += inputRef.current.z * 20 * delta;
      setIsMoving(true);
    } else {
      phys.velocity.x *= 0.85;
      phys.velocity.z *= 0.85;
      setIsMoving(Math.abs(phys.velocity.x) > 0.1 || Math.abs(phys.velocity.z) > 0.1);
    }

    // Cap speed
    const speed = Math.hypot(phys.velocity.x, phys.velocity.z);
    if (speed > MOVE_SPEED) {
      phys.velocity.x = (phys.velocity.x / speed) * MOVE_SPEED;
      phys.velocity.z = (phys.velocity.z / speed) * MOVE_SPEED;
    }

    // 1.5. Body Collision (Prevent walking through opponent)
    if (opponentPosition) {
      const dx = phys.position[0] - opponentPosition[0];
      const dz = phys.position[2] - opponentPosition[2];
      const dist = Math.hypot(dx, dz);
      const minDist = 1.0; // The "size" of the player collider

      if (dist < minDist) {
        // Calculate push direction (away from opponent)
        const angle = Math.atan2(dz, dx);
        const pushForce = 20 * delta; // Strong push to prevent overlapping

        phys.velocity.x += Math.cos(angle) * pushForce;
        phys.velocity.z += Math.sin(angle) * pushForce;
      }
    }

    // Gravity & Jumping (floor at y = -2)
    if (phys.position[1] > -2) {
      phys.velocity.y += GRAVITY * delta;
      setIsGrounded(false);
    } else {
      phys.velocity.y = Math.max(0, phys.velocity.y);
      phys.position[1] = -2;
      setIsGrounded(true);
      if (phys.jumpCooldown > 0) phys.jumpCooldown -= delta * 1000;
    }

    // Apply Velocity
    phys.position[0] += phys.velocity.x * delta;
    phys.position[1] += phys.velocity.y * delta;
    phys.position[2] += phys.velocity.z * delta;

    // Arena Boundaries
    phys.position[0] = Math.max(-6, Math.min(6, phys.position[0]));
    phys.position[2] = Math.max(-4, Math.min(4, phys.position[2]));

    // Update Visual Position
    setPosition2D([...phys.position]);
    if (onPositionChange) onPositionChange(phys.position);

    // 2. Rotation (Face Opponent)
    if (opponentPosition && meshRef.current) {
      const dx = opponentPosition[0] - phys.position[0];
      const dz = opponentPosition[2] - phys.position[2];
      const targetAngle = Math.atan2(dx, dz);
      const targetQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, targetAngle, 0));
      meshRef.current.quaternion.slerp(targetQ, 0.1);
    }

    // 3. AI Logic
    if (!isPlayer && engaged) {
      const now = Date.now();
      
      if (aiTargetPosition) {
        const dx = aiTargetPosition.x - phys.position[0];
        const dz = aiTargetPosition.z - phys.position[2];
        const dist = Math.hypot(dx, dz);
        if (dist > 0.2) {
          inputRef.current.x = (dx / dist) * 0.8;
          inputRef.current.z = (dz / dist) * 0.8;
        } else {
          inputRef.current.x = 0;
          inputRef.current.z = 0;
        }
      } else if (opponentPosition) {
        const dx = opponentPosition[0] - phys.position[0];
        const dz = opponentPosition[2] - phys.position[2];
        const dist = Math.hypot(dx, dz);
        if (dist > 1.8) {
          inputRef.current.x = (dx / dist) * 0.6;
          inputRef.current.z = (dz / dist) * 0.6;
        } else {
          inputRef.current.x = 0;
          inputRef.current.z = 0;
        }
      }

      const cooldownPassed = (now - lastAiActionTime.current) > 400;  // 700ms → 400ms
      if (aiCommand && !isAttacking && attackCooldownRef.current <= 0 && cooldownPassed) {
        if (aiCommand === 'block') {
          performBlock();
        } else {
          performAttack(aiCommand === 'kick' ? 'kick' : 'punch');
        }
        lastAiActionTime.current = now;
      }
    }

    if (attackCooldownRef.current > 0) attackCooldownRef.current -= delta * 1000;
  });

  // --- ANIMATION LOOP ---
  useFrame((state) => {
    if (paused || !meshRef.current) return;
    const time = state.clock.elapsedTime;

    // IDLE / BREATHING - Boxing stance
    if (!isAttacking && !isMoving && !isBlocking) {
      // FIX: Add 1.2 to keep torso at chest height
      if (torsoRef.current) torsoRef.current.position.y = 1.2 + Math.sin(time * 3) * 0.015;
      
      // Boxing guard position - arms up protecting face
      if (leftArmGroup.current) {
        leftArmGroup.current.rotation.x = THREE.MathUtils.lerp(leftArmGroup.current.rotation.x, -1.2, 0.1);
        leftArmGroup.current.rotation.z = THREE.MathUtils.lerp(leftArmGroup.current.rotation.z, 0.3, 0.1);
      }
      if (rightArmGroup.current) {
        rightArmGroup.current.rotation.x = THREE.MathUtils.lerp(rightArmGroup.current.rotation.x, -1.2, 0.1);
        rightArmGroup.current.rotation.z = THREE.MathUtils.lerp(rightArmGroup.current.rotation.z, -0.3, 0.1);
      }
      
      if (leftLegGroup.current) leftLegGroup.current.rotation.x = THREE.MathUtils.lerp(leftLegGroup.current.rotation.x, 0, 0.1);
      if (rightLegGroup.current) rightLegGroup.current.rotation.x = THREE.MathUtils.lerp(rightLegGroup.current.rotation.x, 0, 0.1);
    }

    // WALKING
    if (isMoving && !isAttacking && isGrounded) {
      const walkSpeed = 10;
      if (leftLegGroup.current) leftLegGroup.current.rotation.x = Math.sin(time * walkSpeed) * 0.4;
      if (rightLegGroup.current) rightLegGroup.current.rotation.x = Math.sin(time * walkSpeed + Math.PI) * 0.4;
      
      if (!isBlocking) {
        if (leftArmGroup.current) leftArmGroup.current.rotation.x = -1.0 + Math.sin(time * walkSpeed + Math.PI) * 0.2;
        if (rightArmGroup.current) rightArmGroup.current.rotation.x = -1.0 + Math.sin(time * walkSpeed) * 0.2;
      }
      
      // FIX: Add 1.2 here as well
      if (torsoRef.current) torsoRef.current.position.y = 1.2 + Math.abs(Math.sin(time * walkSpeed)) * 0.02;
    }
  });

  // --- ACTIONS ---
  const performAttack = (type: 'punch' | 'kick') => {
    if (isAttacking || isBlocking || attackCooldownRef.current > 0) return;
    
    setIsAttacking(true);
    attackCooldownRef.current = ATTACK_DURATION + 100;
    
    const isRight = Math.random() > 0.5;
    const arm = isRight ? rightArmGroup.current : leftArmGroup.current;
    const leg = isRight ? rightLegGroup.current : leftLegGroup.current;
    const torso = torsoRef.current;

    if (type === 'punch' && arm && torso) {
      // Wind up
      arm.rotation.x = -0.8; 
      arm.rotation.z = isRight ? -0.3 : 0.3;
      torso.rotation.y = isRight ? 0.2 : -0.2;

      setTimeout(() => {
        if (!arm || !torso) return;
        // Punch forward
        arm.rotation.x = -1.8;
        arm.rotation.z = 0;
        torso.rotation.y = isRight ? -0.3 : 0.3;
        torso.position.z = 0.1;
        
        if (opponentPosition) {
          const dist = Math.hypot(opponentPosition[0] - position2D[0], opponentPosition[2] - position2D[2]);
          if (dist < 2.0) onPlayerAttack?.(10);
        }
      }, 50);

      setTimeout(() => {
        if (!arm || !torso) return;
        // Return to guard
        arm.rotation.x = -1.2;
        arm.rotation.z = isRight ? -0.3 : 0.3;
        torso.rotation.y = 0;
        torso.position.z = 0;
        setIsAttacking(false);
      }, 300);
      
    } else if (type === 'kick' && leg && torso) {
      // Prepare kick
      leg.rotation.x = -0.3;
      torso.rotation.x = -0.1;

      setTimeout(() => {
        if (!leg || !torso) return;
        // Kick forward
        leg.rotation.x = -1.4;
        leg.position.z = 0.3;
        torso.rotation.x = -0.25;

        if (opponentPosition) {
          const dist = Math.hypot(opponentPosition[0] - position2D[0], opponentPosition[2] - position2D[2]);
          if (dist < 2.5) onPlayerAttack?.(15);
        }
      }, 100);

      setTimeout(() => {
        if (!leg || !torso) return;
        // Return to stance
        leg.rotation.x = 0;
        leg.position.z = 0;
        torso.rotation.x = 0;
        setIsAttacking(false);
      }, 450);
    } else {
      setTimeout(() => setIsAttacking(false), ATTACK_DURATION);
    }
    
    postAction?.({
      game_type: 'fighting',
      action_type: 'attack',
      move_type: type,
      success: true,
      timestamp: Date.now(),
      position: [position2D[0], position2D[2]],
      combo_count: 1,
      context: { player_health: playerCurrentHealth, ai_health: aiCurrentHealth }
    });
  };

  const performBlock = () => {
    if (isAttacking) return;
    setIsBlocking(true);
    onBlockStateChange?.(true);
    
    // Tighter guard position - arms crossed in front of face
    if (leftArmGroup.current && rightArmGroup.current) {
      leftArmGroup.current.rotation.set(-1.8, 0.3, 0.4);
      rightArmGroup.current.rotation.set(-1.8, -0.3, -0.4);
    }
    
    // ✅ FIXED: Track block action for ML status display
    postAction?.({
      game_type: 'fighting',
      action_type: 'defend',
      move_type: 'block',
      success: true,
      timestamp: Date.now(),
      position: [position2D[0], position2D[2]],
      combo_count: 0,
      context: { player_health: playerCurrentHealth, ai_health: aiCurrentHealth }
    });

    setTimeout(() => {
      setIsBlocking(false);
      onBlockStateChange?.(false);
    }, 500);
  };

  const jump = () => {
    if (physicsRef.current.jumpCooldown <= 0 && isGrounded) {
      physicsRef.current.velocity.y = JUMP_FORCE;
      physicsRef.current.jumpCooldown = 1000;
    }
  };

  // Keyboard Controls
  useEffect(() => {
    if (!isPlayer) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (paused) return;
      const k = e.key.toLowerCase();
      if (k === 'w') inputRef.current.z = -1;
      if (k === 's') inputRef.current.z = 1;
      if (k === 'a') inputRef.current.x = -1;
      if (k === 'd') inputRef.current.x = 1;
      if (k === ' ') jump();
      if (k === 'j') performAttack('punch');
      if (k === 'k') performAttack('kick');
      if (k === 'l') performBlock();
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (k === 'w' || k === 's') inputRef.current.z = 0;
      if (k === 'a' || k === 'd') inputRef.current.x = 0;
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [isPlayer, isAttacking, isBlocking, isGrounded, paused]);

  // Derive secondary colors for visual variety
  const skinColor = "#FFD5B8"; // Warm skin tone - brighter and more visible
  const pantsColor = isPlayer ? "#2563EB" : "#DC2626"; // Bright blue for player, bright red for AI
  const shirtColor = isPlayer ? "#3B82F6" : "#EF4444"; // Lighter blue/red for shirt
  const gloveColor = isPlayer ? "#EF4444" : "#2563EB"; // Red gloves for player, blue for AI
  const shoeColor = "#1F2937";
  const hairColor = isPlayer ? "#4A3728" : "#1A1A1A";

  return (
    <group ref={meshRef} position={position2D} scale={[1.4, 1.4, 1.4]}>
      {/* === FIGHTER CHARACTER - Proper Proportions === */}
      {/* Total height ~2 units: legs(0.85) + torso(0.7) + head(0.3) = ~1.85 + ground clearance */}
      
      {/* TORSO GROUP - Center of body at y=1.2 */}
      <group ref={torsoRef} position={[0, 1.2, 0]}>
        
        {/* CHEST - Main torso */}
        <Box args={[0.5, 0.4, 0.28]} position={[0, 0.15, 0]}>
          <meshStandardMaterial color={shirtColor} roughness={0.6} />
        </Box>
        
        {/* STOMACH/ABS */}
        <Box args={[0.44, 0.25, 0.24]} position={[0, -0.15, 0]}>
          <meshStandardMaterial color={shirtColor} roughness={0.6} />
        </Box>

        {/* HEAD */}
        <group ref={headRef} position={[0, 0.55, 0]}>
          {/* Face - visible skin */}
          <Box args={[0.28, 0.3, 0.26]}>
            <meshStandardMaterial color={skinColor} roughness={0.4} />
          </Box>
          {/* Headband */}
          <Box args={[0.3, 0.06, 0.14]} position={[0, 0.03, 0.08]}>
            <meshStandardMaterial color={isPlayer ? "#EF4444" : "#2563EB"} roughness={0.3} metalness={0.2} />
          </Box>
          {/* Hair */}
          <Box args={[0.3, 0.12, 0.28]} position={[0, 0.16, -0.02]}>
            <meshStandardMaterial color={hairColor} roughness={0.8} />
          </Box>
          {/* Eyes - small dark boxes */}
          <Box args={[0.04, 0.03, 0.02]} position={[-0.06, 0.02, 0.13]}>
            <meshStandardMaterial color="#1a1a1a" />
          </Box>
          <Box args={[0.04, 0.03, 0.02]} position={[0.06, 0.02, 0.13]}>
            <meshStandardMaterial color="#1a1a1a" />
          </Box>
        </group>

        {/* LEFT ARM - Attached at shoulder */}
        <group ref={leftArmGroup} position={[-0.32, 0.18, 0]}>
          {/* Upper Arm */}
          <Box args={[0.14, 0.28, 0.14]} position={[0, -0.14, 0]}>
            <meshStandardMaterial color={skinColor} roughness={0.4} />
          </Box>
          {/* Forearm */}
          <Box args={[0.12, 0.24, 0.12]} position={[0, -0.38, 0]}>
            <meshStandardMaterial color={skinColor} roughness={0.4} />
          </Box>
          {/* Boxing Glove */}
          <Box args={[0.18, 0.18, 0.18]} position={[0, -0.58, 0]}>
            <meshStandardMaterial color={gloveColor} roughness={0.3} metalness={0.1} />
          </Box>
        </group>

        {/* RIGHT ARM - Attached at shoulder */}
        <group ref={rightArmGroup} position={[0.32, 0.18, 0]}>
          {/* Upper Arm */}
          <Box args={[0.14, 0.28, 0.14]} position={[0, -0.14, 0]}>
            <meshStandardMaterial color={skinColor} roughness={0.4} />
          </Box>
          {/* Forearm */}
          <Box args={[0.12, 0.24, 0.12]} position={[0, -0.38, 0]}>
            <meshStandardMaterial color={skinColor} roughness={0.4} />
          </Box>
          {/* Boxing Glove */}
          <Box args={[0.18, 0.18, 0.18]} position={[0, -0.58, 0]}>
            <meshStandardMaterial color={gloveColor} roughness={0.3} metalness={0.1} />
          </Box>
        </group>

        {/* BELT */}
        <Box args={[0.46, 0.08, 0.26]} position={[0, -0.32, 0]}>
          <meshStandardMaterial color="#1F2937" roughness={0.3} metalness={0.4} />
        </Box>
        {/* Belt Buckle */}
        <Box args={[0.1, 0.06, 0.04]} position={[0, -0.32, 0.14]}>
          <meshStandardMaterial color="#FFD700" roughness={0.2} metalness={0.8} />
        </Box>
      </group>

      {/* LEGS - CRITICAL FIX: Position at HIP LEVEL to connect with torso */}
      {/* LEFT LEG - Starts at hip (y=0.8) and extends DOWN to ground */}
      <group ref={leftLegGroup} position={[-0.14, 0.8, 0]}>
        {/* Thigh - TOP of leg, connects to hip */}
        <Box args={[0.16, 0.36, 0.16]} position={[0, -0.18, 0]}>
          <meshStandardMaterial color={pantsColor} roughness={0.5} />
        </Box>
        {/* Knee */}
        <Box args={[0.14, 0.08, 0.14]} position={[0, -0.40, 0]}>
          <meshStandardMaterial color={pantsColor} roughness={0.5} />
        </Box>
        {/* Shin - extends downward */}
        <Box args={[0.14, 0.32, 0.14]} position={[0, -0.60, 0]}>
          <meshStandardMaterial color={skinColor} roughness={0.4} />
        </Box>
        {/* Foot - at ground level */}
        <Box args={[0.14, 0.08, 0.22]} position={[0, -0.84, 0.04]}>
          <meshStandardMaterial color={shoeColor} roughness={0.4} />
        </Box>
      </group>

      {/* RIGHT LEG */}
      <group ref={rightLegGroup} position={[0.14, 0.8, 0]}>
        {/* Thigh - TOP of leg, connects to hip */}
        <Box args={[0.16, 0.36, 0.16]} position={[0, -0.18, 0]}>
          <meshStandardMaterial color={pantsColor} roughness={0.5} />
        </Box>
        {/* Knee */}
        <Box args={[0.14, 0.08, 0.14]} position={[0, -0.40, 0]}>
          <meshStandardMaterial color={pantsColor} roughness={0.5} />
        </Box>
        {/* Shin - extends downward */}
        <Box args={[0.14, 0.32, 0.14]} position={[0, -0.60, 0]}>
          <meshStandardMaterial color={skinColor} roughness={0.4} />
        </Box>
        {/* Foot - at ground level */}
        <Box args={[0.14, 0.08, 0.22]} position={[0, -0.84, 0.04]}>
          <meshStandardMaterial color={shoeColor} roughness={0.4} />
        </Box>
      </group>
    </group>
  );
};


// Badminton Player Component - Realistic with animations
const BadmintonPlayer = ({ position, color, isPlayer = false, paused = false, isAI = false, followTarget, followVel, onPlayerHit, onPositionChange, aiOrder, postAction, rallyCount, badmintonScore }: { position: [number, number, number], color: string, isPlayer?: boolean, paused?: boolean, isAI?: boolean, followTarget?: [number, number, number], followVel?: [number, number, number], onPlayerHit?: (dir:[number,number,number], power:number, spin?:[number,number,number])=>void, onPositionChange?: (pos:[number,number,number])=>void, aiOrder?: { target?: [number, number]; swing?: { dir:[number,number,number]; power:number; spin?:[number,number,number] } } | null, postAction: typeof postBadmintonAction, rallyCount: number, badmintonScore: [number, number] }) => {
  const [playerPos, setPlayerPos] = useState(position);
  // Defensive: log and check playerPos after initialization
  React.useEffect(() => {
    // console.log('BadmintonPlayer (in GameArena) props:', { position, color, isPlayer, paused, isAI, followTarget, followVel, aiOrder });
    if (!Array.isArray(playerPos) || playerPos.length !== 3 || playerPos.some((v) => typeof v !== 'number' || isNaN(v))) {
      console.error('BadmintonPlayer: Invalid playerPos detected!', playerPos);
    }
  }, [playerPos, position, color, isPlayer, paused, isAI, followTarget, followVel, aiOrder]);

  if (!Array.isArray(playerPos) || playerPos.length !== 3 || playerPos.some((v) => typeof v !== 'number' || isNaN(v))) {
    return null; // Prevent crash if position is invalid
  }
  const groupRef = useRef<THREE.Group>(null);
  const racketRef = useRef<THREE.Group>(null);
  const bodyRef = useRef<THREE.Mesh>(null);
  const leftArmRef = useRef<THREE.Group>(null);
  const rightArmRef = useRef<THREE.Group>(null);
  const leftLegRef = useRef<THREE.Group>(null);
  const rightLegRef = useRef<THREE.Group>(null);
  const [isSwinging, setIsSwinging] = useState(false);
  const [isMoving, setIsMoving] = useState(false);
  const [racketPower, setRacketPower] = useState(0);
  const [facingDirection, setFacingDirection] = useState<number>(position[0] > 0 ? -1 : 1);
  const mVelRef = useRef<{x:number,z:number}>({x:0,z:0});
  const mInputRef = useRef<{x:number,z:number}>({x:0,z:0});

  useFrame((state, delta) => {
    if (paused) return;

    // Player-controlled movement physics (accel/decel)
    if (isPlayer) {
      const accel = 8;
      const maxSpeed = 4.5;
      const frictionPerFrame = 0.85;
      const friction = Math.pow(frictionPerFrame, 60 * delta);
      const v = mVelRef.current;
      const inp = mInputRef.current;

      if (inp.x !== 0) v.x += inp.x * accel * delta; else v.x *= friction;
      if (inp.z !== 0) v.z += inp.z * accel * delta; else v.z *= friction;

      const sp = Math.hypot(v.x, v.z);
      if (sp > maxSpeed) { const s = maxSpeed / sp; v.x *= s; v.z *= s; }

      let nx = playerPos[0] + v.x * delta;
      let nz = playerPos[2] + v.z * delta;
      const rightSide = position[0] > 0;
      nx = rightSide ? Math.max(0.6, Math.min(7, nx)) : Math.max(-7, Math.min(-0.6, nx));
      nz = Math.max(-2.8, Math.min(2.8, nz));
      if (nx !== playerPos[0] || nz !== playerPos[2]) setPlayerPos([nx, playerPos[1], nz]);
      setIsMoving(Math.abs(v.x) > 0.05 || Math.abs(v.z) > 0.05);
    }

    if (bodyRef.current) {
      if (!isSwinging && !isMoving) {
        // Natural breathing and ready stance
        bodyRef.current.position.y = playerPos[1] + Math.sin(state.clock.elapsedTime * 1.2) * 0.03;

        // Racket ready position with subtle movement
        if (racketRef.current) {
          racketRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.8) * 0.08;
          racketRef.current.position.y = 0.4 + Math.sin(state.clock.elapsedTime * 1.5) * 0.02;
        }

        // Subtle arm sway
        if (leftArmRef.current && rightArmRef.current) {
          leftArmRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.6) * 0.05;
          rightArmRef.current.rotation.z = -Math.sin(state.clock.elapsedTime * 0.6) * 0.05;
        }
      }

      // Enhanced walking animation
      if (isMoving && !isSwinging) {
        const walkCycle = state.clock.elapsedTime * 6;

        if (leftLegRef.current && rightLegRef.current) {
          leftLegRef.current.rotation.x = Math.sin(walkCycle) * 0.3;
          rightLegRef.current.rotation.x = Math.sin(walkCycle + Math.PI) * 0.3;
        }

        // Walking bob
        bodyRef.current.position.y = playerPos[1] + Math.abs(Math.sin(walkCycle * 2)) * 0.04;

        // Racket movement during walk
        if (racketRef.current) {
          racketRef.current.rotation.x = Math.sin(walkCycle) * 0.1;
        }
      }

      // AI follows external orders with slower speed and cooldowns
      if (isAI && aiOrder) {
        if (aiOrder.target) {
          const [tx, tz] = aiOrder.target;
          const speed = 0.06;
          const nx = playerPos[0] + Math.sign(tx - playerPos[0]) * speed;
          const nz = playerPos[2] + Math.sign(tz - playerPos[2]) * speed;
          const rightSide = position[0] > 0;
          const clampedX = rightSide ? Math.max(0.6, Math.min(7, nx)) : Math.max(-7, Math.min(-0.6, nx));
          setPlayerPos([clampedX, playerPos[1], Math.max(-2.8, Math.min(2.8, nz))]);
          setFacingDirection(tx > playerPos[0] ? 1 : -1);
        }
        if (aiOrder.swing) {
          // Trigger AI swing if needed
          // performSwing(aiOrder.swing.power, aiOrder.swing.dir, aiOrder.swing.spin);
        }
      }

      // Face across the net (toward center line)
      if (groupRef.current) {
        const target = new THREE.Vector3(0, playerPos[1], playerPos[2]);
        groupRef.current.lookAt(target);
      }
    }
  });

  // Report position changes upward
  useEffect(() => { onPositionChange?.(playerPos); }, [playerPos, onPositionChange]);

  // Enhanced player movement for badminton
  useEffect(() => {
    if (!isPlayer) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (paused || isSwinging) return;

      const pushMove = () => {
        postAction({
          game_type: 'badminton',
          action_type: 'move',
          timestamp: Date.now(),
          success: true,
          shot_type: 'drive',
          court_position: [playerPos[0], playerPos[2] || 0],
          shuttlecock_target: [0, 0],
          power_level: 0,
          rally_position: rallyCount,
          context: { rally_count: rallyCount, court_side: playerPos[2] > 0 ? 'right' : 'left', game_score: badmintonScore },
        });
      };
      switch (event.key.toLowerCase()) {
        case 'w':
          mInputRef.current.z = -1;
          setIsMoving(true);
          pushMove();
          break;
        case 's':
          mInputRef.current.z = 1;
          setIsMoving(true);
          pushMove();
          break;
        case 'a':
          mInputRef.current.x = -1;
          setIsMoving(true);
          pushMove();
          break;
        case 'd':
          mInputRef.current.x = 1;
          setIsMoving(true);
          pushMove();
          break;
        case ' ':
          if (!isSwinging) {
            setRacketPower(Math.min(racketPower + 0.1, 1));
          }
          break;
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      switch (event.key.toLowerCase()) {
        case 'w':
        case 's':
          mInputRef.current.z = 0;
          break;
        case 'a':
        case 'd':
          mInputRef.current.x = 0;
          break;
        case ' ':
          if (!isSwinging) {
            performSwing(racketPower);
            setRacketPower(0);
          }
          break;
      }
      if (mInputRef.current.x === 0 && mInputRef.current.z === 0) setIsMoving(false);
    };

    window.addEventListener('keydown', handleKeyDown, { passive: false });
    window.addEventListener('keyup', handleKeyUp, { passive: false });
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [isPlayer, isSwinging, racketPower, paused, playerPos, postAction, rallyCount, badmintonScore]);

  const performSwing = (power: number = 0.5) => {
    if (isSwinging || !racketRef.current) return;

    setIsSwinging(true);

    // Send badminton shot action (approximate as drive with power)
    postAction({
      game_type: 'badminton',
      action_type: 'shot',
      shot_type: 'drive',
      shuttlecock_target: [0, 0],
      power_level: Math.max(0, Math.min(1, power)),
      rally_position: rallyCount,
      court_position: [playerPos[0], playerPos[2]],
      success: true,
      timestamp: Date.now(),
      context: { rally_count: rallyCount, court_side: playerPos[2] > 0 ? 'right' : 'left', game_score: badmintonScore },
    });

    // Realistic swing animation with power variation
    const swingIntensity = 0.5 + power * 0.8;
    const swingSpeed = 200 + power * 200;

    // Backswing
    racketRef.current.rotation.z = Math.PI / 4;
    racketRef.current.rotation.x = -Math.PI / 6;

    // Body rotation for power
    if (bodyRef.current) {
      bodyRef.current.rotation.y = facingDirection * -0.2 * swingIntensity;
    }

    setTimeout(() => {
      if (racketRef.current) {
        // Forward swing
        racketRef.current.rotation.z = -Math.PI / 2 * swingIntensity;
        racketRef.current.rotation.x = Math.PI / 4;

        // Arm extension
        if (rightArmRef.current) {
          rightArmRef.current.rotation.x = -Math.PI / 3 * swingIntensity;
        }
        // Collision-based impact with shuttlecock near racket head
            const headLocal = new THREE.Vector3(0, 0.48, 0);
        const headWorld = racketRef.current.localToWorld(headLocal.clone());
        const sh = followTarget || [0,0,0];
        const shPos = new THREE.Vector3(sh[0], sh[1], sh[2]);

        const worldQuat = racketRef.current.getWorldQuaternion(new THREE.Quaternion());
        const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(worldQuat).normalize(); // Changed normal vector to be along racket plane
        const toSh = shPos.clone().sub(headWorld);
        const d = toSh.dot(normal);
        if (Math.abs(d) < 0.25) { // Increased margin for hit registration
          const proj = shPos.clone().sub(normal.clone().multiplyScalar(d));
          const radial = proj.clone().sub(headWorld);
          const r = radial.length();
          if (r > 0.08 && r < 0.3) { // Increased hit radius
            const across = playerPos[0] > 0 ? -1 : 1;
            const faceDir = new THREE.Vector3(0, 0, 1).applyQuaternion(worldQuat).normalize();
            const dir = new THREE.Vector3(across, 0.5, 0).add(faceDir.multiplyScalar(0.5)).normalize();
            const sweet = Math.abs(r - 0.12) < 0.08;
            const mishit = sweet ? 1 : 0.7;
            const spin = new THREE.Vector3(0, across * 10 * (sweet ? 1 : 1.5), 0);
            onPlayerHit?.([dir.x, dir.y, dir.z], Math.max(0.25, Math.min(1, power)) * mishit, [spin.x, spin.y, spin.z]);
          }
        }
      }
    }, 100);

    setTimeout(() => {
      // Follow through
      if (racketRef.current && bodyRef.current && rightArmRef.current) {
        racketRef.current.rotation.z = -Math.PI / 6;
        racketRef.current.rotation.x = 0;
        rightArmRef.current.rotation.x = 0;
        bodyRef.current.rotation.y = 0;
      }
    }, swingSpeed);

    setTimeout(() => {
      // Return to ready position
      if (racketRef.current) {
        racketRef.current.rotation.z = 0;
        racketRef.current.rotation.x = 0;
        setIsSwinging(false);
      }
    }, swingSpeed + 200);
  };

  return (
    <group ref={groupRef} position={playerPos}>
      {/* Athletic body */}
      <Box ref={bodyRef} args={[0.32, 1.0, 0.22]} position={[0, 0.1, 0]}>
        <meshPhongMaterial color={color} />
      </Box>

      {/* Head */}
      <Sphere args={[0.13]} position={[0, 0.7, 0]}>
        <meshPhongMaterial color="#f2dcc5" />
      </Sphere>

      {/* Shoulders and arms as groups */}
      <Sphere args={[0.08]} position={[-0.25, 0.45, 0]}>
        <meshPhongMaterial color={color} />
      </Sphere>
      <group ref={leftArmRef} position={[-0.25, 0.05, 0]}>
        <mesh position={[0, 0.22, 0]}>
          <cylinderGeometry args={[0.06, 0.06, 0.45, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        <Sphere args={[0.06]} position={[0, -0.02, 0]}>
          <meshPhongMaterial color="#f2dcc5" />
        </Sphere>
        <mesh position={[0, -0.3, 0]}>
          <cylinderGeometry args={[0.05, 0.05, 0.4, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        <Box args={[0.1, 0.07, 0.12]} position={[0, -0.52, 0]}>
          <meshPhongMaterial color="#f2dcc5" />
        </Box>
      </group>

      <Sphere args={[0.08]} position={[0.25, 0.45, 0]}>
        <meshPhongMaterial color={color} />
      </Sphere>
      <group ref={rightArmRef} position={[0.25, 0.05, 0]}>
        <mesh position={[0, 0.22, 0]}>
          <cylinderGeometry args={[0.06, 0.06, 0.45, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        <Sphere args={[0.06]} position={[0, -0.02, 0]}>
          <meshPhongMaterial color="#f2dcc5" />
        </Sphere>
        <mesh position={[0, -0.3, 0]}>
          <cylinderGeometry args={[0.05, 0.05, 0.4, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        <Box args={[0.1, 0.07, 0.12]} position={[0, -0.52, 0]}>
          <meshPhongMaterial color="#f2dcc5" />
        </Box>
      </group>

      {/* Hips */}
      <Sphere args={[0.08]} position={[-0.14, -0.5, 0]}>
        <meshPhongMaterial color={color} />
      </Sphere>
      <group ref={leftLegRef} position={[-0.14, -0.85, 0]}>
        <mesh position={[0, 0.24, 0]}>
          <cylinderGeometry args={[0.07, 0.07, 0.5, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        <Sphere args={[0.07]} position={[0, -0.04, 0]}>
          <meshPhongMaterial color="#f2dcc5" />
        </Sphere>
        <mesh position={[0, -0.34, 0]}>
          <cylinderGeometry args={[0.06, 0.06, 0.45, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
      </group>

      <Sphere args={[0.08]} position={[0.14, -0.5, 0]}>
        <meshPhongMaterial color={color} />
      </Sphere>
      <group ref={rightLegRef} position={[0.14, -0.85, 0]}>
        <mesh position={[0, 0.24, 0]}>
          <cylinderGeometry args={[0.07, 0.07, 0.5, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        <Sphere args={[0.07]} position={[0, -0.04, 0]}>
          <meshPhongMaterial color="#f2dcc5" />
        </Sphere>
        <mesh position={[0, -0.34, 0]}>
          <cylinderGeometry args={[0.06, 0.06, 0.45, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
      </group>

      {/* Athletic shoes */}
      <Box args={[0.18, 0.08, 0.26]} position={[-0.14, -1.26, 0.04]}>
        <meshPhongMaterial color="#e5e7eb" />
      </Box>
      <Box args={[0.18, 0.08, 0.26]} position={[0.14, -1.26, 0.04]}>
        <meshPhongMaterial color="#e5e7eb" />
      </Box>

      {/* Enhanced Professional Racket */}
      <group ref={racketRef} position={[0.28, 0.4, 0]} rotation={[0, 0, Math.PI / 6]}>
        {/* Grip */}
        <Box args={[0.025, 0.25, 0.025]} position={[0, -0.12, 0]}>
          <meshPhongMaterial color="#1A1A1A" />
        </Box>
        {/* Handle */}
        <Box args={[0.032, 0.45, 0.032]} position={[0, 0.1, 0]}>
          <meshPhongMaterial color="#2D1B69" />
        </Box>
        {/* Racket head frame */}
        <mesh position={[0, 0.48, 0]}>
          <ringGeometry args={[0.14, 0.16, 20]} />
          <meshPhongMaterial color="#FF6B35" />
        </mesh>
        {/* Racket head - realistic oval */}
        <mesh position={[0, 0.48, 0]}>
          <ringGeometry args={[0.08, 0.14, 16]} />
          <meshBasicMaterial color="#FFFFFF" side={THREE.DoubleSide} />
        </mesh>
        {/* Enhanced string pattern */}
        <mesh position={[0, 0.48, 0.005]}>
          <ringGeometry args={[0.09, 0.135, 16]} />
          <meshBasicMaterial color="#E0E0E0" wireframe />
        </mesh>
        {/* Cross strings */}
        <mesh position={[0, 0.48, -0.005]} rotation={[0, 0, Math.PI / 2]}>
          <ringGeometry args={[0.09, 0.135, 16]} />
          <meshBasicMaterial color="#E0E0E0" wireframe />
        </mesh>

        {/* Power indicator when charging */}
        {racketPower > 0 && (
          <Sphere args={[0.02 + racketPower * 0.05]} position={[0, 0.48, 0]}>
            <meshBasicMaterial color="#4ECDC4" transparent opacity={0.6} />
          </Sphere>
        )}
      </group>
      
      {/* Professional effects */}
      <Sphere args={[0.06]} position={[0, 0.75, 0.2]}>
        <meshBasicMaterial color="#4ECDC4" />
      </Sphere>
    </group>
  );
};

// FIXED: Optimized Racing Car Component with better performance
type RacingAI = 'overtake' | 'block_overtake' | 'perfect_racing_line' | null;
const RacingCar = ({ position, color, isPlayer = false, paused = false, raceRunning = true, aiCommand = null, targetX, onPositionUpdate, postAction, currentLap, racePosition, racingTotalLaps, totalDistance }: { 
  position: [number, number, number], 
  color: string, 
  isPlayer?: boolean, 
  paused?: boolean, 
  raceRunning?: boolean, 
  aiCommand?: RacingAI, 
  targetX?: number, 
  onPositionUpdate?: (pos:[number,number,number])=>void,
  // PROPS FOR ACTION LOGGING & CONTEXT:
  postAction: typeof postRacingAction,
  currentLap: number,
  racePosition: number,
  racingTotalLaps: number,
  totalDistance: number,
 }) => {
  const carRef = useRef<THREE.Group>(null);
  const wheelRefs = useRef<THREE.Mesh[]>([]);
  const [carPosition, setCarPosition] = useState(position);
  const [velocity, setVelocity] = useState(0);
  const [steering, setSteering] = useState(0);
  const [steerInput, setSteerInput] = useState(0);
  const [isAccelerating, setIsAccelerating] = useState(false);
  const [isBraking, setIsBraking] = useState(false);

  useFrame((state, delta) => {
    if (paused || !carRef.current) return;
    if (carRef.current) {
      // Realistic car physics
      let newVelocity = velocity;

      const accel = isAccelerating && raceRunning;
      const brake = isBraking && raceRunning;

      if (accel) {
        newVelocity = Math.min(velocity + delta * 3, 8);
      } else if (brake) {
        newVelocity = Math.max(velocity - delta * 5, -2);
      } else {
        // Natural deceleration
        newVelocity = raceRunning ? velocity * 0.98 : 0;
      }

      setVelocity(newVelocity);

      // Smooth steering toward input to simulate grip and tire limits
      const steerRate = 2.2; // rad/s
      const maxSteer = 0.6; // rad
      const nextSteering = THREE.MathUtils.clamp(steering + steerRate * steerInput * delta, -maxSteer, maxSteer);
      setSteering(nextSteering);

      const grip = 1 - Math.min(0.6, Math.abs(nextSteering) * 0.6);
      const moveVel = raceRunning ? newVelocity * grip : 0;
      const newX = carPosition[0] + Math.sin(nextSteering) * moveVel * delta;
      const newZ = carPosition[2] - Math.cos(nextSteering) * moveVel * delta;

      // Keep car on track
      const clampedX = Math.max(-6, Math.min(6, newX));
      const clampedZ = Math.max(-198, Math.min(198, newZ));

      const nextPos: [number, number, number] = [clampedX, -1.75, clampedZ];
      setCarPosition(nextPos); // Car properly on ground
      onPositionUpdate?.(nextPos);

      // Car rotation based on steering
      carRef.current.rotation.y = steering;

      // Wheel rotation based on speed
      wheelRefs.current.forEach((wheel) => {
        if (wheel) {
          wheel.rotation.x += moveVel * delta * 2;
        }
      });

      // Engine vibration when accelerating
      if (accel && Math.abs(moveVel) > 0.1) {
        carRef.current.position.y = carPosition[1] + Math.sin(state.clock.elapsedTime * 30) * 0.005;
      } else {
        carRef.current.position.y = carPosition[1];
      }
    }
  });

  // Simple AI driving
  useEffect(() => {
    if (isPlayer || paused) return;
    // Keep moving forward by default
    setIsAccelerating(true);
    const osc = setInterval(() => {
      setSteering(s => Math.sin(Date.now() * 0.001) * 0.2);
    }, 200);

    if (aiCommand === 'overtake') {
      setSteering(prev => (prev < 0.2 ? prev + 0.1 : 0.2));
      setTimeout(() => { setSteering(0); }, 1000);
    } else if (aiCommand === 'block_overtake' && typeof targetX === 'number') {
      const dir = targetX > carPosition[0] ? 1 : -1;
      setSteering(dir > 0 ? 0.2 : -0.2);
      setTimeout(() => setSteering(0), 800);
    } else if (aiCommand === 'perfect_racing_line') {
      // Slightly reduce steering to follow straight line
      setSteering(0);
    }

    return () => clearInterval(osc);
  }, [aiCommand, isPlayer, paused, targetX, carPosition]);

  // Enhanced car controls
  useEffect(() => {
    if (!isPlayer) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (paused) return;
      const currentSpeed = velocity;

      const send = (type: 'accelerate' | 'brake' | 'steer') => {
        postAction({
          game_type: 'racing',
          action_type: type,
          timestamp: Date.now(),
          success: true,
          // FIX: Send current speed and position for correct analytics
          speed: type === 'accelerate' ? Math.max(30, currentSpeed) : Math.max(0, currentSpeed), 
          position_on_track: [carPosition[0], carPosition[2]],
          overtaking_attempt: false,
          crash_occurred: false,
          // FIX: Use context props from GameArena
          context: { lap_number: currentLap, position_in_race: racePosition, distance_to_finish: Math.max(0, racingTotalLaps * 100 - totalDistance) },
        });
      };
      switch (event.key.toLowerCase()) {
        case 'w':
          setIsAccelerating(true);
          send('accelerate');
          break;
        case 's':
          setIsBraking(true);
          send('brake');
          break;
        case 'a':
          setSteerInput(-1);
          send('steer');
          break;
        case 'd':
          setSteerInput(1);
          send('steer');
          break;
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      switch (event.key.toLowerCase()) {
        case 'w':
          setIsAccelerating(false);
          break;
        case 's':
          setIsBraking(false);
          break;
        case 'a':
          setSteerInput(0);
          break;
        case 'd':
          setSteerInput(0);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown, { passive: false });
    window.addEventListener('keyup', handleKeyUp, { passive: false });
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [isPlayer, postAction, velocity, carPosition, currentLap, racePosition, racingTotalLaps, totalDistance, paused]);

  return (
    <group ref={carRef} position={carPosition}>
      {/* Enhanced car body (longer front-to-back for vertical track) */}
      <Box args={[0.7, 0.3, 1.6]} position={[0, 0.1, 0]}>
        <meshPhongMaterial color={color} shininess={100} />
      </Box>

      {/* Car cabin */}
      <Box args={[0.6, 0.25, 1.2]} position={[0, 0.35, -0.1]}>
        <meshPhongMaterial color={color} />
      </Box>

      {/* Realistic wheels with rims */}
      {[[-0.35, -0.15, 0.7], [0.35, -0.15, 0.7], [-0.35, -0.15, -0.7], [0.35, -0.15, -0.7]].map((wheelPos, i) => (
        <group key={i} position={wheelPos}>
          {/* Tire */}
          <Sphere ref={(el) => { if (el) wheelRefs.current[i] = el; }} args={[0.15]} scale={[1, 0.7, 1]}>
            <meshPhongMaterial color="#2C2C2C" />
          </Sphere>
          {/* Rim */}
          <Sphere args={[0.08]} scale={[1, 0.3, 1]}>
            <meshPhongMaterial color="#C0C0C0" shininess={200} />
          </Sphere>
        </group>
      ))}

      {/* Windshield */}
      <Box args={[0.6, 0.2, 0.02]} position={[0, 0.3, 0.75]}>
        <meshPhongMaterial color="#4FC3F7" transparent opacity={0.8} />
      </Box>

      {/* Rear windshield */}
      <Box args={[0.6, 0.15, 0.02]} position={[0, 0.25, -0.75]}>
        <meshPhongMaterial color="#4FC3F7" transparent opacity={0.8} />
      </Box>

      {/* Headlights */}
      <Sphere args={[0.05]} position={[-0.5, 0.05, 0.35]}>
        <meshBasicMaterial color="#FFFFFF" />
      </Sphere>
      <Sphere args={[0.05]} position={[0.5, 0.05, 0.35]}>
        <meshBasicMaterial color="#FFFFFF" />
      </Sphere>

      {/* Taillights */}
      <Sphere args={[0.04]} position={[-0.4, 0.05, -0.35]}>
        <meshBasicMaterial color="#FF4444" />
      </Sphere>
      <Sphere args={[0.04]} position={[0.4, 0.05, -0.35]}>
        <meshBasicMaterial color="#FF4444" />
      </Sphere>

      {/* Spoiler */}
      <Box args={[1.2, 0.05, 0.15]} position={[0, 0.45, -0.3]}>
        <meshPhongMaterial color={color} />
      </Box>
    </group>
     );
};


const ArenaEnvironment = ({ gameType }: { gameType: 'fighting' | 'badminton' | 'racing' }) => {
  return (
    <>
      {/* Enhanced Arena Floor with professional gaming aesthetics */}
      <Plane args={[25, 25]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]}>
        <meshPhongMaterial
          color={gameType === 'fighting' ? "#1A1A2E" : gameType === 'badminton' ? "#1A2B1A" : "#2A2A2A"}
        />
      </Plane>

      {/* Fighting arena octagon ring - larger and more prominent */}
      {gameType === 'fighting' && (
        <>
          <mesh position={[0, -1.98, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <ringGeometry args={[5, 5.4, 8]} />
            <meshBasicMaterial color="#FFD700" />
          </mesh>
          <mesh position={[0, -1.97, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[5, 8]} />
            <meshPhongMaterial color="#2A2A4A" />
          </mesh>
        </>
      )}

      {/* Arena floor grid pattern for non-fighting games */}
      {gameType !== 'fighting' && Array.from({ length: 10 }, (_, i) => (
        <React.Fragment key={i}>
          <Plane args={[0.05, 25]} rotation={[-Math.PI / 2, 0, 0]} position={[-10 + i * 2, -1.99, 0]}>
            <meshBasicMaterial color="#4ECDC4" transparent opacity={0.3} />
          </Plane>
          <Plane args={[25, 0.05]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.99, -10 + i * 2]}>
            <meshBasicMaterial color="#4ECDC4" transparent opacity={0.3} />
          </Plane>
        </React.Fragment>
      ))}
      
      {/* Enhanced Fighting Arena Environment */}
      {gameType === 'fighting' && (
        <>
          {/* Arena cage structure */}
          <mesh position={[0, 1, 0]}>
            <cylinderGeometry args={[6, 6, 4, 8, 1, true]} />
            <meshBasicMaterial color="#C0C0C0" wireframe transparent opacity={0.6} />
          </mesh>

          {/* Arena posts */}
          {Array.from({ length: 8 }, (_, i) => {
            const angle = (i / 8) * Math.PI * 2;
            const x = Math.cos(angle) * 6;
            const z = Math.sin(angle) * 6;
            return (
              <Box key={i} args={[0.1, 4, 0.1]} position={[x, 1, z]}>
                <meshPhongMaterial color="#808080" />
              </Box>
            );
          })}

          {/* Audience seating */}
          {Array.from({ length: 12 }, (_, i) => {
            const angle = (i / 12) * Math.PI * 2;
            const x = Math.cos(angle) * 8;
            const z = Math.sin(angle) * 8;
            return (
              <Box key={i} args={[2, 1.5, 1]} position={[x, 0.75, z]} rotation={[0, -angle, 0]}>
                <meshPhongMaterial color="#2A2A4A" />
              </Box>
            );
          })}

          {/* Arena lights */}
          <Box args={[0.3, 0.3, 0.3]} position={[0, 8, 0]}>
            <meshBasicMaterial color="#FFFFFF" />
          </Box>
          {Array.from({ length: 8 }, (_, i) => {
            const angle = (i / 8) * Math.PI * 2;
            const x = Math.cos(angle) * 7.5;
            const z = Math.sin(angle) * 7.5;
            return (
              <group key={i}>
                <Box args={[0.2, 0.2, 0.2]} position={[x, 6, z]}>
                  <meshBasicMaterial color="#FFD700" />
                </Box>
                <Sphere args={[0.25]} position={[x, 6.3, z]}>
                  <meshBasicMaterial color="#FFFFAA" transparent opacity={0.9} />
                </Sphere>
                <pointLight position={[x, 6.3, z]} intensity={0.8} color="#FFE680" />
              </group>
            );
          })}

          {/* Background arena walls */}
          <Plane args={[30, 15]} position={[0, 6, -15]}>
            <meshBasicMaterial color="#0f1320" />
          </Plane>
          <Plane args={[30, 15]} rotation={[0, Math.PI, 0]} position={[0, 6, 15]}>
            <meshBasicMaterial color="#0f1320" />
          </Plane>
          <Plane args={[30, 15]} rotation={[0, Math.PI / 2, 0]} position={[-15, 6, 0]}>
            <meshBasicMaterial color="#0c0f1a" />
          </Plane>
          <Plane args={[30, 15]} rotation={[0, -Math.PI / 2, 0]} position={[15, 6, 0]}>
            <meshBasicMaterial color="#0c0f1a" />
          </Plane>
        </>
      )}
      
      {/* Realistic Badminton Court */}
      {gameType === 'badminton' && (
        <>
          {/* Court surface */}
          <Plane args={[13, 6]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.86, 0]}>
            <meshPhongMaterial color="#2E7D32" />
          </Plane>

          {/* Net */}
          <Box args={[0.05, 1.55, 6.1]} position={[0, 0.775, 0]}>
            <meshPhongMaterial color="#FFFFFF" />
          </Box>

          {/* Net posts */}
          <Box args={[0.08, 1.6, 0.08]} position={[0, 0.8, 3.05]}>
            <meshPhongMaterial color="#1A1A1A" />
          </Box>
          <Box args={[0.08, 1.6, 0.08]} position={[0, 0.8, -3.05]}>
            <meshPhongMaterial color="#1A1A1A" />
          </Box>

          {/* Court boundary lines */}
          <Plane args={[13, 0.05]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.84, 3]}>
            <meshBasicMaterial color="#FFFFFF" />
          </Plane>
          <Plane args={[13, 0.05]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.84, -3]}>
            <meshBasicMaterial color="#FFFFFF" />
          </Plane>
          <Plane args={[0.05, 6]} rotation={[-Math.PI / 2, 0, 0]} position={[6.5, -1.84, 0]}>
            <meshBasicMaterial color="#FFFFFF" />
          </Plane>
          <Plane args={[0.05, 6]} rotation={[-Math.PI / 2, 0, 0]} position={[-6.5, -1.84, 0]}>
            <meshBasicMaterial color="#FFFFFF" />
          </Plane>

          {/* Service lines */}
          <Plane args={[4, 0.05]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.83, 1.25]}>
            <meshBasicMaterial color="#FFFFFF" />
          </Plane>
          <Plane args={[4, 0.05]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.83, -1.25]}>
            <meshBasicMaterial color="#FFFFFF" />
          </Plane>
        </>
      )}
      
      {/* FIXED: Optimized Racing Circuit - removed performance-heavy elements */}
      {gameType === 'racing' && (
        <>
          {/* Main track surface - simpler rendering */}
          <Plane args={[15, 400]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.85, 0]}>
            <meshPhongMaterial color="#1A1A1A" />
          </Plane>

          {/* Track borders - fewer elements */}
          <Box args={[0.3, 0.2, 400]} position={[-7.5, -1.75, 0]}>
            <meshBasicMaterial color="#FF6B35" />
          </Box>
          <Box args={[0.3, 0.2, 400]} position={[7.5, -1.75, 0]}>
            <meshBasicMaterial color="#FF6B35" />
          </Box>

          {/* FIXED: Reduced center line elements - only 20 instead of 100 */}
          {Array.from({ length: 20 }, (_, i) => (
            <Box key={i} args={[0.2, 0.02, 1.5]} position={[0, -1.83, -180 + i * 18]} rotation={[-Math.PI / 2, 0, 0]}>
              <meshBasicMaterial color="#FFD700" />
            </Box>
          ))}

          {/* FIXED: Reduced barriers - only 8 instead of 24 */}
          {Array.from({ length: 8 }, (_, i) => (
            <React.Fragment key={i}>
              <Box args={[0.5, 1, 3]} position={[-9, -1, -140 + i * 35]}>
                <meshPhongMaterial color="#C0C0C0" />
              </Box>
              <Box args={[0.5, 1, 3]} position={[9, -1, -140 + i * 35]}>
                <meshPhongMaterial color="#C0C0C0" />
              </Box>
            </React.Fragment>
          ))}

          {/* Simple grandstands */}
          <Box args={[3, 2, 15]} position={[-12, 0, 0]}>
            <meshPhongMaterial color="#2A2A2A" />
          </Box>
          <Box args={[3, 2, 15]} position={[12, 0, 0]}>
            <meshPhongMaterial color="#2A2A2A" />
          </Box>

          {/* Start/finish line */}
          <Plane args={[15, 0.5]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.82, 98]}>
            <meshBasicMaterial color="#FFFFFF" />
          </Plane>

          {/* REMOVED: Sky background and city elements that were causing lag */}
        </>
      )}
      
      {/* Enhanced Professional Gaming Lighting */}
      <ambientLight
        intensity={gameType === 'racing' ? 0.3 : gameType === 'badminton' ? 0.5 : 0.6}
        color={gameType === 'racing' ? "#3A3A5A" : "#FFFFFF"}
      />

      {/* Main arena lighting - boosted for fighting */}
      <directionalLight
        position={[15, 12, 8]}
        intensity={gameType === 'racing' ? 0.8 : gameType === 'badminton' ? 1.2 : 1.5}
        color={gameType === 'fighting' ? "#FFFFFF" : gameType === 'badminton' ? "#FFFFFF" : "#FFFFCC"}
      />

      {/* Accent lighting */}
      <directionalLight
        position={[-10, 8, -6]}
        intensity={gameType === 'racing' ? 0.6 : 1.0}
        color={gameType === 'fighting' ? "#FFDDDD" : gameType === 'badminton' ? "#87CEEB" : "#4ECDC4"}
      />

      {/* Front fill light for fighters */}
      {gameType === 'fighting' && (
        <directionalLight
          position={[0, 5, 10]}
          intensity={1.2}
          color="#FFFFFF"
        />
      )}

      {/* Central spotlight */}
      <spotLight
        position={[0, 15, 0]}
        intensity={gameType === 'racing' ? 1.2 : gameType === 'badminton' ? 1.1 : 1.5}
        angle={Math.PI / 4}
        penumbra={0.3}
        color={gameType === 'racing' ? "#FFFFCC" : "#FFFFFF"}
      />

      {/* Fighter spotlights - illuminate the ring */}
      {gameType === 'fighting' && (
        <>
          <spotLight position={[-5, 8, 5]} intensity={1.0} angle={Math.PI / 4} penumbra={0.5} color="#FF8888" target-position={[-3, 0, 0]} />
          <spotLight position={[5, 8, 5]} intensity={1.0} angle={Math.PI / 4} penumbra={0.5} color="#8888FF" target-position={[3, 0, 0]} />
          <pointLight position={[0, 3, 3]} intensity={0.8} color="#FFFFFF" />
        </>
      )}

      {/* Stadium lighting for badminton */}
      {gameType === 'badminton' && (
        <>
          <spotLight position={[-10, 12, -8]} intensity={0.8} angle={Math.PI / 6} color="#FFFFFF" />
          <spotLight position={[10, 12, -8]} intensity={0.8} angle={Math.PI / 6} color="#FFFFFF" />
        </>
      )}

      {/* FIXED: Reduced racing lights - only 3 pairs instead of 6 */}
      {gameType === 'racing' && (
        <>
          {Array.from({ length: 3 }, (_, i) => (
            <React.Fragment key={i}>
              <spotLight
                position={[-11, 4, -12 + i * 8]}
                intensity={1.2}
                angle={Math.PI / 3}
                penumbra={0.5}
                color="#FFFFCC"
              />
              <spotLight
                position={[11, 4, -12 + i * 8]}
                intensity={1.2}
                angle={Math.PI / 3}
                penumbra={0.5}
                color="#FFFFCC"
              />
            </React.Fragment>
          ))}
        </>
      )}

      {/* Rim lighting (reduced to avoid covering players) */}
      <pointLight position={[-8, 4, 8]} intensity={0.5} color="#4ECDC4" />
      <pointLight position={[8, 4, -8]} intensity={0.5} color="#A855F7" />
      <pointLight position={[-8, 4, -8]} intensity={0.5} color="#FF6B35" />

    </>
  );
};

// Enhanced Professional Gaming Camera Controller
const CameraController = ({ gameType, playerCarPos }: { gameType: 'fighting' | 'badminton' | 'racing'; playerCarPos?: [number, number, number] }) => {
  const { camera, gl } = useThree();
  const controlsRef = useRef<any>(null);

  useEffect(() => {
    if (gameType !== 'racing') {
      const targetPositions = {
        fighting: { position: [4, 2, 4], target: [0, 0.5, 0] },
        badminton: { position: [0, 4, 5], target: [0, 1, 0] },
      } as const;
      const target = targetPositions[gameType as 'fighting' | 'badminton'];
      if (!target) return;
      const startPos = camera.position.clone();
      const endPos = new THREE.Vector3(...target.position);
      const startTime = Date.now();
      const duration = 800;
      const animate = () => {
        const elapsed = Date.now() - startTime;
        const t = Math.min(elapsed / duration, 1);
        const e = 1 - Math.pow(1 - t, 3);
        camera.position.lerpVectors(startPos, endPos, e);
        camera.lookAt(...target.target);
        if (t < 1) requestAnimationFrame(animate);
      };
      animate();
      return;
    }
  }, [camera, gameType]);

  // Follow player car smoothly in racing
  useFrame(() => {
    if (gameType !== 'racing' || !playerCarPos) return;
    const car = new THREE.Vector3(...playerCarPos);
    const desired = car.clone().add(new THREE.Vector3(0, 3, 5));
    camera.position.lerp(desired, 0.05);
    camera.lookAt(car.x, car.y + 0.5, car.z - 2);
  });

  return null;
};

// =======================================================
// MAIN GAME ARENA COMPONENT
// =======================================================

const GameArena: React.FC<GameArenaProps> = ({ gameType, onGameChange, showAnalytics, onToggleAnalytics }) => {
  // Defensive global error logging (must be inside component)
  useEffect(() => {
    window.addEventListener('error', e => {
      console.error("Global error:", e.error || e.message || e);
    });
  }, []);

  // ============================================================================
  // CRITICAL FIX: Use consistent session ID from analytics
  // ============================================================================
  const sessionRef = useRef(getSessionId());
  const wsReconnectAttempts = useRef(0);
  const isFetchingAI = useRef(false); // ✅ Track AI request status to prevent overlap
  
  const [gameStarted, setGameStarted] = useState(false);
  const [paused, setPaused] = useState(false);
  const [playerCarPos, setPlayerCarPos] = useState<[number, number, number]>([0, -1.75, 0]);
  const [raceRunning, setRaceRunning] = useState(false);
  const [raceCountdown, setRaceCountdown] = useState<number | null>(null);
  const [raceOver, setRaceOver] = useState<string | null>(null);
  const [aiCar1Pos, setAiCar1Pos] = useState<[number, number, number]>([2, -1.75, -3]);
  const [aiCar2Pos, setAiCar2Pos] = useState<[number, number, number]>([0, -1.75, -6]);
  const [aiRaceCmd1, setAiRaceCmd1] = useState<RacingAI>(null);
  const [aiRaceCmd2, setAiRaceCmd2] = useState<RacingAI>(null);
  // RACING AI state (Fixes: make AI compete and behave)
  const [aiCar1Strategy, setAiCar1Strategy] = useState<'racing' | 'overtake' | 'defend'>('racing');
  const [aiCar2Strategy, setAiCar2Strategy] = useState<'racing' | 'overtake' | 'defend'>('racing');
  const [aiCar1Speed, setAiCar1Speed] = useState(6);
  const [aiCar2Speed, setAiCar2Speed] = useState(5.5);
  const aiCar1VelRef = useRef<number>(0);
  const aiCar2VelRef = useRef<number>(0);

  // FIXED: Enhanced state management for real analytics with consistent session ID
  const [wsEnabled, setWsEnabled] = useState(false); // ✅ Disabled WS - using HTTP only
  const { connected, lastMessage, sendGameAction, sendGameStateUpdate, connect } = useMultiGameWebSocket(sessionRef.current, wsEnabled);
  const [aiFightCmd, setAiFightCmd] = useState<FightingAI>(null);
  const [aiBadmintonShot, setAiBadmintonShot] = useState<'drop_shot' | 'smash' | 'clear' | 'net_shot' | null>(null);
  const [aiRaceCmd, setAiRaceCmd] = useState<RacingAI>(null);

  // FIGHTING GAME: Player and AI positions (MOVED UP to fix initialization order)
  const [playerPosF, setPlayerPosF] = useState<[number, number, number]>([-4.5, -2, 0]);
  const [aiPosF, setAiPosF] = useState<[number, number, number]>([4.5, -2, 0]);
  
  // ✅ Position refs - declared early so they're available for damage handlers
  const playerPosFRef = useRef([-4.5, -2, 0]);
  const aiPosFRef = useRef([4.5, -2, 0]);
  
  const [playerBlocking, setPlayerBlocking] = useState(false);
  const [aiTargetF, setAiTargetF] = useState<{x: number, y: number, z: number} | null>(null); // Backend target

  // ============================================================================
  // WebSocket DISABLED - Using HTTP-only mode
  // ============================================================================
  // Removed WebSocket reconnection logic since wsEnabled = false

  // Reset reconnect attempts on successful connection
  useEffect(() => {
    if (connected) {
      // Reduced logging for WebSocket connection
      wsReconnectAttempts.current = 0;
    }
  }, [connected]);

  // FIXED: Real-time scoring state with actual updates
  const [fightingRounds, setFightingRounds] = useState<[number, number]>([0, 0]);
  const [playerHealth, setPlayerHealth] = useState(100);
  const [aiHealth, setAiHealth] = useState(100);
  const [combos, setCombos] = useState<{ player: number; ai: number }>({ player: 0, ai: 0 });

  const [badmintonScore, setBadmintonScore] = useState<[number, number]>([0, 0]);
  const [rallyCount, setRallyCount] = useState(0);
  const [gamePoint, setGamePoint] = useState<'player' | 'ai' | null>(null);

  const [currentLap, setCurrentLap] = useState(1);
  const [racingTotalLaps, setRacingTotalLaps] = useState(3);
  const [racePosition, setRacePosition] = useState(1);
  const [racingTotalRacers, setRacingTotalRacers] = useState(6);
  const [lapTimes, setLapTimes] = useState<number[]>([]);
  const [totalDistance, setTotalDistance] = useState(0);

  // Speed approximation for analytics
  const lastCarPosRef = useRef<[number, number, number]>(playerCarPos);
  const lastCarTsRef = useRef<number>(performance.now());

  // ============================================================================
  // 🧠 PERSONALITY ANALYSIS TRACKING - Every 10 actions
  // ============================================================================
  const [actionCount, setActionCount] = useState(0);
  const [lastPersonalityUpdate, setLastPersonalityUpdate] = useState<number>(0);
  const API_BASE = 'http://localhost:8000/api/v1';
  
  // ============================================================================
  // 🎮 LIVE ACTION DISPLAY - For instructor demo
  // ============================================================================
  const [lastPlayerAction, setLastPlayerAction] = useState<{action: string, time: number} | null>(null);
  const [lastAiAction, setLastAiAction] = useState<{action: string, confidence: number, time: number} | null>(null);
  const [playerStyle, setPlayerStyle] = useState<string>('Analyzing...');

  // FIXED: Real analytics state management
  const [analyticsState, setAnalyticsState] = useState<any>(null);
  const [personalityState, setPersonalityState] = useState<any>(null);
  const [analyticsLastUpdated, setAnalyticsLastUpdated] = useState<number | null>(null);
  const [liveAnalytics, setLiveAnalytics] = useState<any>({
    totalGames: 0,
    winRate: 0,
    avgGameTime: '0:00',
    favoriteGame: 'Fighting',
    recentMatches: [],
    skillLevels: { fighting: 0, badminton: 0, racing: 0 }
  });
  // const [actionSent, setActionSent] = useState(false); // Removed: not used

  // FIXED: Real analytics fetching when actions are sent
  useEffect(() => {
    // Removed: actionSent logic (no longer used)
  }, []); // Removed: actionSent dependency

  // Fetch analytics and personality from backend when analytics panel is opened
  const fetchAnalytics = async () => {
    try {
      const sid = sessionRef.current || getSessionId();
      const analytics = await fetchSessionAnalytics(sid);
      setAnalyticsState(analytics);
      setAnalyticsLastUpdated(Date.now());
    } catch (e) {
      if (import.meta.env.DEV) console.warn('Failed to fetch analytics', e);
    }
    try {
      const sid = sessionRef.current || getSessionId();
      const personality = await fetchPersonalityProfile(sid);
      setPersonalityState(personality);
      setAnalyticsLastUpdated(Date.now());
    } catch (e) {
      if (import.meta.env.DEV) console.warn('Failed to fetch personality profile', e);
    }
  };

  useEffect(() => {
    let mounted = true;
    let intervalId: number | null = null;
    if (showAnalytics) {
      // initial load
      void fetchAnalytics();
      // refresh every 3s while panel open
      intervalId = window.setInterval(() => { void fetchAnalyticsAfterAction(); }, 3000);
    }
    return () => {
      mounted = false;
      if (intervalId) window.clearInterval(intervalId);
    };
  }, [showAnalytics]);

  // Update analytics/personality from websocket messages if backend pushes updates
  useEffect(() => {
    if (!lastMessage) return;
    try {
      const msg = lastMessage as any;
      if (msg.type === 'analytics_update' && msg.analytics) {
        setAnalyticsState((prev: any) => ({ ...(prev || {}), ...msg.analytics }));
      }
      if (msg.type === 'personality_update' && msg.personality_profile) {
        setPersonalityState((prev: any) => ({ ...(prev || {}), ...msg.personality_profile }));
      }
      
      // ============================================================================
      // CRITICAL FIX: Handle AI commands from backend (ai_action, not ai_response)
      // ============================================================================
      if (msg.type === 'game_update' && msg.data?.ai_action) {
        const aiAction = msg.data.ai_action;
        
        // Logging removed for performance
        
        // Update AI command based on current game type
        if (gameType === 'fighting') {
          setAiFightCmd(aiAction.action || aiAction);
          // ✅ FIX: Use TARGET position, not direct position (prevents teleporting)
          // Let the physics engine walk the AI to the target smoothly
          if (aiAction.position) {
            setAiTargetF({
              x: aiAction.position.x,
              y: aiAction.position.y || 0,
              z: aiAction.position.z
            });
          }
        } else if (gameType === 'badminton') {
          setAiBadmintonShot(aiAction.action || aiAction);
          if (aiAction.target) {
            // Update AI badminton target position
          }
        } else if (gameType === 'racing') {
          setAiRaceCmd(aiAction.action || aiAction);
          if (aiAction.position) {
            setAiCar1Pos([aiAction.position.x, aiAction.position.y, aiAction.position.z]);
          }
        }
      }
      
      // Also update live summary counters when receiving game updates
      if (msg.type === 'game_update' && msg.data) {
        const { session_stats, personality } = msg.data;
        if (session_stats) {
          setAnalyticsState(prev => ({ ...(prev || {}), session_info: session_stats, performance_metrics: {
            fighting_health: playerHealth,
            ai_health: aiHealth,
            badminton_score: badmintonScore,
            racing_position: racePosition,
            racing_lap: currentLap
          }}));
          if (personality) setPersonalityState(personality);
        }
        setLiveAnalytics(prev => ({
          ...prev,
          totalGames: (prev.totalGames || 0) + 1,
          favoriteGame: gameType.charAt(0).toUpperCase() + gameType.slice(1)
        }));
      }
    } catch (e) {
      if (import.meta.env.DEV) console.debug('Failed to apply WS analytics update', e);
    }
  }, [lastMessage, gameType]);

  // Listen for analytics events dispatched after action POSTs
  useEffect(() => {
    const handleAnalyticsUpdate = (event: Event) => {
      try {
        const ce = event as CustomEvent;
        const { session_stats, personality } = ce.detail || {};
        if (session_stats) {
          setAnalyticsState((prev: any) => ({ ...(prev || {}), ...session_stats }));
        }
        if (personality) {
          setPersonalityState(personality);
        }
        setAnalyticsLastUpdated(Date.now());
      } catch (e) {
        if (import.meta.env.DEV) console.debug('Failed to process analytics update', e);
      }
    };
    window.addEventListener('analyticsUpdate', handleAnalyticsUpdate as EventListener);
    return () => window.removeEventListener('analyticsUpdate', handleAnalyticsUpdate as EventListener);
  }, []);

  // ✅ CRITICAL FIX: Enhanced AI action event listener - MUST BE BEFORE polling starts
  const lastAiEventTime = useRef(0);
  const AI_EVENT_THROTTLE_MS = 300;  // ✅ REDUCED to 300ms for responsive AI
  
  useEffect(() => {
    const handleAiActionUpdate = (event: Event) => {
      try {
        // ✅ THROTTLE: Skip if too soon since last event
        const now = Date.now();
        if (now - lastAiEventTime.current < AI_EVENT_THROTTLE_MS) {
          return; // Skip this event - too frequent
        }
        lastAiEventTime.current = now;
        
        const ce = event as CustomEvent;
        const { gameType: actionGameType, ai_action } = ce.detail || {};
        
        // Reduced logging - only log occasionally
        // console.log('🤖 AI ACTION RECEIVED:', { actionGameType, ai_action });
        
        if (!ai_action || actionGameType !== 'fighting') {
          if (actionGameType === 'badminton') {
            setAiBadmintonShot(ai_action?.action || ai_action);
            setTimeout(() => setAiBadmintonShot(null), 500);
          } else if (actionGameType === 'racing') {
            setAiRaceCmd(ai_action?.action || ai_action);
            if (ai_action?.position) {
              setAiCar1Pos([ai_action.position.x, ai_action.position.y, ai_action.position.z]);
            }
            setTimeout(() => setAiRaceCmd(null), 500);
          }
          return;
        }
        
        const action = ai_action.action || ai_action;
        const position = ai_action.position;
        
        // Reduced logging
        
        // ✅ SIMPLIFIED: Direct position update, less validation overhead
        if (position && typeof position === 'object') {
          setAiFightCmd(action);
          setAiTargetF({
            x: position.x || 0,
            y: position.y || 0,
            z: position.z || 0
          });
        } else {
          // No position, just set the command
          setAiFightCmd(action);
        }
        
        // Clear command after action completes
        setTimeout(() => {
          setAiFightCmd(null);
        }, 800);
        
      } catch (e) {
        console.error('❌ Failed to process AI action:', e);
      }
    };
    
    // Register listener IMMEDIATELY when component mounts
    window.addEventListener('aiActionUpdate', handleAiActionUpdate as EventListener);
    // Reduced logging
    
    return () => {
      window.removeEventListener('aiActionUpdate', handleAiActionUpdate as EventListener);
    };
  }, []); // Empty deps = runs once on mount, BEFORE polling starts

  // Fetch analytics after actions helper
  const fetchAnalyticsAfterAction = async () => {
    try {
      const sid = sessionRef.current || getSessionId();
      const [analytics, personality] = await Promise.all([
        fetchSessionAnalytics(sid),
        fetchPersonalityProfile(sid)
      ]);
      setAnalyticsState(analytics);
      setPersonalityState(personality);
      setAnalyticsLastUpdated(Date.now());
      if (import.meta.env.DEV) console.debug('✅ Analytics refreshed:', {
        total_actions: analytics?.session_info?.total_actions,
        personality_type: personality?.personality_type || personality?.archetype
      });
    } catch (e) {
      if (import.meta.env.DEV) console.warn('Failed to fetch analytics', e);
    }
  };

  // ============================================================================
  // 🧠 PERSONALITY ANALYSIS - Triggered every 10 actions
  // ============================================================================
  const triggerPersonalityAnalysis = async () => {
    try {
      const response = await fetch(`${API_BASE}/player/quick-analyze?session_id=${sessionRef.current}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success') {
          setLastPersonalityUpdate(Date.now());
          
          // Dispatch event to update UI components
          window.dispatchEvent(new CustomEvent('personalityUpdate', {
            detail: {
              personality_type: data.personality_type,
              playstyle: data.playstyle,
              traits: data.traits,
              confidence: data.confidence
            }
          }));
        }
      }
    } catch (e) {
      console.warn('Personality analysis failed:', e);
    }
  };

  // Track action and trigger personality analysis every 10 actions
  const trackAction = () => {
    setActionCount(prev => {
      const newCount = prev + 1;
      
      // Trigger personality analysis every 10 actions
      if (newCount % 10 === 0) {
        triggerPersonalityAnalysis();
      }
      
      return newCount;
    });
  };

  // Tracked action wrappers - use these instead of direct postAction calls
  const postFightingActionTracked = async (actionData: any) => {
    try {
      // Update player action display IMMEDIATELY
      const playerAction = actionData.move_type || actionData.action_type || 'action';
      console.log('🎮 Player Action:', playerAction); // Debug log
      setLastPlayerAction({ action: playerAction, time: Date.now() });
      
      // Determine player style based on recent actions
      if (playerAction === 'punch' || playerAction === 'kick' || playerAction === 'combo') {
        setPlayerStyle('Aggressive');
      } else if (playerAction === 'block') {
        setPlayerStyle('Defensive');
      } else if (playerAction === 'dodge') {
        setPlayerStyle('Evasive');
      }
      
      const result = await postFightingAction(actionData);
      console.log('🤖 API Response:', result); // Debug log
      trackAction();
      
      // Update AI action display from response
      if (result?.ai_action) {
        console.log('🤖 AI Action:', result.ai_action.action, 'Confidence:', result.ai_action.confidence); // Debug log
        setLastAiAction({
          action: result.ai_action.action || 'idle',
          confidence: result.ai_action.confidence || 0.5,
          time: Date.now()
        });
      } else {
        console.warn('⚠️ No ai_action in response:', result);
      }
      
      return result;
    } catch (e) {
      console.error('Fighting action post failed:', e);
      return null;
    }
  };

  const postBadmintonActionTracked = async (actionData: any) => {
    try {
      const result = await postBadmintonAction(actionData);
      trackAction();
      return result;
    } catch (e) {
      console.error('Badminton action post failed:', e);
      return null;
    }
  };

  const postRacingActionTracked = async (actionData: any) => {
    try {
      const result = await postRacingAction(actionData);
      trackAction();
      return result;
    } catch (e) {
      console.error('Racing action post failed:', e);
      return null;
    }
  };

  // Wrapper to send action then refresh analytics
  const sendActionAndRefresh = async (actionFn: () => Promise<any>) => {
    try {
      await actionFn();
      // allow backend a short moment to process
      setTimeout(() => { void fetchAnalyticsAfterAction(); }, 500);
    } catch (e) {
      console.error('Action send failed:', e);
    }
  };

  // FIXED: Enhanced AI opponent behavior with damage callbacks
  const handlePlayerTakeDamage = (damage: number) => {
    if (playerBlocking) {
      // Reduced logging for block
      damage = Math.floor(damage * 0.1); // Reduce damage to 10% when blocking
    }
    setPlayerHealth(prev => {
      const newHealth = Math.max(0, prev - damage);
      if (newHealth <= 0) {
        // Round over - AI wins - reset positions too
        setFightingRounds(prev => [prev[0], prev[1] + 1]);
        setPlayerHealth(100);
        setAiHealth(100);
        // ✅ FIX: Reset ALL position state on round end
        setPlayerPosF([-4.5, -2, 0]);
        setAiPosF([4.5, -2, 0]);
        playerPosFRef.current = [-4.5, -2, 0];
        aiPosFRef.current = [4.5, -2, 0];
        setAiTargetF(null); // ✅ Clear AI target to prevent movement after reset
        setAiFightCmd(null); // ✅ Clear AI command
        if (import.meta.env.DEV) console.debug(`Round over! AI wins. Score: ${fightingRounds[0]}-${fightingRounds[1] + 1}`);
      }
      return newHealth;
    });
  };

  const handleAITakeDamage = (damage: number) => {
    setAiHealth(prev => {
      const newHealth = Math.max(0, prev - damage);
      if (newHealth <= 0) {
        // Round over - Player wins - reset positions too
        setFightingRounds(prev => [prev[0] + 1, prev[1]]);
        setPlayerHealth(100);
        setAiHealth(100);
        // ✅ FIX: Reset ALL position state on round end
        setPlayerPosF([-4.5, -2, 0]);
        setAiPosF([4.5, -2, 0]);
        playerPosFRef.current = [-4.5, -2, 0];
        aiPosFRef.current = [4.5, -2, 0];
        setAiTargetF(null); // ✅ Clear AI target to prevent movement after reset
        setAiFightCmd(null); // ✅ Clear AI command
        if (import.meta.env.DEV) console.debug(`Round over! Player wins. Score: ${fightingRounds[0] + 1}-${fightingRounds[1]}`);
      }
      return newHealth;
    });
  };

  // Player attack detection for fighting
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (gameType === 'fighting' && (e.key.toLowerCase() === 'j' || e.key.toLowerCase() === 'k')) {
        const dx = aiPosF[0] - playerPosF[0];
        const dz = aiPosF[2] - playerPosF[2];
        const dist = Math.hypot(dx, dz);
        if (dist < 1.2) {
          const damage = e.key.toLowerCase() === 'k' ? 15 : 10;
          handleAITakeDamage(damage);
          if (import.meta.env.DEV) console.debug(`Player dealt ${damage} damage! AI health: ${aiHealth - damage}`);
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [gameType, aiPosF, playerPosF, aiHealth]);

  // ✅ CRITICAL: Periodic AI action requests with overlap prevention
  // Position refs are declared earlier (after state declarations)
  const playerHealthRef = useRef(playerHealth);
  const aiHealthRef = useRef(aiHealth);
  
  // ✅ FIX: Create handlers that update BOTH ref AND state immediately
  const handlePlayerPosChange = useCallback((newPos: [number, number, number]) => {
    playerPosFRef.current = newPos; // Update ref IMMEDIATELY (sync)
    setPlayerPosF(newPos);          // Also update state (async)
  }, []);
  
  const handleAiPosChange = useCallback((newPos: [number, number, number]) => {
    // Only update if position actually changed significantly (prevents flickering)
    const oldX = aiPosFRef.current[0];
    const oldZ = aiPosFRef.current[2];
    const moved = Math.abs(newPos[0] - oldX) > 0.02 || Math.abs(newPos[2] - oldZ) > 0.02;
    
    if (moved) {
      aiPosFRef.current = newPos; // Update ref IMMEDIATELY (sync) 
      setAiPosF(newPos);          // Also update state (async)
    }
  }, []);
  
  // Keep health refs in sync with state
  useEffect(() => { playerHealthRef.current = playerHealth; }, [playerHealth]);
  useEffect(() => { aiHealthRef.current = aiHealth; }, [aiHealth]);
  
  // ✅ CRITICAL: Keep position refs in sync with state (fixes stale position bug)
  useEffect(() => { playerPosFRef.current = playerPosF; }, [playerPosF]);
  useEffect(() => { aiPosFRef.current = aiPosF; }, [aiPosF]);
  
  useEffect(() => {
    if (!gameStarted || paused || gameType !== 'fighting') {
      return; // Silent return - no spam
    }
    
    // Reduced logging
    
    const requestAIAction = async () => {
      // STOP if a request is already in flight - prevent concurrent requests
      if (isFetchingAI.current) {
        return; // Silent skip - prevents lag from stacking requests
      }
      
      // Also check if game is still active
      if (!gameStarted || paused || gameType !== 'fighting') {
        return;
      }
      
      try {
        isFetchingAI.current = true; // Lock
        
        // ✅ FIX: Use refs to get CURRENT positions, not stale closure values
        const currentPlayerPos = playerPosFRef.current;
        const currentAiPos = aiPosFRef.current;
        const currentPlayerHealth = playerHealthRef.current;
        const currentAiHealth = aiHealthRef.current;
        
        // Calculate distance safely using current state
        const dist = Math.hypot(
          currentAiPos[0] - currentPlayerPos[0], 
          currentAiPos[2] - currentPlayerPos[2]
        );
        
        // ✅ Only poll when distance < 6 (combat range)
        if (dist > 6) {
          isFetchingAI.current = false;
          return;
        }
        
        // Logging removed for performance
        
        const response = await postFightingAction({
          action_type: 'ai_update',
          timestamp: Date.now(),
          position: [currentPlayerPos[0], currentPlayerPos[2]], // Send ACTUAL player position
          context: {
            player_health: currentPlayerHealth,
            ai_health: currentAiHealth,
            distance_to_opponent: isNaN(dist) ? 5 : dist,
            ai_position: { x: currentAiPos[0], z: currentAiPos[2] } // Also send AI position for context
          }
        });
        
        // Logging removed for performance
      } catch (e) {
        console.error('❌ AI action request failed:', e);
      } finally {
        isFetchingAI.current = false; // Unlock
      }
    };
    
    // Delay initial request slightly to let game state stabilize
    const initialTimeout = setTimeout(requestAIAction, 200);
    
    // ✅ OPTIMIZED: 600ms for responsive AI combat
    const intervalId = setInterval(requestAIAction, 300);  // 600ms → 300ms
    // Reduced logging
    
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(intervalId);
      // Reduced logging
    };
    
  }, [gameStarted, paused, gameType]); // Remove position/health deps to prevent recreating interval

  // FIXED: Auto-start badminton
  useEffect(() => {
    if (gameType === 'badminton') {
      setGameStarted(true);
      setPaused(false);
  if (import.meta.env.DEV) console.debug('Badminton auto-started');
    }
  }, [gameType]);

  // RACING AI - decision making and physics (adds competitive behavior)
  useEffect(() => {
    if (gameType !== 'racing' || !raceRunning || paused) return;

    const aiDecisionInterval = window.setInterval(() => {
      // AI Car 1 decisions
      const distToCar1 = Math.abs(playerCarPos[2] - aiCar1Pos[2]);
      const lateralDistToCar1 = Math.abs(playerCarPos[0] - aiCar1Pos[0]);

      if (distToCar1 < 5 && lateralDistToCar1 < 2) {
        if (aiCar1Pos[2] < playerCarPos[2]) {
          setAiCar1Strategy('defend');
          setAiCar1Speed(6.5);
        } else {
          setAiCar1Strategy('overtake');
          setAiCar1Speed(7.5);
        }
      } else {
        setAiCar1Strategy('racing');
        setAiCar1Speed(6 + Math.random() * 0.5);
      }

      // AI Car 2 decisions
      const distToCar2 = Math.abs(playerCarPos[2] - aiCar2Pos[2]);
      const lateralDistToCar2 = Math.abs(playerCarPos[0] - aiCar2Pos[0]);

      if (distToCar2 < 5 && lateralDistToCar2 < 2) {
        if (aiCar2Pos[2] < playerCarPos[2]) {
          setAiCar2Strategy('defend');
          setAiCar2Speed(6);
        } else {
          setAiCar2Strategy('overtake');
          setAiCar2Speed(7);
        }
      } else {
        setAiCar2Strategy('racing');
        setAiCar2Speed(5.5 + Math.random() * 0.5);
      }
    }, 1200 + Math.random() * 400);  // ✅ PERF FIX: Increased from 800ms to 1200ms

    return () => window.clearInterval(aiDecisionInterval);
  }, [gameType, raceRunning, paused, playerCarPos, aiCar1Pos, aiCar2Pos]);

  // AI physics loop for racing cars (runs on requestAnimationFrame to avoid useFrame outside Canvas)
  useEffect(() => {
    if (gameType !== 'racing' || !raceRunning || paused) return;

    let last = performance.now();
    let rafId: number;

    const loop = (ts: number) => {
      const delta = Math.max(0, (ts - last) / 1000);
      last = ts;

      // AI Car 1 physics using functional state update
      aiCar1VelRef.current += Math.sign(aiCar1Speed - aiCar1VelRef.current) * 3 * delta;
      setAiCar1Pos((prev) => {
        const current = prev;
        const targetX1 = aiCar1Strategy === 'overtake'
          ? (playerCarPos[0] > 0 ? -3 : 3)
          : (aiCar1Strategy === 'defend' ? playerCarPos[0] : Math.sin(current[2] * 0.05) * 2);
        const steering1 = (targetX1 - current[0]) * 0.2;
        const newX1 = current[0] + steering1 * delta * 10;
        const newZ1 = current[2] - aiCar1VelRef.current * delta;
        const clampedX1 = Math.max(-6, Math.min(6, newX1));
        const clampedZ1 = Math.max(-198, Math.min(198, newZ1));
        return [clampedX1, -1.75, clampedZ1];
      });

      // AI Car 2 physics
      aiCar2VelRef.current += Math.sign(aiCar2Speed - aiCar2VelRef.current) * 3 * delta;
      setAiCar2Pos((prev) => {
        const current = prev;
        const targetX2 = aiCar2Strategy === 'overtake'
          ? (playerCarPos[0] > 0 ? -2.5 : 2.5)
          : (aiCar2Strategy === 'defend' ? playerCarPos[0] : Math.cos(current[2] * 0.04) * 1.5);
        const steering2 = (targetX2 - current[0]) * 0.2;
        const newX2 = current[0] + steering2 * delta * 10;
        const newZ2 = current[2] - aiCar2VelRef.current * delta;
        const clampedX2 = Math.max(-6, Math.min(6, newX2));
        const clampedZ2 = Math.max(-198, Math.min(198, newZ2));
        return [clampedX2, -1.75, clampedZ2];
      });

      rafId = window.requestAnimationFrame(loop);
    };

    rafId = window.requestAnimationFrame(loop);
    return () => window.cancelAnimationFrame(rafId);
  }, [gameType, raceRunning, paused, playerCarPos, aiCar1Speed, aiCar2Speed, aiCar1Strategy, aiCar2Strategy]);

  // Racing countdown timer
  useEffect(() => {
    if (raceCountdown === null) return;
    if (raceCountdown > 0) {
      const t = setTimeout(() => setRaceCountdown((c) => (c === null ? null : c - 1)), 1000);
      return () => clearTimeout(t);
    } else {
      setRaceCountdown(null);
      setRaceRunning(true);
    }
  }, [raceCountdown]);

  // Auto-start racing on input if not started
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (gameType === 'racing' && (k === 'w' || k === 'a' || k === 's' || k === 'd')) {
        if (!gameStarted) { setGameStarted(true); setPaused(false); setRaceOver(null); setRaceCountdown(3); }
        else if (!raceRunning && raceCountdown === null) { setRaceOver(null); setRaceCountdown(3); }
      }
      if (gameType === 'fighting' && (k === 'w' || k === 'a' || k === 's' || k === 'd' || k === 'j' || k === 'k' || k === 'l')) {
        if (!gameStarted) { setGameStarted(true); setPaused(false); }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [gameStarted, gameType, raceRunning, raceCountdown]);

  // Racing proximity AI reactions
  useEffect(() => {
    if (!raceRunning) { setAiRaceCmd1(null); setAiRaceCmd2(null); return; }
    const collides = (a:[number,number,number], b:[number,number,number]) => {
      const dx = a[0]-b[0];
      const dz = a[2]-b[2];
      return Math.hypot(dx, dz) < 1.2;
    };
    if (collides(playerCarPos, aiCar1Pos) || collides(playerCarPos, aiCar2Pos)) {
      setRaceOver('ACCIDENT');
      setPaused(true);
      setRaceRunning(false);
    }
  }, [playerCarPos, aiCar1Pos, aiCar2Pos, raceRunning, raceOver]);

  // Badminton state
  const [shuttlePos, setShuttlePos] = useState<[number, number, number]>([0, 2.5, 0]);
  const [shuttleVel, setShuttleVel] = useState<[number, number, number]>([0,0,0]);
  const shuttlePrevRef = useRef<{pos:[number,number,number], t:number}|null>(null);
  const updateShuttleState = (p:[number,number,number]) => {
    const now = performance.now();
    const prev = shuttlePrevRef.current;
    if (prev) {
      const dt = Math.max(0.001, (now - prev.t)/1000);
      setShuttleVel([(p[0]-prev.pos[0])/dt, (p[1]-prev.pos[1])/dt, (p[2]-prev.pos[2])/dt]);
    }
    shuttlePrevRef.current = {pos: p, t: now};
    setShuttlePos(p);
  };

  // Badminton positional state needed by AI logic
  const [playerBadPos, setPlayerBadPos] = useState<[number, number, number]>([-5, 0, 0]);
  const [aiBadPos, setAiBadPos] = useState<[number, number, number]>([5, 0, 0]);
  const [aiTarget, setAiTarget] = useState<[number, number, number]>([5, 0, 0]);
  const [aiAction, setAiAction] = useState<any>(null);
  const [aiBadmintonOrder, setAiBadmintonOrder] = useState<{ target?: [number, number]; swing?: { dir:[number,number,number]; power:number; spin?:[number,number,number] } } | null>(null);
  const [playerShot, setPlayerShot] = useState<{ dir: [number, number, number]; power: number; spin?: [number,number,number] } | null>(null);

  // Badminton rally and scoring tracking
  useEffect(() => {
    const prev = shuttlePrevRef.current?.pos;
    if (prev) {
      const moved = Math.hypot(
        shuttlePos[0] - prev[0],
        shuttlePos[1] - prev[1],
        shuttlePos[2] - prev[2]
      );
      if (moved > 1.0 && shuttlePos[1] > 1.0) setRallyCount((c) => c + 1);
    }

    const justLanded = shuttlePos[1] <= 0.12 && (prev ? prev[1] > 0.12 : true);
    if (justLanded) {
      const landedOnPlayerSide = shuttlePos[0] < 0;
      const scorer = landedOnPlayerSide ? 'ai' : 'player';
      const newScore: [number, number] = scorer === 'player'
        ? [badmintonScore[0] + 1, badmintonScore[1]]
        : [badmintonScore[0], badmintonScore[1] + 1];
      setBadmintonScore(newScore);
      setRallyCount(0);
      const isGamePoint = Math.max(newScore[0], newScore[1]) >= 20;
      setGamePoint(isGamePoint ? (scorer as 'player' | 'ai') : null);
  if (import.meta.env.DEV) console.debug(`Point scored by ${scorer}! Score: ${newScore[0]}-${newScore[1]}`);
      
      // Post action for scoring point to ensure context update
      postBadmintonAction({ 
        game_type: 'badminton',
        action_type: 'point_scored',
        shot_type: 'drop', // Use a default for point scoring
        shuttlecock_target: [shuttlePos[0], shuttlePos[2]],
        power_level: 0.5,
        rally_position: rallyCount,
        court_position: [0,0],
        success: true,
        timestamp: Date.now(),
        context: {
          score_player: newScore[0],
          score_ai: newScore[1],
          rally_count: 0,
          court_position: { x: shuttlePos[0], y: shuttlePos[2] },
          shuttlecock_target: { x: shuttlePos[0], y: shuttlePos[2] }
        }
      });
    }
  }, [shuttlePos, badmintonScore, rallyCount]);

  // Badminton state (moved earlier so AI logic has access before use)
  

  // Badminton AI decision-making: predict landing and generate aiBadmintonOrder when appropriate
  useEffect(() => {
    if (gameType !== 'badminton') return;
    let mounted = true;
    const aiInterval = window.setInterval(() => {
      if (!mounted) return;

      // Predict shuttle landing (short horizon)
      const shuttleLandingX = shuttlePos[0] + shuttleVel[0] * 0.5;
      const shuttleLandingZ = shuttlePos[2] + shuttleVel[2] * 0.5;

      // Move AI target toward predicted landing if shuttle is on AI side
      if (shuttlePos[0] > 0) {
        const clampedX = Math.max(0.6, Math.min(7, shuttleLandingX));
        const clampedZ = Math.max(-2.8, Math.min(2.8, shuttleLandingZ));
        setAiTarget([clampedX, 0, clampedZ]);
        // If close enough and shuttle low, request a swing
        const distanceToShuttle = Math.hypot(shuttlePos[0] - aiBadPos[0], shuttlePos[2] - aiBadPos[2]);
        if (distanceToShuttle < 0.8 && shuttlePos[1] < 1.5 && shuttlePos[1] > 0.5) {
          // Compute direction toward player
          const directionToPlayer = [playerBadPos[0] - shuttlePos[0], 0.5, playerBadPos[2] - shuttlePos[2]];
          const magnitude = Math.hypot(directionToPlayer[0], directionToPlayer[2]) || 1;
          const normalized: [number, number, number] = [directionToPlayer[0] / magnitude, directionToPlayer[1], directionToPlayer[2] / magnitude];
          const action = { type: 'swing', direction: normalized, power: 0.6 + Math.random() * 0.3 };
          setAiAction(action);
          // Publish aiBadmintonOrder for BadmintonPlayer
          setAiBadmintonOrder({ target: [clampedX, clampedZ], swing: { dir: normalized, power: action.power } });
          setRallyCount((r) => r + 1);
          // Clear ai action shortly after
          setTimeout(() => {
            setAiAction(null);
          }, 120);
        } else {
          // If no swing, still update the AI target order so the AI player moves
          setAiBadmintonOrder({ target: [clampedX, clampedZ] });
        }
      } else {
        // Shuttle not on AI side: move toward center/ready
        setAiTarget([5, 0, 0]);
        setAiBadmintonOrder({ target: [5, 0] });
      }
    }, 100);

    return () => { mounted = false; window.clearInterval(aiInterval); };
  }, [gameType, shuttlePos, shuttleVel, aiBadPos, playerBadPos]);

  // Racing progress and position tracking
  useEffect(() => {
    const now = performance.now();
    const lastPos = lastCarPosRef.current;
    const dt = Math.max(0.001, (now - lastCarTsRef.current) / 1000);
    const dz = playerCarPos[2] - lastPos[2];
    const dx = playerCarPos[0] - lastPos[0];
    const speed = Math.sqrt(dx*dx + dz*dz) / dt;
    lastCarPosRef.current = playerCarPos;
    lastCarTsRef.current = now;

    const newDist = Math.max(totalDistance, Math.abs(playerCarPos[2]));
    if (newDist !== totalDistance) setTotalDistance(newDist);

    const computedLap = Math.min(3, Math.floor(newDist / 100) + 1);
    if (computedLap !== currentLap) {
      setCurrentLap(computedLap);
      setLapTimes((t) => [...t, Date.now()]);
  if (import.meta.env.DEV) console.debug(`Lap ${computedLap} completed!`);
    }

    const ranking = [
      { who: 'human', z: playerCarPos[2] },
      { who: 'ai1', z: aiCar1Pos[2] },
      { who: 'ai2', z: aiCar2Pos[2] },
    ].sort((a, b) => a.z - b.z);
    const newPos = ranking.findIndex((r) => r.who === 'human') + 1;
    if (newPos !== racePosition) {
      setRacePosition(newPos);
      if (import.meta.env.DEV) console.debug(`Position changed to ${newPos}!`);
    }
  }, [playerCarPos, aiCar1Pos, aiCar2Pos, currentLap, totalDistance, racePosition]);

  

  const renderGameContent = () => {
    switch (gameType) {
      case 'fighting':
        return (
          <>
            <FighterCharacter 
              position={playerPosF} 
              color="#00B3FF" 
              isPlayer 
              initialFacing={1} 
              engaged={gameStarted} 
              paused={paused} 
              opponentPosition={aiPosF} 
              onPositionChange={handlePlayerPosChange}
              onPlayerAttack={handleAITakeDamage}
              onBlockStateChange={setPlayerBlocking}
              playerCurrentHealth={playerHealth}
              aiCurrentHealth={aiHealth}
              postAction={postFightingActionTracked}
            />
            <FighterCharacter 
              position={aiPosF} 
              color="#FF4455" 
              initialFacing={-1} 
              engaged={gameStarted} 
              paused={paused} 
              opponentPosition={playerPosF} 
              onPositionChange={handleAiPosChange} 
              aiCommand={aiFightCmd}
              aiTargetPosition={aiTargetF}
              onPlayerAttack={handlePlayerTakeDamage}
              playerCurrentHealth={playerHealth}
              aiCurrentHealth={aiHealth}
              postAction={postFightingActionTracked}
            />
          </>
        );
      case 'badminton':
        // Defensive rendering: always render badminton game content if gameType is badminton
        return (
          <Physics gravity={[0, -9.81, 0]}>
            <RigidBody type="fixed">
              <CuboidCollider args={[60, 0.1, 60]} position={[0, -1.9, 0]} />
            </RigidBody>
            {import.meta.env.DEV && console.debug('Rendering BadmintonPlayer (player):', playerBadPos)}
            <BadmintonPlayer
              position={toPositionArray(playerBadPos)}
              color="#22D3EE"
              isPlayer
              paused={paused}
              followTarget={toPositionArray(shuttlePos)}
              followVel={shuttleVel}
              onPlayerHit={(dir, power, spin) => setPlayerShot({ dir: toPositionArray(dir), power, spin })}
              onPositionChange={p => setPlayerBadPos(toPositionArray(p))}
              postAction={postBadmintonActionTracked}
              rallyCount={rallyCount}
              badmintonScore={badmintonScore}
            />
            {import.meta.env.DEV && console.debug('Rendering BadmintonPlayer (ai):', aiBadmintonOrder?.target)}
            <BadmintonPlayer
              position={toPositionArray(aiBadmintonOrder?.target || [5, 0, 0])}
              color="#F97316"
              paused={paused}
              isAI={true} // <-- FIX: Ensures AI logic runs
              aiOrder={aiBadmintonOrder}
              onPositionChange={p => setAiBadPos(toPositionArray(p))}
              postAction={postBadmintonActionTracked}
              rallyCount={rallyCount}
              badmintonScore={badmintonScore}
            />
            {import.meta.env.DEV && console.debug('Rendering Shuttlecock:', playerBadPos)}
            <Shuttlecock
              paused={paused}
              aiShot={aiBadmintonShot}
              onPositionChange={p => updateShuttleState(toPositionArray(p))}
              playerHit={playerShot ? { ...playerShot, dir: toPositionArray(playerShot.dir) } : null}
              idleAnchor={toPositionArray(playerBadPos)}
              autoReturn={true}
              startPosition={toPositionArray(shuttlePos)}
            />
            <BadmintonAIController
              sessionId={sessionRef.current}
              enabled={wsEnabled}
              gameState={{
                score: badmintonScore,
                player: { pos: playerBadPos },
                ai: { pos: aiCar1Pos },
                shuttle: { pos: shuttlePos, vel: shuttleVel },
              }}
              onAIMove={setAiBadmintonOrder}
            />
          </Physics>
        );
      case 'racing':
        return (
          <>
            <RacingCar 
              position={[-2, -1.75, 0]} 
              color="#4ECDC4" 
              isPlayer 
              paused={paused || !!raceOver} 
              raceRunning={raceRunning} 
              onPositionUpdate={setPlayerCarPos} 
              postAction={postRacingActionTracked}
              currentLap={currentLap}
              racePosition={racePosition}
              racingTotalLaps={racingTotalLaps}
              totalDistance={totalDistance}
            />
            <RacingCar position={aiCar1Pos} color="#FF6B35" paused={paused || !!raceOver} raceRunning={raceRunning} aiCommand={aiCar1Strategy === 'overtake' ? 'overtake' : aiCar1Strategy === 'defend' ? 'block_overtake' : 'perfect_racing_line'} targetX={playerCarPos[0]} onPositionUpdate={setAiCar1Pos} postAction={postRacingActionTracked} currentLap={currentLap} racePosition={racePosition} racingTotalLaps={racingTotalLaps} totalDistance={totalDistance} />
            <RacingCar position={aiCar2Pos} color="#A855F7" paused={paused || !!raceOver} raceRunning={raceRunning} aiCommand={aiCar2Strategy === 'overtake' ? 'overtake' : aiCar2Strategy === 'defend' ? 'block_overtake' : 'perfect_racing_line'} targetX={playerCarPos[0]} onPositionUpdate={setAiCar2Pos} postAction={postRacingActionTracked} currentLap={currentLap} racePosition={racePosition} racingTotalLaps={racingTotalLaps} totalDistance={totalDistance} />
          </>
        );
      default:
        return null;
    }
  };

  const arenaRef = useRef<HTMLDivElement>(null);
  useEffect(() => { arenaRef.current?.focus(); }, []);

  return (
    <div ref={arenaRef} tabIndex={0} onClick={() => arenaRef.current?.focus()} className="relative w-full h-screen bg-background overflow-hidden">
      {/* Game Arena */}
      <div className="absolute inset-0">
        <Canvas
          dpr={[1, 1.25]}
          camera={{
            position: [6, 2, 6],
            fov: gameType === 'racing' ? 60 : 75,
            near: 0.1,
            far: 1000
          }}
          shadows={false}
          gl={{
            antialias: false,
            alpha: false,
            powerPreference: "high-performance"
            }}
            onCreated={(state) => {
              try {
                const canvas = state.gl.domElement as HTMLCanvasElement | null;
                if (!canvas) return;

                const handleContextLost = (ev: Event) => {
                  try {
                    ev.preventDefault();
                  } catch {}
                  // Log and attempt graceful fallback
                  if (import.meta.env.DEV) console.error('WebGL context lost');
                  // Optionally show a simple overlay or reload to attempt recovery
                  setTimeout(() => {
                    try { window.location.reload(); } catch {}
                  }, 1200);
                };

                const handleContextRestored = () => {
                  if (import.meta.env.DEV) console.info('WebGL context restored');
                  try { window.location.reload(); } catch {}
                };

                canvas.addEventListener('webglcontextlost', handleContextLost as EventListener);
                canvas.addEventListener('webglcontextrestored', handleContextRestored as EventListener);
              } catch (e) {
                if (import.meta.env.DEV) console.debug('onCreated attach listeners failed', e);
              }
            }}
        >
          <Suspense fallback={null}>
            <CameraController gameType={gameType} playerCarPos={playerCarPos} />
            <OrbitControls
            enabled={gameType !== 'racing'}
            enablePan={false}
            enableZoom={gameType !== 'racing'}
            minDistance={gameType === 'racing' ? 6 : 4}
            maxDistance={gameType === 'racing' ? 15 : 12}
            minPolarAngle={Math.PI / 12}
            maxPolarAngle={Math.PI / 2.5}
            minAzimuthAngle={gameType === 'fighting' ? -Math.PI / 3 : -Math.PI / 2}
            maxAzimuthAngle={gameType === 'fighting' ? Math.PI / 3 : Math.PI / 2}
            autoRotate={false}
            enableDamping={true}
            dampingFactor={0.1}
            rotateSpeed={0.5}
            zoomSpeed={0.6}
            target={[0, 0, 0]}
          />
          
          <ArenaEnvironment gameType={gameType} />
            {renderGameContent()}
          </Suspense>
        </Canvas>
      </div>

      {/* Game UI Overlay */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Top Score Bar */}
        <div className="absolute top-4 left-0 right-0 flex justify-center">
          {gameType === 'fighting' && (
            <ScoreBar game="fighting" playerHealth={playerHealth} aiHealth={aiHealth} rounds={fightingRounds} />
          )}
          {gameType === 'badminton' && (
            <ScoreBar game="badminton" score={badmintonScore} />
          )}
          {gameType === 'racing' && (
            <ScoreBar game="racing" lap={currentLap} totalLaps={racingTotalLaps} position={racePosition} totalRacers={racingTotalRacers} />
          )}
        </div>

        {/* Top HUD */}
        <div className="absolute top-20 left-4 right-4 flex justify-between items-start pointer-events-auto z-50">
          {/* Game Switcher */}
          <div className="flex gap-2">
            {(['fighting', 'badminton', 'racing'] as const).map((game) => (
              <motion.button
                key={game}
                onClick={() => onGameChange(game)}
                className={`px-4 py-2 rounded-lg font-bold uppercase tracking-wider text-sm transition-all duration-300 ${
                  gameType === game
                    ? 'btn-gaming'
                    : 'btn-gaming-outline'
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {game}
              </motion.button>
            ))}
          </div>

          {/* Right Controls */}
          <div className="flex gap-3 items-center">
            {/* Start/Pause Button (compact) */}
            <motion.button
              onClick={() => {
                if (!gameStarted) {
                  setGameStarted(true);
                  setPaused(false);
                  if (gameType === 'racing') { setRaceOver(null); setRaceRunning(false); setRaceCountdown(3); }
                } else {
                  setPaused(p => !p);
                }
              }}
              className="hud-element px-3 py-2 rounded-lg text-xs font-medium pointer-events-auto"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {!gameStarted ? 'Start' : paused ? 'Resume' : 'Pause'}
            </motion.button>

            {/* Analytics Button */}
            <motion.button
              onClick={onToggleAnalytics}
              className="hud-element px-3 py-2 rounded-lg text-xs font-medium pointer-events-auto"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Analytics
            </motion.button>

            {/* WebSocket status with reconnect info */}
            <div
              className={`px-3 py-2 rounded-lg text-xs font-medium border ${
                connected
                  ? 'bg-green-900/40 border-green-500/50 text-green-400'
                  : 'bg-yellow-900/40 border-yellow-500/50 text-yellow-400'
              }`}
            >
              {connected ? '🟢 Live Sync' : '🟡 AI Active (HTTP)'}
              {!connected && wsReconnectAttempts.current > 0 && (
                <span className="ml-2 text-xs opacity-70">
                  (WS retry {wsReconnectAttempts.current}/5)
                </span>
              )}
            </div>

            {/* AI Command Status */}
            {aiFightCmd && gameType === 'fighting' && (
              <div className="px-3 py-2 rounded-lg text-xs font-medium border bg-purple-900/40 border-purple-500/50 text-purple-400">
                🤖 AI: {aiFightCmd}
              </div>
            )}
          </div>
        </div>

        {/* 🎮 LIVE ML ACTION STATUS - For instructor demo */}
        {gameType === 'fighting' && (
          <div className="absolute top-24 left-4 bg-black/80 border border-cyan-500/50 rounded-lg p-4 z-40 min-w-[200px]">
            <div className="text-sm font-bold text-cyan-400 mb-3 flex items-center gap-2">
              🤖 ML-Powered Combat
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            </div>
            
            {/* Player Action */}
            <div className="mb-3 p-2 bg-blue-900/40 rounded border border-blue-500/30">
              <div className="text-[10px] text-blue-300 uppercase tracking-wider">Player</div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-blue-400">
                  {lastPlayerAction ? lastPlayerAction.action.toUpperCase() : 'IDLE'}
                </span>
                <span className={`text-xs px-2 py-0.5 rounded ${
                  playerStyle === 'Aggressive' ? 'bg-red-600/50 text-red-300' :
                  playerStyle === 'Defensive' ? 'bg-blue-600/50 text-blue-300' :
                  playerStyle === 'Evasive' ? 'bg-yellow-600/50 text-yellow-300' :
                  'bg-gray-600/50 text-gray-300'
                }`}>
                  {playerStyle}
                </span>
              </div>
            </div>
            
            {/* AI Action */}
            <div className="mb-3 p-2 bg-red-900/40 rounded border border-red-500/30">
              <div className="text-[10px] text-red-300 uppercase tracking-wider">AI Opponent</div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-red-400">
                  {lastAiAction ? lastAiAction.action.toUpperCase() : aiFightCmd?.toUpperCase() || 'ANALYZING'}
                </span>
                {lastAiAction && (
                  <span className="text-[10px] text-gray-400">
                    {(lastAiAction.confidence * 100).toFixed(0)}% conf
                  </span>
                )}
              </div>
            </div>
            
            {/* ML Stats */}
            <div className="border-t border-gray-700 pt-2 mt-2">
              <div className="flex justify-between text-[10px] text-gray-400">
                <span>Actions Tracked:</span>
                <span className="text-cyan-400 font-bold">{actionCount}</span>
              </div>
              <div className="flex justify-between text-[10px] text-gray-400">
                <span>ML Analysis:</span>
                <span className={lastPersonalityUpdate > 0 ? 'text-green-400' : 'text-yellow-400'}>
                  {lastPersonalityUpdate > 0 ? 'Active' : `In ${10 - (actionCount % 10)} actions`}
                </span>
              </div>
              <div className="mt-1 text-[9px] text-gray-500">
                Backend: localhost:8000 • Model: 1.4GB
              </div>
            </div>
          </div>
        )}

        {/* 🧠 AI Learning Status - For non-fighting games */}
        {gameType !== 'fighting' && (
          <div className="absolute top-24 right-4 bg-purple-900/40 border border-purple-500/50 rounded-lg p-3 z-40">
            <div className="text-xs text-purple-300 mb-1">
              {lastPersonalityUpdate > 0 ? '🧠 AI Learning Active' : '🎮 Action Tracker'}
            </div>
            {lastPersonalityUpdate > 0 && (
              <div className="text-xs text-gray-400">
                Last analysis: {Math.floor((Date.now() - lastPersonalityUpdate) / 1000)}s ago
              </div>
            )}
            <div className="text-xs text-gray-400">
              Actions: {actionCount} {actionCount > 0 && `(next analysis at ${Math.ceil(actionCount / 10) * 10})`}
            </div>
            <div className="text-[10px] text-gray-500 mt-1">
              {actionCount < 10 ? `${10 - actionCount} more for personality update` : '✨ Active Learning'}
            </div>
          </div>
        )}

        {/* Center overlays */}
        {gameType === 'racing' && (
          <div className="absolute inset-0 flex items-center justify-center">
            {raceCountdown !== null && (
              <div className="text-7xl font-extrabold text-white drop-shadow-lg">{raceCountdown}</div>
            )}
            {raceOver && (
              <div className="px-6 py-3 rounded-xl bg-black/70 border border-white/10 text-2xl font-bold text-red-400">
                {raceOver}
              </div>
            )}
          </div>
        )}

        {/* Game Controls Info */}
        <div className="absolute bottom-4 left-4 hud-element p-4 rounded-lg">
          <div className="text-sm font-medium text-foreground/80">
            <div className="text-primary font-bold mb-2 font-gaming tracking-wider">CONTROLS</div>
            {gameType === 'fighting' && (
              <>
                <div>WASD: Move</div>
                <div>J: Punch | K: Kick</div>
                <div>L: Block | SPACE: Jump</div>
              </>
            )}
            {gameType === 'badminton' && (
              <>
                <div>WASD: Move</div>
                <div>SPACE: Swing (Hold for power)</div>
                <div>Position for accuracy</div>
              </>
            )}
            {gameType === 'racing' && (
              <>
                <div>W: Accelerate | S: Brake</div>
                <div>A/D: Steer</div>
                <div>Stay on track for speed</div>
              </>
            )}
          </div>
        </div>

        {/* Game Status */}
        <div className="absolute bottom-4 right-4 hud-element p-4 rounded-lg">
          <div className="text-sm font-medium">
            <div className="text-primary font-bold font-gaming tracking-wider">STATUS</div>
            <div className="text-gaming-teal font-medium">
              {gameType === 'fighting' && 'COMBAT READY'}
              {gameType === 'badminton' && 'COURT ACTIVE'}  
              {gameType === 'racing' && 'ENGINES HOT'}
            </div>
          </div>
        </div>
      </div>

      {/* FIXED: Analytics Dashboard with real data */}
      {showAnalytics && (
        <div className="absolute top-0 left-0 w-full h-full z-50 bg-black/80 flex items-center justify-center">
          <div className="bg-gray-900 p-6 rounded-lg max-w-4xl max-h-[80vh] overflow-y-auto">
            <AnalyticsDashboard
              sessionId={sessionRef.current}
              analytics={analyticsState || {
                session_info: {
                  session_id: sessionRef.current,
                  total_actions: 0,
                  games_played: [gameType],
                  current_game: gameType
                },
                performance_metrics: {
                  fighting_health: playerHealth,
                  ai_health: aiHealth,
                  fighting_rounds: fightingRounds,
                  badminton_score: badmintonScore,
                  racing_position: racePosition,
                  racing_lap: currentLap
                }
              }} 
              personality={personalityState || {
                personality_type: "Analyzing...",
                traits: {
                  aggression: 0.5,
                  risk_tolerance: 0.5,
                  precision: 0.5,
                  patience: 0.5
                }
              }} 
              onRefresh={() => void fetchAnalytics()}
              lastUpdated={analyticsLastUpdated}
            
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default GameArena;
