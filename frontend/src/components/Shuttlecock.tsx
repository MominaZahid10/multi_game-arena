// src/components/Shuttlecock.tsx
import React, { useRef, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, Cone } from '@react-three/drei';
import * as THREE from 'three';

type ShotType = 'drop_shot' | 'smash' | 'clear' | 'net_shot' | null;
const Shuttlecock = ({ paused = false, aiShot = null, onPositionChange, playerHit = null, idleAnchor, autoReturn = false, startPosition = [0, 2.5, 0] }: { paused?: boolean; aiShot?: ShotType; onPositionChange?: (pos: [number, number, number]) => void; playerHit?: { dir: [number, number, number]; power: number; spin?: [number, number, number] } | null; idleAnchor?: [number, number, number]; autoReturn?: boolean; startPosition?: [number, number, number] }) => {
  
  // ====================================================================
  // FIX: All state variables must be initialized first.
  // Renamed initialPos -> startPosition to avoid R3F conflicts.
  // ====================================================================
  const [shuttlecockPosition, setShuttlecockPosition] = useState<[number, number, number]>(startPosition);
  const [shuttlecockVelocity, setShuttlecockVelocity] = useState<[number, number, number]>([0, 0, 0]);
  const [shuttlecockRotation, setShuttlecockRotation] = useState<[number, number, number]>([0, 0, 0]);
  const [isInPlay, setIsInPlay] = useState(false);
  const [spin, setSpin] = useState<[number, number, number]>([0, 0, 0]);
  const [lastHitTime, setLastHitTime] = useState(0);
  const [lastLanding, setLastLanding] = useState<{pos:[number,number,number]}|null>(null);
  
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
  
  // Defensive normalization for idleAnchor and playerHit
  const normIdleAnchor = idleAnchor ? safeArray3(idleAnchor, [0,0,0], 'Shuttlecock.idleAnchor') : [0,0,0];
  let normPlayerHit = null;
  if (playerHit && typeof playerHit === 'object') {
    normPlayerHit = {
      ...playerHit,
      dir: safeArray3(playerHit.dir, [0,0,0], 'Shuttlecock.playerHit.dir'),
      spin: playerHit.spin ? safeArray3(playerHit.spin, [0,0,0], 'Shuttlecock.playerHit.spin') : undefined
    };
  }
  
  // Use state variables for norm references:
  const normPosition = safeArray3(shuttlecockPosition, startPosition, 'Shuttlecock.position');
  const normRotation = safeArray3(shuttlecockRotation, [0,0,0], 'Shuttlecock.rotation');
  const normVelocity = safeArray3(shuttlecockVelocity, [0,0,0], 'Shuttlecock.velocity');
  
  try {
    if (import.meta.env.DEV) console.debug('Shuttlecock props:', JSON.stringify({ paused, aiShot, onPositionChange: !!onPositionChange, normPlayerHit, normIdleAnchor, autoReturn, normPosition, normRotation, normVelocity }));
  } catch (e) {
    if (import.meta.env.DEV) console.debug('Shuttlecock props (stringify failed):', { paused, aiShot, onPositionChange, normPlayerHit, normIdleAnchor, autoReturn, normPosition, normRotation, normVelocity }, e);
  }
  const shuttleRef = useRef<THREE.Group>(null);

  useFrame((state, delta) => {
    if (paused) return;
    if (shuttleRef.current && isInPlay) {
      // Enhanced physics simulation
      const [x, y, z] = shuttlecockPosition;
      const [vx, vy, vz] = shuttlecockVelocity;
      const [rx, ry, rz] = shuttlecockRotation;
      const [ox, oy, oz] = spin;

      // Physical constants
      const g = -12; // m/s^2
      const mass = 0.005; // kg (approx)
      const kDrag = 0.25; // drag coefficient
      const kMagnus = 0.02; // Magnus effect strength

      // Quadratic drag: Fd = -k * v * |v|
      const v = new THREE.Vector3(vx, vy, vz);
      const speed = v.length();
      const drag = v.clone().multiplyScalar(-kDrag * speed);

      // Magnus force: Fm = k * (omega x v)
      const omega = new THREE.Vector3(ox, oy, oz);
      const magnus = omega.clone().cross(v).multiplyScalar(kMagnus);

      // Sum forces
      const force = new THREE.Vector3(drag.x, drag.y + mass * g, drag.z).add(magnus);
      const accel = force.multiplyScalar(1 / mass);

      // Integrate velocity and position (semi-implicit Euler)
      const newV = v.add(accel.multiplyScalar(delta));
      let newX = x + newV.x * delta;
      let newY = y + newV.y * delta;
      let newZ = z + newV.z * delta;

      // Ground bounce
      if (newY <= 0.12) {
        newY = 0.12;
        newV.y = -newV.y * 0.35; // restitution
        newV.x *= 0.6; newV.z *= 0.6; // friction
        // Spin decay on bounce
        setSpin([ox * 0.7, oy * 0.7, oz * 0.7]);
      }
      
      const nextPos: [number, number, number] = [newX, newY, newZ];
      setShuttlecockPosition(nextPos);
      setShuttlecockVelocity([newV.x, newV.y, newV.z]);
      onPositionChange?.(nextPos);

      // Rotation follows velocity direction and spin
      const rotSpeed = newV.length() * 2.5;
      setShuttlecockRotation([
        rx + rotSpeed * delta + oy * 0.02,
        ry + (newV.x * delta * 1.5),
        rz + (newV.z * delta * 1.5)
      ]);

      shuttleRef.current.rotation.set(...shuttlecockRotation);
      
      // Net collision detection
      if (Math.abs(x) < 0.2 && y < 2.5 && y > 0.5) {
        const bounce = new THREE.Vector3(-v.x * 0.3, Math.abs(v.y) * 0.5, v.z * 0.8);
        setShuttlecockVelocity([bounce.x, bounce.y, bounce.z]);
        setLastHitTime(state.clock.elapsedTime);
      }
      
      // Court boundary detection
      if (Math.abs(x) > 7 || Math.abs(z) > 5) {
        setShuttlecockVelocity([vx * -0.4, Math.abs(vy) * 0.3, vz * -0.4]);
      }
      
      // Reset if hits ground or goes too far
      if (Math.abs(newX) > 10 || Math.abs(newZ) > 8 || (newY <= 0.12 && Math.abs(newV.y) < 0.5)) {
        setLastLanding({ pos: [newX, 0.12, newZ] });
        setTimeout(() => {
          setShuttlecockVelocity([0, 0, 0]);
          setShuttlecockRotation([0, 0, 0]);
          setSpin([0, 0, 0]);
          setIsInPlay(false);
        }, 400);
      }
    } else if (shuttleRef.current) {
      // Idle near player's racket when not in play
      if (idleAnchor) {
        const [ax, ay, az] = normIdleAnchor;
        const floatY = 0.95 + Math.sin(state.clock.elapsedTime * 0.8) * 0.04;
        const anchorPos: [number, number, number] = [ax + (ax > 0 ? -0.2 : 0.2), floatY, az];
        setShuttlecockPosition(anchorPos);
        onPositionChange?.(anchorPos);
      } else {
        const floatY = 2.0 + Math.sin(state.clock.elapsedTime * 0.8) * 0.08;
        shuttleRef.current.position.y = floatY;
      }
      shuttleRef.current.rotation.y = state.clock.elapsedTime * 0.5;
    }
  });

  // If landed on AI side, auto-pick and return after short delay
  useEffect(() => {
    if (!autoReturn) return;
    if (!isInPlay && lastLanding && lastLanding.pos[0] > 0.6) {
      const t = setTimeout(() => {
        setShuttlecockPosition([lastLanding.pos[0], 0.9, lastLanding.pos[2]]);
        setIsInPlay(true);
        setShuttlecockVelocity([-3 - Math.random()*2, 5, (Math.random()-0.5)*2]);
        setLastHitTime(Date.now());
      }, 700);
      return () => clearTimeout(t);
    }
  }, [isInPlay, lastLanding, autoReturn]);

  // AI shots are triggered via racket collision; ignore aiShot to prevent auto-launches
  useEffect(() => {}, [aiShot, isInPlay]);

  // Apply player hit impulses
  useEffect(() => {
    if (!playerHit) return;
    setIsInPlay(true);
    const [dx, dy, dz] = normPlayerHit.dir;
    const power = Math.max(0.2, Math.min(1.5, normPlayerHit.power));
    const vx = dx * power * 12;
    const vy = Math.max(2.8, dy * power * 6);
    const vz = dz * power * 2;
    setShuttlecockVelocity([vx, vy, vz]);
    setSpin(normPlayerHit.spin || [0, 0, 0]);
    setLastHitTime(Date.now());
  }, [playerHit]);

  // Keep aligned to idle anchor when provided
  useEffect(() => {
    if (!isInPlay && idleAnchor) {
      setShuttlecockPosition([normIdleAnchor[0] + (normIdleAnchor[0] > 0 ? -0.2 : 0.2), 0.95, normIdleAnchor[2]]);
    }
  }, [idleAnchor, isInPlay]);

  // Enhanced launch mechanics with power system
  useEffect(() => {
    let powerCharging = false;
    let chargePower = 0;
    
    const handleLaunchStart = (event: KeyboardEvent) => {
      if (event.key === ' ' && !isInPlay && !powerCharging) {
        powerCharging = true;
        chargePower = 0;
        
        const chargeInterval = setInterval(() => {
          chargePower = Math.min(chargePower + 0.02, 1);
        }, 16);
        
        const handleLaunchEnd = (endEvent: KeyboardEvent) => {
          if (endEvent.key === ' ') {
            clearInterval(chargeInterval);
            powerCharging = false;
            
            // Launch with accumulated power
            setIsInPlay(true);
            const power = 0.3 + chargePower * 0.7;
            const angle = Math.random() * Math.PI / 6 - Math.PI / 12; // ±15 degrees
            
            setShuttlecockVelocity([
              (Math.random() - 0.5) * 6 * power,
              (2 + power * 4) * Math.cos(angle),
              (Math.random() - 0.5) * 4 * power
            ]);
            
            setLastHitTime(Date.now());
            
            window.removeEventListener('keyup', handleLaunchEnd);
          }
        };
        
        window.addEventListener('keyup', handleLaunchEnd);
      }
    };

    window.addEventListener('keydown', handleLaunchStart);
    return () => window.removeEventListener('keydown', handleLaunchStart);
  }, [isInPlay]);

  return (
    <group ref={shuttleRef} position={normPosition} rotation={normRotation}>
      {/* Enhanced shuttlecock head */}
      <Sphere args={[0.06]} position={[0, 0, 0]}>
        <meshPhongMaterial 
          color="#F8F8F8" 
          shininess={100}
          specular="#FFFFFF"
        />
      </Sphere>
      
      {/* Cork base */}
      <Sphere args={[0.04]} position={[0, -0.04, 0]}>
        <meshPhongMaterial color="#8B4513" />
      </Sphere>
      
      {/* Realistic feathers with better arrangement */}
      {Array.from({ length: 16 }, (_, i) => {
        const angle = (i / 16) * Math.PI * 2;
        const radius = i % 2 === 0 ? 0.045 : 0.038;
        const height = i % 2 === 0 ? 0.25 : 0.22;
        
        return (
          <Cone
            key={i}
            args={[0.015, height]}
            position={[
              Math.cos(angle) * radius,
              -height / 2 - 0.06,
              Math.sin(angle) * radius
            ]}
            rotation={[Math.PI + (Math.random() - 0.5) * 0.1, 0, angle]}
          >
            <meshPhongMaterial 
              color="#FFFFFF" 
              transparent 
              opacity={0.95}
              side={THREE.DoubleSide}
            />
          </Cone>
        );
      })}
      
      {/* Speed trail when moving fast */}
      {isInPlay && normVelocity && Math.sqrt(normVelocity[0]**2 + normVelocity[1]**2 + normVelocity[2]**2) > 2 && (
        <>
          <Sphere args={[0.12]} position={[0, 0, 0]}>
            <meshBasicMaterial color="#4ECDC4" transparent opacity={0.2} />
          </Sphere>
          <Sphere args={[0.08]} position={[-normVelocity[0] * 0.1, -normVelocity[1] * 0.1, -normVelocity[2] * 0.1]}>
            <meshBasicMaterial color="#A855F7" transparent opacity={0.15} />
          </Sphere>
        </>
      )}
      
      {/* Hit effect */}
      {Date.now() - lastHitTime < 500 && (
        <Sphere args={[0.2]} position={[0, 0, 0]}>
          <meshBasicMaterial 
            color="#FFD700" 
            transparent 
            opacity={Math.max(0, 0.5 - (Date.now() - lastHitTime) / 1000)} 
          />
        </Sphere>
      )}
    </group>
  );
};

export default Shuttlecock;