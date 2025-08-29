import React, { useRef, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, Cone } from '@react-three/drei';
import * as THREE from 'three';

const Shuttlecock = ({ paused = false }: { paused?: boolean }) => {
  const shuttleRef = useRef<THREE.Group>(null);
  const [position, setPosition] = useState<[number, number, number]>([0, 2.5, 0]);
  const [velocity, setVelocity] = useState<[number, number, number]>([0, 0, 0]);
  const [isInPlay, setIsInPlay] = useState(false);
  const [rotation, setRotation] = useState<[number, number, number]>([0, 0, 0]);
  const [lastHitTime, setLastHitTime] = useState(0);

  useFrame((state, delta) => {
    if (paused) return;
    if (shuttleRef.current && isInPlay) {
      // Enhanced physics simulation
      const [x, y, z] = position;
      const [vx, vy, vz] = velocity;
      const [rx, ry, rz] = rotation;
      
      // Realistic badminton physics
      const gravity = -12 * delta; // Stronger gravity for shuttlecock
      const airResistanceXZ = 0.96; // Higher air resistance for shuttlecock
      const airResistanceY = 0.98;
      
      // Shuttlecock-specific aerodynamics
      const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
      const dragFactor = 1 - (speed * 0.02 * delta);
      
      // Update velocity with enhanced physics
      const newVy = (vy + gravity) * airResistanceY;
      const newVx = vx * airResistanceXZ * dragFactor;
      const newVz = vz * airResistanceXZ * dragFactor;
      
      // Update position
      const newX = x + newVx * delta;
      const newY = Math.max(0.12, y + newVy * delta);
      const newZ = z + newVz * delta;
      
      setPosition([newX, newY, newZ]);
      setVelocity([newVx, newVy, newVz]);
      
      // Realistic rotation based on velocity
      const rotSpeed = speed * 3;
      setRotation([
        rx + rotSpeed * delta,
        ry + (vx * delta * 2),
        rz + (vz * delta * 2)
      ]);
      
      shuttleRef.current.rotation.set(...rotation);
      
      // Net collision detection
      if (Math.abs(x) < 0.2 && y < 2.5 && y > 0.5) {
        setVelocity([-vx * 0.3, Math.abs(vy) * 0.5, vz * 0.8]);
        setLastHitTime(state.clock.elapsedTime);
      }
      
      // Court boundary detection
      if (Math.abs(x) > 7 || Math.abs(z) > 5) {
        setVelocity([vx * -0.4, Math.abs(vy) * 0.3, vz * -0.4]);
      }
      
      // Reset if hits ground or goes too far
      if (newY <= 0.12 || Math.abs(newX) > 10 || Math.abs(newZ) > 8) {
        setTimeout(() => {
          setPosition([0, 2.5, 0]);
          setVelocity([0, 0, 0]);
          setRotation([0, 0, 0]);
          setIsInPlay(false);
        }, 800);
      }
    } else if (shuttleRef.current) {
      // Gentle floating when not in play
      const floatY = 2.5 + Math.sin(state.clock.elapsedTime * 0.8) * 0.08;
      shuttleRef.current.position.y = floatY;
      shuttleRef.current.rotation.y = state.clock.elapsedTime * 0.5;
    }
  });

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
            
            setVelocity([
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
    <group ref={shuttleRef} position={position}>
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
      {isInPlay && velocity && Math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2) > 2 && (
        <>
          <Sphere args={[0.12]} position={[0, 0, 0]}>
            <meshBasicMaterial color="#4ECDC4" transparent opacity={0.2} />
          </Sphere>
          <Sphere args={[0.08]} position={[-velocity[0] * 0.1, -velocity[1] * 0.1, -velocity[2] * 0.1]}>
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
