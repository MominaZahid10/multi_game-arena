import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Box, Sphere, Plane } from '@react-three/drei';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import Shuttlecock from './Shuttlecock';
import ScoreBar from './ScoreBar';

interface GameArenaProps {
  gameType: 'fighting' | 'badminton' | 'racing';
  onGameChange: (game: 'fighting' | 'badminton' | 'racing') => void;
  showAnalytics: boolean;
  onToggleAnalytics: () => void;
}

// Fighter Character Component - Realistic with animations
const FighterCharacter = ({ position, color, isPlayer = false, initialFacing = 1, engaged = false, paused = false }: { position: [number, number, number], color: string, isPlayer?: boolean, initialFacing?: -1 | 1, engaged?: boolean, paused?: boolean }) => {
  const meshRef = useRef<THREE.Group>(null);
  const leftArmRef = useRef<THREE.Mesh>(null);
  const rightArmRef = useRef<THREE.Mesh>(null);
  const leftLegRef = useRef<THREE.Mesh>(null);
  const rightLegRef = useRef<THREE.Mesh>(null);
  const bodyRef = useRef<THREE.Mesh>(null);
  const [isAttacking, setIsAttacking] = useState(false);
  const [isWalking, setIsWalking] = useState(false);
  const [isBlocking, setIsBlocking] = useState(false);
  const [position2D, setPosition2D] = useState(position);
  const [facingDirection, setFacingDirection] = useState<number>(initialFacing);

  useFrame((state, delta) => {
    if (paused) return;
    if (meshRef.current) {
      // Realistic idle animation - breathing and slight movement
      if (!isAttacking && !isWalking) {
        meshRef.current.position.y = position2D[1] + Math.sin(state.clock.elapsedTime * 1.5) * 0.02;

        // Subtle arm sway during idle
        if (leftArmRef.current) {
          leftArmRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.8) * 0.05;
          leftArmRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.6) * 0.03;
        }
        if (rightArmRef.current) {
          rightArmRef.current.rotation.z = -Math.sin(state.clock.elapsedTime * 0.8) * 0.05;
          rightArmRef.current.rotation.x = -Math.sin(state.clock.elapsedTime * 0.6) * 0.03;
        }

        // Body breathing animation
        if (bodyRef.current) {
          bodyRef.current.scale.y = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.02;
        }
      }

      // Walking animation
      if (isWalking && !isAttacking) {
        const walkCycle = state.clock.elapsedTime * 8;

        if (leftLegRef.current && rightLegRef.current) {
          leftLegRef.current.rotation.x = Math.sin(walkCycle) * 0.4;
          rightLegRef.current.rotation.x = Math.sin(walkCycle + Math.PI) * 0.4;
        }

        if (leftArmRef.current && rightArmRef.current) {
          leftArmRef.current.rotation.x = Math.sin(walkCycle + Math.PI) * 0.3;
          rightArmRef.current.rotation.x = Math.sin(walkCycle) * 0.3;
        }

        // Walking bob
        meshRef.current.position.y = position2D[1] + Math.abs(Math.sin(walkCycle * 2)) * 0.05;
      }

      // Always face opponent properly
      if (meshRef.current) {
        meshRef.current.rotation.set(0, facingDirection < 0 ? Math.PI : 0, 0);
      }
    }
  });

  // When engagement starts, orient toward each other (player faces right, opponent faces left)
  useEffect(() => {
    if (engaged) {
      setFacingDirection(isPlayer ? 1 : -1);
    }
  }, [engaged, isPlayer]);

  const performAttack = (attackType: 'punch' | 'kick' = 'punch') => {
    if (!meshRef.current || isAttacking) return;

    setIsAttacking(true);
    const originalX = position2D[0];

    // Send attack action
    import('@/lib/analytics').then(({ addAction }) => {
      const dist = Math.abs(originalX - (-originalX));
      addAction({
        game_type: 'fighting',
        action_type: 'attack',
        timestamp: Date.now(),
        success: true,
        move_type: 'attack',
        position: [position2D[0], position2D[2] || 0],
        damage_dealt: attackType === 'kick' ? 15 : 8,
        combo_count: 1,
        context: { player_health: 100, ai_health: 100, distance_to_opponent: dist },
      });
    });

    if (attackType === 'punch') {
      // Realistic punch animation with full body movement
      if (rightArmRef.current && bodyRef.current) {
        rightArmRef.current.rotation.x = -Math.PI / 2;
        rightArmRef.current.rotation.z = facingDirection * -0.3;
        bodyRef.current.rotation.y = facingDirection * -0.1;
      }

      // Forward lunge with proper spacing
      setPosition2D([originalX + (facingDirection * 0.3), position2D[1], position2D[2]]);

      setTimeout(() => {
        if (rightArmRef.current && bodyRef.current) {
          rightArmRef.current.rotation.x = 0;
          rightArmRef.current.rotation.z = 0;
          bodyRef.current.rotation.y = 0;
        }
        setPosition2D([originalX, position2D[1], position2D[2]]);
        setIsAttacking(false);
      }, 400);
    } else if (attackType === 'kick') {
      // Kick animation
      if (rightLegRef.current && bodyRef.current) {
        rightLegRef.current.rotation.x = Math.PI / 3;
        bodyRef.current.rotation.y = facingDirection * -0.15;
      }

      setPosition2D([originalX + (facingDirection * 0.4), position2D[1], position2D[2]]);

      setTimeout(() => {
        if (rightLegRef.current && bodyRef.current) {
          rightLegRef.current.rotation.x = 0;
          bodyRef.current.rotation.y = 0;
        }
        setPosition2D([originalX, position2D[1], position2D[2]]);
        setIsAttacking(false);
      }, 500);
    }
  };

  // Enhanced movement with WASD
  useEffect(() => {
    if (!isPlayer) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (isAttacking) return;

      const moveSpeed = 0.08;

      const pushMove = () => {
        import('@/lib/analytics').then(({ addAction }) => {
          const x = position2D[0];
          const dist = Math.abs(x - (-x));
          addAction({
            game_type: 'fighting',
            action_type: 'move',
            timestamp: Date.now(),
            success: true,
            move_type: 'move',
            position: [position2D[0], position2D[2] || 0],
            combo_count: 0,
            context: { player_health: 100, ai_health: 100, distance_to_opponent: dist },
          });
        });
      };

      switch (event.key.toLowerCase()) {
        case 'w':
          setPosition2D(prev => [prev[0], prev[1], Math.max(-4, prev[2] - moveSpeed)]);
          setIsWalking(true);
          pushMove();
          break;
        case 's':
          setPosition2D(prev => [prev[0], prev[1], Math.min(4, prev[2] + moveSpeed)]);
          setIsWalking(true);
          pushMove();
          break;
        case 'a':
          setPosition2D(prev => [Math.max(-6, prev[0] - moveSpeed), prev[1], prev[2]]);
          setFacingDirection(-1);
          setIsWalking(true);
          pushMove();
          break;
        case 'd':
          setPosition2D(prev => [Math.min(6, prev[0] + moveSpeed), prev[1], prev[2]]);
          setFacingDirection(1);
          setIsWalking(true);
          pushMove();
          break;
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      switch (event.key.toLowerCase()) {
        case 'w':
        case 's':
        case 'a':
        case 'd':
          setIsWalking(false);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [isPlayer, isAttacking, position2D]);

  // Combat controls
  useEffect(() => {
    if (!isPlayer) return;

    const handleCombatKeys = (event: KeyboardEvent) => {
      switch (event.key.toLowerCase()) {
        case 'j':
          performAttack('punch');
          break;
        case 'k':
          performAttack('kick');
          break;
        case 'l':
          // Block animation
          if (!isAttacking && !isBlocking) {
            setIsBlocking(true);
            if (leftArmRef.current && rightArmRef.current) {
              leftArmRef.current.rotation.x = -Math.PI / 4;
              rightArmRef.current.rotation.x = -Math.PI / 4;
              leftArmRef.current.position.z = facingDirection * 0.2;
              rightArmRef.current.position.z = facingDirection * 0.2;
            }
            import('@/lib/analytics').then(({ addAction }) => {
              const x = position2D[0];
              const dist = Math.abs(x - (-x));
              addAction({
                game_type: 'fighting',
                action_type: 'block',
                timestamp: Date.now(),
                success: true,
                move_type: 'block',
                position: [position2D[0], position2D[2] || 0],
                combo_count: 0,
                context: { player_health: 100, ai_health: 100, distance_to_opponent: dist },
              });
            });
            setTimeout(() => {
              if (leftArmRef.current && rightArmRef.current) {
                leftArmRef.current.rotation.x = 0;
                rightArmRef.current.rotation.x = 0;
                leftArmRef.current.position.z = 0;
                rightArmRef.current.position.z = 0;
              }
              setIsBlocking(false);
            }, 800);
          }
          break;
        case ' ':
          // Jump animation with realistic arc
          if (meshRef.current && !isAttacking && !isBlocking) {
            const originalY = position2D[1];
            let jumpHeight = 0;
            const jumpDuration = 600;
            const startTime = Date.now();

            const animateJump = () => {
              const elapsed = Date.now() - startTime;
              const progress = Math.min(elapsed / jumpDuration, 1);

              // Parabolic jump curve
              jumpHeight = Math.sin(progress * Math.PI) * 0.8;
              setPosition2D(prev => [prev[0], originalY + jumpHeight, prev[2]]);

              if (progress < 1) {
                requestAnimationFrame(animateJump);
              }
            };

            animateJump();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleCombatKeys);
    return () => window.removeEventListener('keydown', handleCombatKeys);
  }, [isPlayer, isAttacking, isBlocking, position2D, facingDirection]);

  return (
    <group ref={meshRef} position={position2D}>
      {/* Body */}
      <Box ref={bodyRef} args={[0.4, 1.1, 0.28]} position={[0, 0.1, 0]}>
        <meshPhongMaterial color={color} shininess={80} />
      </Box>

      {/* Head */}
      <Sphere args={[0.16]} position={[0, 0.8, 0]}>
        <meshPhongMaterial color="#f2dcc5" />
      </Sphere>
      {/* Eyes */}
      <Sphere args={[0.02]} position={[-0.04, 0.82, 0.14]}>
        <meshBasicMaterial color="#111" />
      </Sphere>
      <Sphere args={[0.02]} position={[0.04, 0.82, 0.14]}>
        <meshBasicMaterial color="#111" />
      </Sphere>

      {/* Shoulders */}
      <Sphere args={[0.12]} position={[-0.32, 0.5, 0]}>
        <meshPhongMaterial color={color} />
      </Sphere>
      <group ref={leftArmRef} position={[-0.35, 0.1, 0]}>
        {/* Upper arm */}
        <mesh position={[0, 0.25, 0]}>
          <cylinderGeometry args={[0.07, 0.07, 0.5, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        {/* Elbow */}
        <Sphere args={[0.07]} position={[0, -0.02, 0]}>
          <meshPhongMaterial color="#f2dcc5" />
        </Sphere>
        {/* Forearm */}
        <mesh position={[0, -0.35, 0]}>
          <cylinderGeometry args={[0.06, 0.06, 0.5, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        {/* Hand */}
        <Box args={[0.12, 0.08, 0.14]} position={[0, -0.62, 0]}>
          <meshPhongMaterial color="#f2dcc5" />
        </Box>
      </group>

      <Sphere args={[0.12]} position={[0.32, 0.5, 0]}>
        <meshPhongMaterial color={color} />
      </Sphere>
      <group ref={rightArmRef} position={[0.35, 0.1, 0]}>
        <mesh position={[0, 0.25, 0]}>
          <cylinderGeometry args={[0.07, 0.07, 0.5, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        <Sphere args={[0.07]} position={[0, -0.02, 0]}>
          <meshPhongMaterial color="#f2dcc5" />
        </Sphere>
        <mesh position={[0, -0.35, 0]}>
          <cylinderGeometry args={[0.06, 0.06, 0.5, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        <Box args={[0.12, 0.08, 0.14]} position={[0, -0.62, 0]}>
          <meshPhongMaterial color="#f2dcc5" />
        </Box>
      </group>

      {/* Hips */}
      <Sphere args={[0.10]} position={[-0.15, -0.6, 0]}>
        <meshPhongMaterial color={color} />
      </Sphere>
      <group ref={leftLegRef} position={[-0.15, -1.0, 0]}>
        {/* Upper leg */}
        <mesh position={[0, 0.28, 0]}>
          <cylinderGeometry args={[0.08, 0.08, 0.6, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        {/* Knee */}
        <Sphere args={[0.08]} position={[0, -0.05, 0]}>
          <meshPhongMaterial color="#f2dcc5" />
        </Sphere>
        {/* Lower leg */}
        <mesh position={[0, -0.42, 0]}>
          <cylinderGeometry args={[0.07, 0.07, 0.55, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        {/* Shoe */}
        <Box args={[0.2, 0.1, 0.3]} position={[0, -0.75, 0.05]}>
          <meshPhongMaterial color="#e5e7eb" />
        </Box>
      </group>

      <Sphere args={[0.10]} position={[0.15, -0.6, 0]}>
        <meshPhongMaterial color={color} />
      </Sphere>
      <group ref={rightLegRef} position={[0.15, -1.0, 0]}>
        <mesh position={[0, 0.28, 0]}>
          <cylinderGeometry args={[0.08, 0.08, 0.6, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        <Sphere args={[0.08]} position={[0, -0.05, 0]}>
          <meshPhongMaterial color="#f2dcc5" />
        </Sphere>
        <mesh position={[0, -0.42, 0]}>
          <cylinderGeometry args={[0.07, 0.07, 0.55, 12]} />
          <meshPhongMaterial color={color} />
        </mesh>
        <Box args={[0.2, 0.1, 0.3]} position={[0, -0.75, 0.05]}>
          <meshPhongMaterial color="#e5e7eb" />
        </Box>
      </group>

      {/* Suit accent */}
      <Box args={[0.3, 0.4, 0.05]} position={[0, 0.2, 0.18]}>
        <meshPhongMaterial color="#3ddbd9" />
      </Box>
    </group>
  );
};

// Badminton Player Component - Realistic with animations
const BadmintonPlayer = ({ position, color, isPlayer = false, paused = false }: { position: [number, number, number], color: string, isPlayer?: boolean, paused?: boolean }) => {
  const groupRef = useRef<THREE.Group>(null);
  const racketRef = useRef<THREE.Group>(null);
  const bodyRef = useRef<THREE.Mesh>(null);
  const leftArmRef = useRef<THREE.Mesh>(null);
  const rightArmRef = useRef<THREE.Mesh>(null);
  const leftLegRef = useRef<THREE.Mesh>(null);
  const rightLegRef = useRef<THREE.Mesh>(null);
  const [playerPos, setPlayerPos] = useState(position);
  const [isSwinging, setIsSwinging] = useState(false);
  const [isMoving, setIsMoving] = useState(false);
  const [racketPower, setRacketPower] = useState(0);
  const [facingDirection, setFacingDirection] = useState<number>(position[0] > 0 ? -1 : 1);

  useFrame((state, delta) => {
    if (paused) return;
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

      // Face the net properly
      if (groupRef.current) {
        groupRef.current.rotation.set(0, facingDirection > 0 ? -Math.PI / 2 : Math.PI / 2, 0);
      }
    }
  });

  // Enhanced player movement for badminton
  useEffect(() => {
    if (!isPlayer) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (paused || isSwinging) return;

      const moveSpeed = 0.12;
      const pushMove = () => {
        import('@/lib/analytics').then(({ addAction }) => {
          addAction({
            game_type: 'badminton',
            action_type: 'move',
            timestamp: Date.now(),
            success: true,
            shot_type: 'drive',
            court_position: [playerPos[0], playerPos[2] || 0],
            shuttlecock_target: [0, 0],
            power_level: 0,
            rally_position: 0,
            context: { rally_count: 0, court_side: playerPos[2] > 0 ? 'right' : 'left', game_score: [0, 0] },
          });
        });
      };
      switch (event.key.toLowerCase()) {
        case 'w':
          setPlayerPos(prev => [prev[0], prev[1], Math.max(-5, prev[2] - moveSpeed)]);
          setIsMoving(true);
          pushMove();
          break;
        case 's':
          setPlayerPos(prev => [prev[0], prev[1], Math.min(5, prev[2] + moveSpeed)]);
          setIsMoving(true);
          pushMove();
          break;
        case 'a':
          setPlayerPos(prev => [Math.max(-7, prev[0] - moveSpeed), prev[1], prev[2]]);
          setIsMoving(true);
          pushMove();
          break;
        case 'd':
          setPlayerPos(prev => [Math.min(7, prev[0] + moveSpeed), prev[1], prev[2]]);
          setIsMoving(true);
          pushMove();
          break;
        case ' ':
          // Power swing - hold for more power
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
        case 'a':
        case 'd':
          setIsMoving(false);
          break;
        case ' ':
          // Release swing with accumulated power
          if (!isSwinging) {
            performSwing(racketPower);
            setRacketPower(0);
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [isPlayer, isSwinging, racketPower]);

  const performSwing = (power: number = 0.5) => {
    if (isSwinging || !racketRef.current) return;

    setIsSwinging(true);

    // Send badminton shot action (approximate as drive with power)
    import('@/lib/analytics').then(({ addAction }) => {
      addAction({
        game_type: 'badminton',
        action_type: 'shot',
        timestamp: Date.now(),
        success: true,
        shot_type: 'drive',
        court_position: [playerPos[0], playerPos[2] || 0],
        shuttlecock_target: [0, 0],
        power_level: Math.max(0, Math.min(1, power)),
        rally_position: 1,
        context: { rally_count: 1, court_side: playerPos[2] > 0 ? 'right' : 'left', game_score: [0, 0] },
      });
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

      {/* Hips and legs */}
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

// Enhanced Racing Car Component
const RacingCar = ({ position, color, isPlayer = false, paused = false }: { position: [number, number, number], color: string, isPlayer?: boolean, paused?: boolean }) => {
  const carRef = useRef<THREE.Group>(null);
  const wheelRefs = useRef<THREE.Mesh[]>([]);
  const [carPosition, setCarPosition] = useState(position);
  const [velocity, setVelocity] = useState(0);
  const [steering, setSteering] = useState(0);
  const [isAccelerating, setIsAccelerating] = useState(false);
  const [isBraking, setIsBraking] = useState(false);

  useFrame((state, delta) => {
    if (paused || !carRef.current) return;
    if (carRef.current) {
      // Realistic car physics
      let newVelocity = velocity;

      if (isAccelerating) {
        newVelocity = Math.min(velocity + delta * 3, 8);
      } else if (isBraking) {
        newVelocity = Math.max(velocity - delta * 5, -2);
      } else {
        // Natural deceleration
        newVelocity = velocity * 0.98;
      }

      setVelocity(newVelocity);

      // Update position based on velocity and steering
      const newX = carPosition[0] + Math.sin(steering) * newVelocity * delta;
      const newZ = carPosition[2] + Math.cos(steering) * newVelocity * delta;

      // Keep car on track
      const clampedX = Math.max(-6, Math.min(6, newX));
      const clampedZ = Math.max(-15, Math.min(15, newZ));

      setCarPosition([clampedX, -1.75, clampedZ]); // Car properly on ground

      // Car rotation based on steering
      carRef.current.rotation.y = steering;

      // Wheel rotation based on speed
      wheelRefs.current.forEach((wheel) => {
        if (wheel) {
          wheel.rotation.x += newVelocity * delta * 2;
        }
      });

      // Engine vibration when accelerating
      if (isAccelerating && Math.abs(newVelocity) > 0.1) {
        carRef.current.position.y = carPosition[1] + Math.sin(state.clock.elapsedTime * 30) * 0.005;
      } else {
        carRef.current.position.y = carPosition[1];
      }
    }
  });

  // Enhanced car controls
  useEffect(() => {
    if (!isPlayer) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (paused) return;
      const send = (type: 'accelerate' | 'brake' | 'steer') => {
        import('@/lib/analytics').then(({ addAction }) => {
          addAction({
            game_type: 'racing',
            action_type: type,
            timestamp: Date.now(),
            success: true,
            speed: Math.max(0, velocity),
            position_on_track: [carPosition[0], carPosition[2]],
            overtaking_attempt: false,
            crash_occurred: false,
            context: { lap_number: 1, position_in_race: 1, distance_to_finish: Math.max(0, 100 - Math.abs(carPosition[2]) * 3) },
          });
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
          setSteering(prev => Math.max(prev - 0.05, -0.5));
          send('steer');
          break;
        case 'd':
          setSteering(prev => Math.min(prev + 0.05, 0.5));
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
        case 'd':
          setSteering(0);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [isPlayer]);

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

      {/* Speed effect when moving fast */}
      {Math.abs(velocity) > 3 && (
        <>
          <Sphere args={[0.3]} position={[0, 0, -1]}>
            <meshBasicMaterial color="#4ECDC4" transparent opacity={0.2} />
          </Sphere>
          <Sphere args={[0.2]} position={[0, 0, -1.5]}>
            <meshBasicMaterial color="#A855F7" transparent opacity={0.15} />
          </Sphere>
        </>
      )}
    </group>
  );
};

// Arena Environment Component
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

          {/* Background arena walls with glow */}
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
          {/* Large soft glows */}
          <Sphere args={[10]} position={[0, 3, -12]}>
            <meshBasicMaterial color="#1e90ff" transparent opacity={0.06} />
          </Sphere>
          <Sphere args={[7]} position={[-8, 4, 8]}>
            <meshBasicMaterial color="#a78bfa" transparent opacity={0.04} />
          </Sphere>
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

          {/* Background stadium */}
          <Plane args={[40, 15]} position={[0, 5, -15]}>
            <meshBasicMaterial color="#1A4B3A" />
          </Plane>

          {/* Stadium seating */}
          {Array.from({ length: 6 }, (_, i) => (
            <Box key={i} args={[8, 2, 1]} position={[-15 + i * 6, 2 + i * 0.5, -12]}>
              <meshPhongMaterial color="#4A4A4A" />
            </Box>
          ))}

          {/* Stadium lights */}
          <Box args={[0.3, 8, 0.3]} position={[-10, 4, -8]}>
            <meshPhongMaterial color="#C0C0C0" />
          </Box>
          <Box args={[0.3, 8, 0.3]} position={[10, 4, -8]}>
            <meshPhongMaterial color="#C0C0C0" />
          </Box>
        </>
      )}
      
      {/* Night Racing Circuit with Enhanced Atmosphere */}
      {gameType === 'racing' && (
        <>
          {/* Main track surface - darker for night */}
          <Plane args={[15, 40]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.85, 0]}>
            <meshPhongMaterial color="#1A1A1A" roughness={0.8} />
          </Plane>

          {/* Illuminated track borders */}
          <Box args={[0.3, 0.2, 40]} position={[-7.5, -1.75, 0]}>
            <meshBasicMaterial color="#FF6B35" />
          </Box>
          <Box args={[0.3, 0.2, 40]} position={[7.5, -1.75, 0]}>
            <meshBasicMaterial color="#FF6B35" />
          </Box>

          {/* Reflective center line */}
          {Array.from({ length: 20 }, (_, i) => (
            <Box key={i} args={[0.2, 0.02, 1.5]} position={[0, -1.83, -18 + i * 2]} rotation={[-Math.PI / 2, 0, 0]}>
              <meshBasicMaterial color="#FFD700" />
            </Box>
          ))}

          {/* Reflective lane dividers */}
          {Array.from({ length: 20 }, (_, i) => (
            <React.Fragment key={i}>
              <Box args={[0.15, 0.02, 1]} position={[-3.5, -1.83, -18 + i * 2]} rotation={[-Math.PI / 2, 0, 0]}>
                <meshBasicMaterial color="#E0E0E0" />
              </Box>
              <Box args={[0.15, 0.02, 1]} position={[3.5, -1.83, -18 + i * 2]} rotation={[-Math.PI / 2, 0, 0]}>
                <meshBasicMaterial color="#E0E0E0" />
              </Box>
            </React.Fragment>
          ))}

          {/* Illuminated track barriers */}
          {Array.from({ length: 10 }, (_, i) => (
            <React.Fragment key={i}>
              <Box args={[0.5, 1, 3]} position={[-9, -1, -15 + i * 6]}>
                <meshPhongMaterial color="#C0C0C0" />
              </Box>
              <Box args={[0.5, 1, 3]} position={[9, -1, -15 + i * 6]}>
                <meshPhongMaterial color="#C0C0C0" />
              </Box>
            </React.Fragment>
          ))}

          {/* Night grandstands with lights */}
          <Box args={[3, 2, 15]} position={[-12, 0, 0]}>
            <meshPhongMaterial color="#2A2A2A" />
          </Box>
          <Box args={[3, 2, 15]} position={[12, 0, 0]}>
            <meshPhongMaterial color="#2A2A2A" />
          </Box>

          {/* Stadium floodlights */}
          {Array.from({ length: 8 }, (_, i) => (
            <React.Fragment key={i}>
              <Box args={[0.2, 4, 0.2]} position={[-11, 2, -12 + i * 3]}>
                <meshPhongMaterial color="#808080" />
              </Box>
              <Sphere args={[0.3]} position={[-11, 4, -12 + i * 3]}>
                <meshBasicMaterial color="#FFFFCC" />
              </Sphere>
              <Box args={[0.2, 4, 0.2]} position={[11, 2, -12 + i * 3]}>
                <meshPhongMaterial color="#808080" />
              </Box>
              <Sphere args={[0.3]} position={[11, 4, -12 + i * 3]}>
                <meshBasicMaterial color="#FFFFCC" />
              </Sphere>
            </React.Fragment>
          ))}

          {/* Start/finish line with night lighting */}
          <Plane args={[15, 0.5]} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.82, 15]}>
            <meshBasicMaterial color="#FFFFFF" />
          </Plane>

          {/* Checkered pattern for start/finish */}
          {Array.from({ length: 8 }, (_, i) => (
            <Plane key={i} args={[1.8, 0.25]} rotation={[-Math.PI / 2, 0, 0]} position={[-6.75 + i * 1.9, -1.815, 15]}>
              <meshBasicMaterial color={i % 2 === 0 ? "#000000" : "#FFFFFF"} />
            </Plane>
          ))}

          {/* Night sky background */}
          <Plane args={[80, 15]} position={[0, 6, -25]} rotation={[0, 0, 0]}>
            <meshBasicMaterial color="#0A0A1A" />
          </Plane>

          {/* City skyline silhouette */}
          {Array.from({ length: 8 }, (_, i) => (
            <Box key={i} args={[3, 2 + Math.random() * 4, 2]} position={[-16 + i * 4, 2, -22]}>
              <meshBasicMaterial color="#1A1A2E" />
            </Box>
          ))}

          {/* City lights */}
          {Array.from({ length: 20 }, (_, i) => (
            <Sphere key={i} args={[0.05]} position={[-18 + Math.random() * 36, 1 + Math.random() * 4, -21]}>
              <meshBasicMaterial color={Math.random() > 0.5 ? "#FFD700" : "#4ECDC4"} />
            </Sphere>
          ))}

          {/* Stars in the sky */}
          {Array.from({ length: 30 }, (_, i) => (
            <Sphere key={i} args={[0.02]} position={[
              (Math.random() - 0.5) * 60,
              8 + Math.random() * 6,
              -20 - Math.random() * 10
            ]}>
              <meshBasicMaterial color="#FFFFFF" />
            </Sphere>
          ))}
        </>
      )}
      
      {/* Enhanced Professional Gaming Lighting */}
      <ambientLight
        intensity={gameType === 'racing' ? 0.1 : gameType === 'badminton' ? 0.4 : 0.3}
        color={gameType === 'racing' ? "#1A1A3A" : "#2A2A4A"}
      />

      {/* Main arena lighting */}
      <directionalLight
        position={[15, 12, 8]}
        intensity={gameType === 'racing' ? 0.8 : gameType === 'badminton' ? 2.0 : 1.5}
        color={gameType === 'fighting' ? "#4ECDC4" : gameType === 'badminton' ? "#FFFFFF" : "#FFFFCC"}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-far={50}
        shadow-camera-left={-15}
        shadow-camera-right={15}
        shadow-camera-top={15}
        shadow-camera-bottom={-15}
      />

      {/* Accent lighting */}
      <directionalLight
        position={[-10, 8, -6]}
        intensity={gameType === 'racing' ? 0.5 : 0.8}
        color={gameType === 'fighting' ? "#A855F7" : gameType === 'badminton' ? "#87CEEB" : "#4ECDC4"}
      />

      {/* Central spotlight */}
      <spotLight
        position={[0, 15, 0]}
        intensity={gameType === 'racing' ? 2.0 : gameType === 'badminton' ? 1.5 : 1.2}
        angle={Math.PI / 4}
        penumbra={0.3}
        color={gameType === 'racing' ? "#FFFFCC" : "#FFFFFF"}
        castShadow
      />

      {/* Stadium lighting for badminton */}
      {gameType === 'badminton' && (
        <>
          <spotLight position={[-10, 12, -8]} intensity={0.8} angle={Math.PI / 6} color="#FFFFFF" />
          <spotLight position={[10, 12, -8]} intensity={0.8} angle={Math.PI / 6} color="#FFFFFF" />
        </>
      )}

      {/* Night racing track lighting */}
      {gameType === 'racing' && (
        <>
          {Array.from({ length: 6 }, (_, i) => (
            <React.Fragment key={i}>
              <spotLight
                position={[-11, 4, -12 + i * 4]}
                intensity={1.5}
                angle={Math.PI / 3}
                penumbra={0.5}
                color="#FFFFCC"
                target-position={[0, -2, -12 + i * 4]}
              />
              <spotLight
                position={[11, 4, -12 + i * 4]}
                intensity={1.5}
                angle={Math.PI / 3}
                penumbra={0.5}
                color="#FFFFCC"
                target-position={[0, -2, -12 + i * 4]}
              />
            </React.Fragment>
          ))}
        </>
      )}

      {/* Rim lighting (reduced to avoid covering players) */}
      <pointLight position={[-8, 4, 8]} intensity={0.4} color="#4ECDC4" />
      <pointLight position={[8, 4, -8]} intensity={0.4} color="#A855F7" />
      <pointLight position={[-8, 4, -8]} intensity={0.4} color="#FF6B35" />

    </>
  );
};

// Enhanced Professional Gaming Camera Controller
const CameraController = ({ gameType }: { gameType: 'fighting' | 'badminton' | 'racing' }) => {
  const { camera, gl } = useThree();
  const controlsRef = useRef<any>(null);

  useEffect(() => {
    // Smooth camera transitions for professional gaming perspective
    const targetPositions = {
      fighting: { position: [4, 2, 4], target: [0, 0.5, 0] },
      badminton: { position: [0, 4, 5], target: [0, 1, 0] },
      racing: { position: [0, 2, 8], target: [0, -1, 0] }
    };

    const target = targetPositions[gameType];

    // Smooth camera animation
    const animateCamera = () => {
      const startPos = camera.position.clone();
      const endPos = new THREE.Vector3(...target.position);
      const startTime = Date.now();
      const duration = 1000; // 1 second transition

      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Smooth easing function
        const easeProgress = 1 - Math.pow(1 - progress, 3);

        camera.position.lerpVectors(startPos, endPos, easeProgress);
        camera.lookAt(...target.target);

        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };

      animate();
    };

    animateCamera();
  }, [camera, gameType]);

  return null;
};

const GameArena: React.FC<GameArenaProps> = ({ gameType, onGameChange, showAnalytics, onToggleAnalytics }) => {
  const [gameStarted, setGameStarted] = useState(false);
  const [paused, setPaused] = useState(false);

  const renderGameContent = () => {
    switch (gameType) {
      case 'fighting':
        return (
          <>
            <FighterCharacter position={[-4.5, 0, 0]} color="#00B3FF" isPlayer initialFacing={1} engaged={gameStarted} paused={paused} />
            <FighterCharacter position={[4.5, 0, 0]} color="#FF4455" initialFacing={1} engaged={gameStarted} paused={paused} />
          </>
        );
      case 'badminton':
        return (
          <>
            {/* Players face each other across the net with realistic spacing (left-right) */}
            <BadmintonPlayer position={[-5, 0, 0]} color="#22D3EE" isPlayer paused={paused} />
            <BadmintonPlayer position={[5, 0, 0]} color="#F97316" paused={paused} />
            {/* Realistic Shuttlecock with physics */}
            <Shuttlecock paused={paused} />
          </>
        );
      case 'racing':
        return (
          <>
            <RacingCar position={[-2, -1.75, 0]} color="#4ECDC4" isPlayer paused={paused} />
            <RacingCar position={[2, -1.75, -3]} color="#FF6B35" paused={paused} />
            <RacingCar position={[0, -1.75, -6]} color="#A855F7" paused={paused} />
          </>
        );
      default:
        return null;
    }
  };

  return (
    <div className="relative w-full h-screen bg-background overflow-hidden">
      {/* Game Arena */}
      <div className="absolute inset-0">
        <Canvas
          camera={{
            position: [6, 2, 6],
            fov: gameType === 'racing' ? 70 : 75,
            near: 0.1,
            far: 100
          }}
          shadows
          gl={{
            antialias: true,
            alpha: false,
            powerPreference: "high-performance"
          }}
        >
          <CameraController gameType={gameType} />
          <OrbitControls
            enablePan={false}
            enableZoom={true}
            minDistance={gameType === 'racing' ? 6 : 4}
            maxDistance={gameType === 'racing' ? 15 : 12}
            minPolarAngle={Math.PI / 12}
            maxPolarAngle={Math.PI / 2.5}
            minAzimuthAngle={gameType === 'fighting' ? -Math.PI / 3 : -Math.PI / 2}
            maxAzimuthAngle={gameType === 'fighting' ? Math.PI / 3 : Math.PI / 2}
            autoRotate={false}
            enableDamping={true}
            dampingFactor={0.08}
            rotateSpeed={0.5}
            zoomSpeed={0.6}
            target={gameType === 'racing' ? [0, -1, 0] : [0, 0, 0]}
          />
          
          <ArenaEnvironment gameType={gameType} />
          {renderGameContent()}
        </Canvas>
      </div>

      {/* Game UI Overlay */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Top Score Bar */}
        <div className="absolute top-4 left-0 right-0 flex justify-center">
          {gameType === 'fighting' && (
            <ScoreBar game="fighting" playerHealth={100} aiHealth={100} rounds={[0, 0]} />
          )}
          {gameType === 'badminton' && (
            <ScoreBar game="badminton" score={[0, 0]} />
          )}
          {gameType === 'racing' && (
            <ScoreBar game="racing" lap={1} totalLaps={3} position={1} totalRacers={6} />
          )}
        </div>

        {/* Top HUD */}
        <div className="absolute top-20 left-4 right-4 flex justify-between items-start pointer-events-auto">
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
                if (!gameStarted) { setGameStarted(true); setPaused(false); }
                else { setPaused(p => !p); }
              }}
              className="hud-element px-3 py-2 rounded-lg text-xs font-medium"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {!gameStarted ? 'Start' : paused ? 'Resume' : 'Pause'}
            </motion.button>

            {/* Analytics Button */}
            <motion.button
              onClick={onToggleAnalytics}
              className="hud-element px-3 py-2 rounded-lg text-xs font-medium"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Analytics
            </motion.button>
          </div>
        </div>

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
    </div>
  );
};

export default GameArena;
