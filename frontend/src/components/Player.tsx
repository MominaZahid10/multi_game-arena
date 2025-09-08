import React, { useEffect, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF, useAnimations } from "@react-three/drei";
import { useBox } from "@react-three/cannon";

export type PlayerAction = "idle" | "run" | "jump" | "fight";

type PlayerProps = {
  modelUrl?: string;
  position?: [number, number, number];
  moveDirection?: { x: number; z: number };
  jump?: boolean;
  onLand?: () => void;
  action?: PlayerAction;
  leanIntensity?: number; // 0..1
};

const PLAYER_HEIGHT = 1.8;
const PLAYER_RADIUS = 0.4;
const JUMP_FORCE = 6;
const MOVE_SPEED = 4;

export default function Player({
  modelUrl = "/models/player.glb",
  position = [0, PLAYER_HEIGHT / 2, 0],
  moveDirection = { x: 0, z: 0 },
  jump = false,
  onLand,
  action = "idle",
  leanIntensity = 1,
}: PlayerProps) {
  const group = useRef<THREE.Group>(null);
  const rig = useRef<THREE.Group>(null);

  // Physics body (capsule-like box)
  const [isGrounded, setIsGrounded] = useState(true);
  const [ref, api] = useBox(() => ({
    type: "Dynamic",
    mass: 1,
    position,
    args: [PLAYER_RADIUS * 2, PLAYER_HEIGHT, PLAYER_RADIUS * 2],
    onCollide: (e: any) => {
      const impact = e?.contact?.impactVelocity ?? 0;
      if (impact > 0.2) setIsGrounded(true);
      if (onLand && impact > 2) onLand();
    },
  }));

  // Model + animations
  const { scene, animations } = useGLTF(modelUrl);
  const { actions } = useAnimations(animations, rig);

  // Animation state machine with blending
  useEffect(() => {
    let clip = "Idle";
    if (action === "jump") clip = "Jump";
    else if (action === "fight") clip = "Fight";
    else if (moveDirection.x !== 0 || moveDirection.z !== 0) clip = "Run";

    Object.entries(actions || {}).forEach(([name, act]) => {
      if (!act) return;
      if (name === clip) act.reset().fadeIn(0.2).play();
      else act.fadeOut(0.2);
    });
  }, [action, moveDirection, actions]);

  // Movement + leaning
  useFrame(() => {
    // Horizontal velocity
    if (moveDirection.x !== 0 || moveDirection.z !== 0) {
      api.velocity.set(
        moveDirection.x * MOVE_SPEED,
        undefined as unknown as number,
        moveDirection.z * MOVE_SPEED
      );
      if (rig.current) {
        // Lean body: strafe leans Z, forward/back leans X
        rig.current.rotation.z = -moveDirection.x * 0.25 * leanIntensity;
        rig.current.rotation.x = moveDirection.z * 0.12 * leanIntensity;
      }
    } else if (rig.current) {
      rig.current.rotation.z = 0;
      rig.current.rotation.x = 0;
    }

    // Jump impulse
    if (jump && isGrounded) {
      api.velocity.set(
        undefined as unknown as number,
        JUMP_FORCE,
        undefined as unknown as number
      );
      setIsGrounded(false);
    }
  });

  return (
    <group ref={ref as any}>
      <group ref={group}>
        <group ref={rig}>
          {/* Render full scene to avoid hardcoding node names */}
          <primitive object={scene} />
        </group>
      </group>
    </group>
  );
}

useGLTF.preload("/models/player.glb");
