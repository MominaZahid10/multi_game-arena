import React, { useEffect, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { RigidBody, CapsuleCollider } from "@react-three/rapier";

export type PlayerAction = "idle" | "run" | "jump" | "fight";

type PlayerProps = {
  position?: [number, number, number];
  moveDirection?: { x: number; z: number };
  jump?: boolean;
  onLand?: () => void;
  action?: PlayerAction;
  color?: string;
  leanIntensity?: number; // 0..1
};

const PLAYER_HEIGHT = 1.8;
const PLAYER_RADIUS = 0.35;
const JUMP_FORCE = 6;
const MOVE_SPEED = 4.5;

export default function Player({
  position = [0, PLAYER_HEIGHT / 2, 0],
  moveDirection = { x: 0, z: 0 },
  jump = false,
  onLand,
  action = "idle",
  color = "#22D3EE",
  leanIntensity = 1,
}: PlayerProps) {
  const body = useRef<any>(null);
  const rig = useRef<THREE.Group>(null);
  const [isGrounded, setIsGrounded] = useState(true);

  // Basic animation mimic via lean based on intent
  useEffect(() => {
    if (!rig.current) return;
    if (action === "fight") rig.current.rotation.y = Math.sin(Date.now() * 0.01) * 0.1;
    else rig.current.rotation.y = 0;
  }, [action]);

  useFrame(() => {
    const rb = body.current;
    if (!rb) return;
    const lin = rb.linvel();
    const targetX = moveDirection.x * MOVE_SPEED;
    const targetZ = moveDirection.z * MOVE_SPEED;

    rb.setLinvel({ x: targetX, y: lin.y, z: targetZ }, true);

    if (rig.current) {
      rig.current.rotation.z = -moveDirection.x * 0.25 * leanIntensity;
      rig.current.rotation.x = moveDirection.z * 0.12 * leanIntensity;
    }

    if (jump && isGrounded) {
      rb.setLinvel({ x: lin.x, y: JUMP_FORCE, z: lin.z }, true);
      setIsGrounded(false);
    }
  });

  return (
    <RigidBody
      ref={body}
      type="dynamic"
      mass={1}
      canSleep={false}
      enabledRotations={[false, true, false]}
      onCollisionEnter={(e) => {
        if ((e?.other?.rigidBody?.mass() ?? 0) === 0) setIsGrounded(true);
        if (onLand) onLand();
      }}
    >
      <CapsuleCollider args={[PLAYER_HEIGHT / 2, PLAYER_RADIUS]} position={[0, PLAYER_HEIGHT / 2, 0]} />
      <group ref={rig}>
        {/* Simple capsule visual */}
        <mesh position={[0, PLAYER_HEIGHT / 2, 0]}>
          <cylinderGeometry args={[PLAYER_RADIUS, PLAYER_RADIUS, PLAYER_HEIGHT - PLAYER_RADIUS * 2, 12]} />
          <meshStandardMaterial color={color} />
        </mesh>
        <mesh position={[0, PLAYER_HEIGHT - PLAYER_RADIUS, 0]}>
          <sphereGeometry args={[PLAYER_RADIUS, 16, 16]} />
          <meshStandardMaterial color={color} />
        </mesh>
        <mesh position={[0, PLAYER_RADIUS, 0]}>
          <sphereGeometry args={[PLAYER_RADIUS, 16, 16]} />
          <meshStandardMaterial color={color} />
        </mesh>
      </group>
    </RigidBody>
  );
}
