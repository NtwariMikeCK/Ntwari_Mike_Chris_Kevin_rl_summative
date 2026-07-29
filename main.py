"""
Demo entry point for the Autonomous Medical Drone Delivery environment.

Runs the environment with a random (or simple heuristic) policy so you can
see the world, the cinematic camera intro, and the manual camera controls.

Controls once the intro finishes:
  M / F       toggle manual camera mode
  Arrows/WASD pan the camera (manual mode only)
  +/-         zoom in/out (manual mode only)
  Esc         quit
"""

import argparse
import time

from environment.custom_env import MedicalDroneEnv


def heuristic_action(env):
    """A simple greedy heuristic: fly toward the current objective, avoiding
    nothing in particular (used only to make the demo look purposeful)."""
    obj_name, obj_pos = env._current_objective()
    dx = obj_pos[0] - env.drone_pos[0]
    dy = obj_pos[1] - env.drone_pos[1]
    if abs(dx) < 1.0 and abs(dy) < 1.0:
        return 5  # INTERACT
    if abs(dx) > abs(dy):
        return 2 if dx > 0 else 3  # EAST / WEST
    return 1 if dy > 0 else 0      # SOUTH / NORTH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--policy", choices=["random", "heuristic"], default="heuristic")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    env = MedicalDroneEnv(render_mode="human")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed)
        env.render()
        terminated = truncated = False
        ep_reward = 0.0

        while not (terminated or truncated):
            if getattr(env.renderer, "_quit_requested", False):
                env.close()
                return

            if args.policy == "random":
                action = env.action_space.sample()
            else:
                action = heuristic_action(env)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            env.render()

        print(f"[Episode {ep + 1}] return={ep_reward:.1f} steps={info['steps']} "
              f"event='{info['event']}'")
        time.sleep(1.0)

    env.close()


if __name__ == "__main__":
    main()
