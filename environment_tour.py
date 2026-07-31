"""
Narrated environment tour — Autonomous Medical Drone Delivery.

Shows, in order:
  1. The whole environment from a wide overview
  2. A zoomed-in look at the hospital
  3. A zoomed-in look at the clinic
  4. A zoomed-in look at the charging station
  5. A zoomed-in look at the drone
  ...then a short optional follow-cam clip of the drone acting under a simple heuristic
  policy, so the tour doesn't end on a static frame.

Includes offline spoken narration (pyttsx3) describing the problem, the agent's objective,
and the reward structure — content written so you can reuse it directly for the assignment's
video demonstration requirement. Subtitles are also drawn on screen (and printed to the
terminal) in case TTS isn't available on your machine, so the script always runs.

This script does NOT modify environment/custom_env.py or environment/rendering.py — it only
imports and drives them from the outside, by taking manual control of the existing Camera.

Setup:
    pip install pyttsx3
    # Linux also needs a system TTS backend, e.g.: sudo apt install espeak
    # (If pyttsx3 isn't installed/working, the script still runs — narration becomes
    #  on-screen subtitles + terminal text only, no audio.)

Run (from the project root):
    uv run python environment_tour.py
    # or: python environment_tour.py
"""

import time
import threading

from environment.custom_env import MedicalDroneEnv

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


class Narrator:
    """Non-blocking offline text-to-speech, with a console/subtitle fallback."""

    def __init__(self, rate=165):
        self.engine = None
        self.current_text = ""
        self.rate_wps = rate / 60.0  # words per second, used to estimate how long to hold a stage
        if _TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", rate)
            except Exception as e:
                print(f"[narrator] TTS init failed ({e}); continuing with subtitles only.")
                self.engine = None
        else:
            print("[narrator] pyttsx3 not installed; continuing with subtitles only. "
                  "Run `pip install pyttsx3` for spoken narration.")

    def say(self, text):
        self.current_text = text
        print(f"\n[narration] {text}")
        if self.engine is not None:
            def _speak():
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"[narrator] speech failed: {e}")
            threading.Thread(target=_speak, daemon=True).start()

    def estimate_duration(self, text, minimum=3.5):
        n_words = len(text.split())
        return max(minimum, n_words / self.rate_wps + 0.8)


def wrap_text(text, font, max_width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_subtitle(renderer, text):
    import pygame
    if not text:
        return
    font = renderer.font
    W = renderer.screen.get_width()
    H = renderer.screen.get_height()
    lines = wrap_text(text, font, W - 80)
    bar_h = 26 * len(lines) + 20
    bar = pygame.Surface((W, bar_h), pygame.SRCALPHA)
    bar.fill((10, 12, 18, 215))
    renderer.screen.blit(bar, (0, H - bar_h))
    for i, line in enumerate(lines):
        surf = font.render(line, True, (235, 238, 245))
        renderer.screen.blit(surf, (40, H - bar_h + 10 + i * 26))
    pygame.display.flip()


def run_stage(env, narrator, target_center_px, target_view_w, narration_text, extra_hold=0.0):
    """Smoothly pan/zoom the existing camera to a target and hold while narrating."""
    renderer = env.renderer
    camera = renderer.camera
    camera.mode = "manual"  # bypasses the built-in intro/follow state machine; still smoothed
    camera.target_center[0] = target_center_px[0]
    camera.target_center[1] = target_center_px[1]
    camera.target_view_w = target_view_w

    narrator.say(narration_text)
    duration = narrator.estimate_duration(narration_text) + extra_hold

    start = time.time()
    while time.time() - start < duration:
        if getattr(renderer, "_quit_requested", False):
            return False
        env.render()
        draw_subtitle(renderer, narrator.current_text)
    return True


def main():
    env = MedicalDroneEnv(render_mode="human")
    obs, info = env.reset(seed=7)
    env.render()  # lazy-inits the renderer/camera so we can take manual control of it below

    renderer = env.renderer
    g2p = renderer.grid_to_px
    world_center_px = (renderer.world_w / 2, renderer.world_h / 2)

    narrator = Narrator()

    stages = [
        (world_center_px, renderer.world_w,
         "Welcome to the Autonomous Medical Drone Delivery environment. This is a rural "
         "healthcare network where a drone has to carry medical supplies from a hospital to a "
         "remote clinic, while managing a limited battery and avoiding storms, mountains, and "
         "no-fly zones. Every episode randomizes the layout, so the agent has to learn a "
         "general strategy rather than memorize one fixed route."),

        (g2p(env.hospital), 640,
         "This is the hospital. Whenever the drone isn't already carrying a package, its "
         "objective is to fly here first and collect medical supplies before attempting any "
         "delivery."),

        (g2p(env.clinic), 640,
         "This is the clinic, the delivery destination. Once the drone is carrying a package, "
         "its objective switches to reaching this clinic and delivering the supplies here."),

        (g2p(env.charging), 480,
         "This is the charging station. If the drone's battery gets critically low, its "
         "objective automatically switches here so it can recharge before continuing the "
         "mission. Recharging too early, while the battery is still high, is actually "
         "penalized, so the agent has to learn to time it well."),

        (g2p(env.drone_pos), 420,
         "And this is the drone itself, the reinforcement learning agent. "
         f"Right now it's carrying {'a package' if env.has_package else 'no package'}, with "
         f"{env.battery:.0f} percent battery remaining, on a "
         f"{'high-priority, urgent' if env.priority == 1 else 'normal-priority'} mission. "
         "Its reward function gives plus thirty for picking up a package, plus two hundred for "
         "a successful delivery, and penalties for flying through storms, entering no-fly "
         "zones, or crashing into mountains — so it has to balance speed against safety and "
         "battery life."),
    ]

    for target_center, target_w, text in stages:
        if not run_stage(env, narrator, target_center, target_w, text):
            env.close()
            return

    # Bonus: hand off to a short heuristic follow-cam clip so the tour ends on the agent
    # actually moving, not a static frame.
    narrator.say("Let's watch the drone attempt its mission using a simple heuristic policy, "
                  "so you can see how the camera follows it around the environment.")
    renderer.camera.mode = "follow"
    demo_start = time.time()
    while time.time() - demo_start < 20:
        if getattr(renderer, "_quit_requested", False):
            break
        obj_name, obj_pos = env._current_objective()
        dx = obj_pos[0] - env.drone_pos[0]
        dy = obj_pos[1] - env.drone_pos[1]
        if abs(dx) < 1.0 and abs(dy) < 1.0:
            action = 5  # INTERACT
        elif abs(dx) > abs(dy):
            action = 2 if dx > 0 else 3  # EAST / WEST
        else:
            action = 1 if dy > 0 else 0  # SOUTH / NORTH
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        draw_subtitle(renderer, narrator.current_text)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
