"""
Autonomous Medical Drone Delivery Environment
===============================================
A custom Gymnasium environment simulating a medical supply drone operating
in a rural healthcare network. The drone must pick up supplies at a hospital,
navigate hazards (storms, mountains, no-fly zones), manage a limited battery,
and deliver supplies to a clinic.

See the environment specification document for full design rationale.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Action:
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    HOVER = 4
    INTERACT = 5
    WAIT = 6


MOVE_DELTAS = {
    Action.NORTH: (0, -1),
    Action.SOUTH: (0, 1),
    Action.EAST: (1, 0),
    Action.WEST: (-1, 0),
}


class MedicalDroneEnv(gym.Env):
    """Custom environment for training RL agents on medical drone delivery."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    GRID_SIZE = 30
    MAX_STEPS = 300

    N_STORMS = 1
    N_MOUNTAINS = 2
    N_NOFLY = 1

    HAZARD_RADIUS = 1.8          # storm / no-fly zone influence radius (grid units)
    MOUNTAIN_RADIUS = 1.3
    INTERACT_RADIUS = 1.1        # how close the drone must be to "use" a landmark
    SAFE_MARGIN = 2.5            # clearance kept clear of no-fly zones around key sites
    LOW_BATTERY_THRESHOLD = 15.0

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.renderer = None

        self.action_space = spaces.Discrete(7)

        # Observation layout (float32):
        # [drone_x, drone_y, battery, has_package,
        #  hospital_x, hospital_y, clinic_x, clinic_y, charge_x, charge_y,
        #  priority, wind,
        #  storm_x,y * N_STORMS, mountain_x,y * N_MOUNTAINS, nofly_x,y * N_NOFLY,
        #  dist_hospital, dist_clinic, dist_charge, dist_storm, dist_mountain, dist_nofly]
        obs_dim = 4 + 6 + 2 + self.N_STORMS * 2 + self.N_MOUNTAINS * 2 + self.N_NOFLY * 2 + 6
        high = np.full(obs_dim, self.GRID_SIZE, dtype=np.float32)
        high[2] = 100.0   # battery
        high[3] = 1.0     # package flag
        low = np.zeros(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._rng = np.random.default_rng()

        # episode state placeholders
        self.drone_pos = np.zeros(2, dtype=np.float32)
        self.battery = 100.0
        self.has_package = False
        self.hospital = np.zeros(2, dtype=np.float32)
        self.clinic = np.zeros(2, dtype=np.float32)
        self.charging = np.zeros(2, dtype=np.float32)
        self.mountains = []
        self.storms = []
        self.no_fly_zones = []
        self.wind = 0.0
        self.priority = 0  # 0 normal, 1 urgent
        self.steps = 0
        self.last_reward = 0.0
        self.last_event = ""
        self.terrain_seed = 0

    # ------------------------------------------------------------------ #
    # Episode setup
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._generate_episode()
        self.steps = 0
        self.last_reward = 0.0
        self.last_event = "Mission start"
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _rand_point(self, margin=1.0):
        return self._rng.uniform(margin, self.GRID_SIZE - margin, size=2).astype(np.float32)

    @staticmethod
    def _far_enough(points, new_point, min_dist):
        for p in points:
            if np.linalg.norm(np.asarray(p) - np.asarray(new_point)) < min_dist:
                return False
        return True

    def _generate_episode(self):
        g = self.GRID_SIZE
        key_sites = []

        self.hospital = self._rand_point(3)
        key_sites.append(self.hospital)

        self.clinic = self.hospital
        for _ in range(200):
            c = self._rand_point(3)
            if self._far_enough(key_sites, c, g * 0.35):
                self.clinic = c
                break
        key_sites.append(self.clinic)

        # Charging station sits at the midpoint of the hospital-clinic route, so it is
        # always directly on the path between them rather than an unrelated detour —
        # this guarantees a recharge stop is reachable within roughly half the total
        # mission distance, on either leg of the trip.
        self.charging = (self.hospital + self.clinic) / 2.0
        key_sites.append(self.charging)

        # Drone starts close to the hospital (a short hop away, not on top of it) so the
        # first leg of every mission is short and the agent can spend most of its battery
        # budget on the hospital->charger->clinic route rather than finding the hospital.
        for _ in range(50):
            angle = self._rng.uniform(0, 2 * np.pi)
            radius = self._rng.uniform(1.5, 4.0)
            candidate = self.hospital + radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            self.drone_pos = np.clip(candidate, 0, g).astype(np.float32)
            if np.linalg.norm(self.drone_pos - self.hospital) >= self.INTERACT_RADIUS + 0.3:
                break

        # Mountains: static obstacles, kept clear of the interact radius of key sites
        self.mountains = []
        for _ in range(self.N_MOUNTAINS):
            for _ in range(100):
                m = self._rand_point(2)
                if self._far_enough(key_sites, m, self.MOUNTAIN_RADIUS + self.INTERACT_RADIUS + 0.5) \
                        and self._far_enough([self.drone_pos], m, 2.0):
                    self.mountains.append(m)
                    break

        # Storms: dynamic hazards, may legitimately overlap anything
        self.storms = [self._rand_point(2) for _ in range(self.N_STORMS)]

        # No-fly zones: NEVER intersect hospital, clinic, charging station, or drone start
        forbidden = key_sites + [self.drone_pos]
        self.no_fly_zones = []
        for _ in range(self.N_NOFLY):
            for _ in range(300):
                z = self._rand_point(2)
                if self._far_enough(forbidden, z, self.HAZARD_RADIUS + self.SAFE_MARGIN):
                    self.no_fly_zones.append(z)
                    break

        self.wind = float(self._rng.uniform(0, 10))
        self.priority = int(self._rng.integers(0, 2))
        self.battery = 100.0  # every mission starts with a full charge
        # self.has_package = bool(self._rng.integers(0, 2))
        self.has_package = False
        self.terrain_seed = int(self._rng.integers(0, 1_000_000))

    # ------------------------------------------------------------------ #
    # Step logic
    # ------------------------------------------------------------------ #
    def _current_objective(self):
        if self.battery < self.LOW_BATTERY_THRESHOLD:
            return "charging", self.charging
        if not self.has_package:
            return "hospital", self.hospital
        return "clinic", self.clinic

    def _nearest_dist(self, points):
        if not points:
            return float(self.GRID_SIZE)
        return float(min(np.linalg.norm(self.drone_pos - p) for p in points))


        
    def step(self, action):
        self.steps += 1
        reward = -0.5  # base step penalty
        terminated = False
        truncated = False
        event = ""

        _, obj_pos = self._current_objective()
        prev_dist = float(np.linalg.norm(self.drone_pos - obj_pos))

        # -------- movement / battery cost --------
        # Battery drain rates are halved from the original design (movement 2%->1%,
        # hover 1%->0.5%, wait 0.5%->0.25%, storm surcharge 2%->1%) so the drone has
        # roughly double the number of moves per full charge — enough to explore the
        # map, detour around hazards, and still comfortably complete the mission now
        # that the charging station sits on the direct hospital-clinic route.
        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self.drone_pos[0] = np.clip(self.drone_pos[0] + dx, 0, self.GRID_SIZE)
            self.drone_pos[1] = np.clip(self.drone_pos[1] + dy, 0, self.GRID_SIZE)
            self.battery -= 1.0
        elif action == Action.HOVER:
            self.battery -= 0.5
            reward -= 1.0
            event = "Hovering"
        elif action == Action.WAIT:
            self.battery -= 0.25
            reward -= 1.0
            event = "Waiting"
        elif action == Action.INTERACT:
            interact_reward, event, delivered = self._handle_interact()
            reward += interact_reward
            if delivered:
                terminated = True




        # -------- automatic pickup / delivery --------
        dist_hosp = np.linalg.norm(self.drone_pos - self.hospital)
        dist_clinic = np.linalg.norm(self.drone_pos - self.clinic)

        # Auto-pickup when entering hospital zone
        if dist_hosp < self.INTERACT_RADIUS and not self.has_package:
            self.has_package = True
            reward += 30.0
            event = "Package collected"

        # Auto-delivery when entering clinic zone with package
        if dist_clinic < self.INTERACT_RADIUS and self.has_package:
            reward += 200.0

            # Battery bonus
            if self.battery > 30:
                reward += 20.0

            # Urgent delivery bonus
            if self.priority == 1:
                remaining_frac = max(0.0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
                reward += 50.0 * remaining_frac

            self.has_package = False
            terminated = True
            event = "Delivery successful"




        # storm penalty, checked at the (possibly new) position
        if self._nearest_dist(self.storms) < self.HAZARD_RADIUS:
            self.battery -= 1.0
            reward -= 20.0
            event = "Flying through storm"

        self.battery = float(np.clip(self.battery, 0, 100))

        # -------- distance shaping (only for movement actions) --------
        # Reuses the SAME objective reference captured before this step's movement,
        # rather than recomputing it afterward — recomputing could silently compare
        # distance-to-A (before) against distance-to-B (after) on the rare step where
        # crossing LOW_BATTERY_THRESHOLD flips the objective mid-step.
        new_dist = float(np.linalg.norm(self.drone_pos - obj_pos))
        if action in MOVE_DELTAS:
            if new_dist < prev_dist:
                reward += 2.0
            elif new_dist > prev_dist:
                reward -= 2.0

        # -------- hazard / terminal checks --------
        if self._nearest_dist(self.mountains) < self.MOUNTAIN_RADIUS:
            reward -= 100.0
            terminated = True
            event = "Crashed into mountain"
        elif self._nearest_dist(self.no_fly_zones) < self.HAZARD_RADIUS:
            reward -= 60.0
            terminated = True
            event = "Entered no-fly zone"
        elif self.battery <= 0:
            reward -= 100.0
            terminated = True
            event = "Battery depleted"
        elif not terminated and self.steps >= self.MAX_STEPS:
            reward -= 50.0
            truncated = True
            event = "Max episode length exceeded"

        self.last_reward = reward
        if event:
            self.last_event = event
        obs = self._get_obs()
        info = self._get_info()
        info["event"] = self.last_event
        return obs, reward, terminated, truncated, info

    def _handle_interact(self):
      """Returns (reward, event_string, delivered_bool)."""
      dist_charge = np.linalg.norm(self.drone_pos - self.charging)

      # Charging still requires an explicit INTERACT action
      if dist_charge < self.INTERACT_RADIUS:
          r = 0.0
          if self.battery < 40:
              r += 15.0
          elif self.battery > 80:
              r -= 5.0

          self.battery = 100.0
          return r, "Recharging", False

      return -10.0, "Interact with nothing", False
      
    # ------------------------------------------------------------------ #
    # Observation / info
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        parts = [
            self.drone_pos,
            np.array([self.battery, float(self.has_package)], dtype=np.float32),
            self.hospital, self.clinic, self.charging,
            np.array([float(self.priority), self.wind], dtype=np.float32),
        ]
        parts.extend(self.storms)
        parts.extend(self.mountains)
        parts.extend(self.no_fly_zones)
        parts.append(np.array([
            np.linalg.norm(self.drone_pos - self.hospital),
            np.linalg.norm(self.drone_pos - self.clinic),
            np.linalg.norm(self.drone_pos - self.charging),
            self._nearest_dist(self.storms),
            self._nearest_dist(self.mountains),
            self._nearest_dist(self.no_fly_zones),
        ], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def _get_info(self):
        obj_name, _ = self._current_objective()
        return {
            "battery": self.battery,
            "has_package": self.has_package,
            "objective": obj_name,
            "steps": self.steps,
            "priority": "urgent" if self.priority == 1 else "normal",
            "event": self.last_event,
        }

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self):
        if self.renderer is None:
            from environment.rendering import Renderer
            self.renderer = Renderer(self)
        return self.renderer.render(mode=self.render_mode or "human")

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
