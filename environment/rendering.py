"""
Renderer for the Medical Drone Delivery environment.

Implements:
- Procedurally generated rural terrain (cached per episode)
- Hand-drawn vector-style sprites for every entity (no external image assets needed)
- A cinematic camera: intro flyover -> hospital -> clinic -> drone -> smooth follow
- Manual camera override (pan with arrow keys / WASD, zoom with +/-, 'F' to resume follow)
"""

import math
import numpy as np
import pygame

TILE = 64                     # pixels per grid unit at zoom = 1.0
SCREEN_W, SCREEN_H = 1280, 800
HUD_HEIGHT = 90

COL_BG_SKY = (18, 24, 38)
COL_GRASS_A = (58, 102, 54)
COL_GRASS_B = (72, 122, 64)
COL_GRASS_C = (46, 84, 44)
COL_DIRT = (109, 87, 60)
COL_WATER = (48, 92, 128)

COL_UI_PANEL = (15, 18, 26, 210)
COL_UI_TEXT = (235, 238, 245)
COL_BATTERY_GOOD = (86, 214, 130)
COL_BATTERY_MID = (240, 196, 70)
COL_BATTERY_LOW = (232, 78, 78)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Camera:
    """World-pixel-space camera with smooth interpolation and manual override."""

    def __init__(self, world_w, world_h):
        self.world_w = world_w
        self.world_h = world_h

        self.center = np.array([world_w / 2, world_h / 2], dtype=np.float64)
        self.target_center = self.center.copy()
        self.view_w = float(world_w)   # visible world width (controls zoom)
        self.view_h = float(world_h * (SCREEN_H / SCREEN_W))
        self.target_view_w = self.view_w

        self.mode = "intro"           # intro -> follow -> manual
        self.intro_stage = 0
        self.stage_timer = 0.0
        self.follow_zoom_w = TILE * 14   # world-pixels visible width while following

        self.smooth_pos = 0.06
        self.smooth_zoom = 0.05

    def start_intro(self, hospital_px, clinic_px, drone_px):
        self.mode = "intro"
        self.intro_stage = 0
        self.stage_timer = 0.0
        self._hospital_px = hospital_px
        self._clinic_px = clinic_px
        self._drone_px = drone_px
        self.center = np.array([self.world_w / 2, self.world_h / 2], dtype=np.float64)
        self.view_w = float(self.world_w)

    def toggle_manual(self, drone_px):
        if self.mode == "manual":
            self.mode = "follow"
        else:
            self.mode = "manual"

    def manual_pan(self, dx, dy, dt):
        speed = self.view_w * 0.9  # world units per second, scales with zoom
        self.target_center[0] += dx * speed * dt
        self.target_center[1] += dy * speed * dt

    def manual_zoom(self, delta):
        self.target_view_w = clamp(self.target_view_w * (1.0 - delta * 0.12),
                                    TILE * 4, self.world_w * 1.3)

    def update(self, dt, drone_px):
        aspect = SCREEN_H / SCREEN_W

        if self.mode == "intro":
            stages = [
                # (target_center, target_width, hold_duration)
                (np.array([self.world_w / 2, self.world_h / 2]), self.world_w, 2.2),
                (np.array(self._hospital_px, dtype=np.float64), TILE * 8, 1.8),
                (np.array(self._clinic_px, dtype=np.float64), TILE * 8, 1.8),
                (np.array(self._drone_px, dtype=np.float64), self.follow_zoom_w, 1.2),
            ]
            self.stage_timer += dt
            idx = min(self.intro_stage, len(stages) - 1)
            tgt_center, tgt_w, hold = stages[idx]
            self.target_center = tgt_center
            self.target_view_w = tgt_w
            self.center += (self.target_center - self.center) * min(1.0, 3.0 * dt)
            self.view_w += (self.target_view_w - self.view_w) * min(1.0, 3.0 * dt)
            if self.stage_timer >= hold:
                self.stage_timer = 0.0
                self.intro_stage += 1
                if self.intro_stage >= len(stages):
                    self.mode = "follow"
        elif self.mode == "follow":
            self.target_center = np.array(drone_px, dtype=np.float64)
            self.target_view_w = self.follow_zoom_w
            self.center += (self.target_center - self.center) * self.smooth_pos
            self.view_w += (self.target_view_w - self.view_w) * self.smooth_zoom
        else:  # manual
            self.center += (self.target_center - self.center) * 0.25
            self.view_w += (self.target_view_w - self.view_w) * 0.15

        self.view_h = self.view_w * aspect

        half_w, half_h = self.view_w / 2, self.view_h / 2
        self.center[0] = clamp(self.center[0], half_w, max(half_w, self.world_w - half_w))
        self.center[1] = clamp(self.center[1], half_h, max(half_h, self.world_h - half_h))
        self.target_center[0] = clamp(self.target_center[0], 0, self.world_w)
        self.target_center[1] = clamp(self.target_center[1], 0, self.world_h)

    def world_to_screen_rect(self):
        """Returns the world-pixel rect currently visible, for extraction+scaling."""
        x = self.center[0] - self.view_w / 2
        y = self.center[1] - self.view_h / 2
        return pygame.Rect(int(x), int(y), int(self.view_w), int(self.view_h))


class Renderer:
    def __init__(self, env):
        self.env = env
        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
        self.font_big = None
        self.world_surf = None
        self.terrain_surf = None
        self.terrain_seed_cached = None
        self.time = 0.0
        self.camera = None
        self._initialized = False
        self._last_drone_px = None

    def _lazy_init(self):
        if self._initialized:
            return
        pygame.init()
        pygame.display.set_caption("Autonomous Medical Drone Delivery")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 18)
        self.font_small = pygame.font.SysFont("arial", 14)
        self.font_big = pygame.font.SysFont("arial", 26, bold=True)

        g = self.env.GRID_SIZE
        world_w, world_h = int(g * TILE), int(g * TILE)
        self.world_w, self.world_h = world_w, world_h
        self.world_surf = pygame.Surface((world_w, world_h))
        self.camera = Camera(world_w, world_h)
        self._initialized = True

    # ------------------------------------------------------------------ #
    # Terrain
    # ------------------------------------------------------------------ #
    def _build_terrain(self, seed):
        g = self.env.GRID_SIZE
        w, h = g * TILE, g * TILE
        surf = pygame.Surface((w, h))
        rng = np.random.default_rng(seed)

        # low-res noise field, upscaled, for a soft mottled field look
        low_res = 40
        noise = rng.random((low_res, low_res))
        for _ in range(3):
            noise = 0.25 * (
                np.roll(noise, 1, 0) + np.roll(noise, -1, 0) +
                np.roll(noise, 1, 1) + np.roll(noise, -1, 1)
            ) + 0.0 * noise  # box blur, smooths field boundaries
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

        block = w / low_res
        for iy in range(low_res):
            for ix in range(low_res):
                v = noise[iy, ix]
                if v < 0.35:
                    col = COL_GRASS_C
                elif v < 0.7:
                    col = COL_GRASS_A
                else:
                    col = COL_GRASS_B
                rect = pygame.Rect(int(ix * block), int(iy * block), int(block) + 1, int(block) + 1)
                pygame.draw.rect(surf, col, rect)

        # winding dirt road connecting a few random anchor points, purely decorative
        pts = [(rng.uniform(0, w), rng.uniform(0, h)) for _ in range(5)]
        for i in range(len(pts) - 1):
            pygame.draw.line(surf, COL_DIRT, pts[i], pts[i + 1], width=10)

        # scattered tree clusters for texture
        for _ in range(int(g * g * 0.35)):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            r = rng.uniform(3, 7)
            shade = rng.integers(-10, 10)
            col = (max(0, 30 + shade), max(0, 70 + shade), max(0, 34 + shade))
            pygame.draw.circle(surf, col, (int(x), int(y)), int(r))

        return surf

    def _ensure_terrain(self):
        if self.terrain_seed_cached != self.env.terrain_seed:
            self.terrain_surf = self._build_terrain(self.env.terrain_seed)
            self.terrain_seed_cached = self.env.terrain_seed

    # ------------------------------------------------------------------ #
    # Coordinate helpers
    # ------------------------------------------------------------------ #
    def grid_to_px(self, pos):
        return (float(pos[0]) * TILE, float(pos[1]) * TILE)

    # ------------------------------------------------------------------ #
    # Sprite drawing (all in world_surf pixel space)
    # ------------------------------------------------------------------ #
    def _draw_hospital(self, surf, px):
        x, y = px
        w, h = 70, 54
        rect = pygame.Rect(int(x - w / 2), int(y - h / 2), w, h)
        pygame.draw.rect(surf, (235, 235, 240), rect, border_radius=4)
        pygame.draw.rect(surf, (180, 40, 40), rect, width=3, border_radius=4)
        # roof helipad
        pygame.draw.circle(surf, (60, 60, 70), (int(x), int(y - h / 2 - 14)), 18)
        pygame.draw.circle(surf, (235, 235, 240), (int(x), int(y - h / 2 - 14)), 15)
        txt = self.font_small.render("H", True, (180, 40, 40))
        surf.blit(txt, txt.get_rect(center=(int(x), int(y - h / 2 - 14))))
        # red cross
        cx, cy = int(x), int(y)
        pygame.draw.rect(surf, (200, 40, 40), (cx - 4, cy - 14, 8, 28))
        pygame.draw.rect(surf, (200, 40, 40), (cx - 14, cy - 4, 28, 8))
        label = self.font_small.render("Hospital", True, (255, 255, 255))
        surf.blit(label, (int(x - label.get_width() / 2), int(y + h / 2 + 4)))

    def _draw_clinic(self, surf, px):
        x, y = px
        w, h = 54, 42
        rect = pygame.Rect(int(x - w / 2), int(y - h / 2), w, h)
        pygame.draw.rect(surf, (225, 235, 245), rect, border_radius=4)
        pygame.draw.rect(surf, (60, 120, 190), rect, width=3, border_radius=4)
        cx, cy = int(x), int(y)
        pygame.draw.rect(surf, (60, 120, 190), (cx - 3, cy - 10, 6, 20))
        pygame.draw.rect(surf, (60, 120, 190), (cx - 10, cy - 3, 20, 6))
        label = self.font_small.render("Clinic", True, (255, 255, 255))
        surf.blit(label, (int(x - label.get_width() / 2), int(y + h / 2 + 4)))

    def _draw_charging(self, surf, px):
        x, y = px
        pygame.draw.circle(surf, (40, 44, 54), (int(x), int(y)), 22)
        pygame.draw.circle(surf, (240, 196, 70), (int(x), int(y)), 22, width=3)
        bolt = [(x - 3, y - 10), (x + 5, y - 10), (x - 2, y), (x + 4, y), (x - 5, y + 11), (x + 2, y - 1), (x - 3, y - 1)]
        pygame.draw.polygon(surf, (240, 196, 70), bolt)
        label = self.font_small.render("Charge", True, (255, 255, 255))
        surf.blit(label, (int(x - label.get_width() / 2), int(y + 26)))

    def _draw_mountain(self, surf, px, radius_px):
        x, y = px
        base_w = radius_px * 2.1
        pts_back = [(x - base_w * 0.55, y + radius_px * 0.5),
                    (x - base_w * 0.05, y - radius_px * 1.1),
                    (x + base_w * 0.35, y + radius_px * 0.5)]
        pts_front = [(x - base_w * 0.35, y + radius_px * 0.55),
                     (x + base_w * 0.05, y - radius_px * 0.85),
                     (x + base_w * 0.55, y + radius_px * 0.55)]
        pygame.draw.polygon(surf, (96, 92, 88), pts_back)
        pygame.draw.polygon(surf, (120, 114, 108), pts_front)
        snow = [(x + base_w * 0.05 - 12, y - radius_px * 0.85 + 16),
                (x + base_w * 0.05, y - radius_px * 0.85),
                (x + base_w * 0.05 + 12, y - radius_px * 0.85 + 16)]
        pygame.draw.polygon(surf, (240, 240, 245), snow)

    def _draw_storm(self, surf, px, radius_px, t):
        x, y = px
        cloud_surf = pygame.Surface((int(radius_px * 3), int(radius_px * 3)), pygame.SRCALPHA)
        cx, cy = cloud_surf.get_width() / 2, cloud_surf.get_height() / 2
        for i in range(5):
            ang = t * 0.8 + i * (2 * math.pi / 5)
            ox = math.cos(ang) * radius_px * 0.45
            oy = math.sin(ang) * radius_px * 0.3
            r = radius_px * (0.55 + 0.1 * math.sin(t + i))
            pygame.draw.circle(cloud_surf, (60, 65, 80, 130), (int(cx + ox), int(cy + oy)), int(r))
        pygame.draw.circle(cloud_surf, (90, 96, 112, 60), (int(cx), int(cy)), int(radius_px * 1.15), width=3)
        surf.blit(cloud_surf, (x - cx, y - cy))

    def _draw_nofly(self, surf, px, radius_px, t):
        x, y = px
        hz = pygame.Surface((int(radius_px * 2.4), int(radius_px * 2.4)), pygame.SRCALPHA)
        c = hz.get_width() / 2
        pygame.draw.circle(hz, (220, 60, 50, 45), (int(c), int(c)), int(radius_px))
        n_stripes = 16
        for i in range(n_stripes):
            a0 = t * 0.3 + i * (2 * math.pi / n_stripes)
            a1 = a0 + math.pi / n_stripes
            if i % 2 == 0:
                p0 = (c + math.cos(a0) * radius_px, c + math.sin(a0) * radius_px)
                p1 = (c + math.cos(a1) * radius_px, c + math.sin(a1) * radius_px)
                pygame.draw.line(hz, (235, 90, 70, 230), p0, p1, width=6)
        pygame.draw.circle(hz, (235, 90, 70, 220), (int(c), int(c)), int(radius_px), width=3)
        surf.blit(hz, (x - c, y - c))
        icon = self.font_small.render("NO FLY", True, (255, 235, 235))
        surf.blit(icon, icon.get_rect(center=(int(x), int(y))))

    def _draw_drone(self, surf, px, t, has_package, heading):
        x, y = px
        body_w = 26
        arm_len = 22
        angle = heading

        def rot(ox, oy):
            ca, sa = math.cos(angle), math.sin(angle)
            return (x + ox * ca - oy * sa, y + ox * sa + oy * ca)

        arm_ends = [rot(arm_len, arm_len), rot(-arm_len, arm_len),
                    rot(arm_len, -arm_len), rot(-arm_len, -arm_len)]
        for ex, ey in arm_ends:
            pygame.draw.line(surf, (35, 38, 45), (x, y), (ex, ey), width=4)
            spin = t * 28
            for k in range(3):
                a = spin + k * (2 * math.pi / 3)
                bx = ex + math.cos(a) * 12
                by = ey + math.sin(a) * 4
                pygame.draw.line(surf, (210, 214, 222), (ex, ey), (bx, by), width=2)
            pygame.draw.circle(surf, (60, 64, 72), (int(ex), int(ey)), 5)

        body_rect_pts = [rot(body_w * 0.6, body_w * 0.35), rot(-body_w * 0.6, body_w * 0.35),
                         rot(-body_w * 0.5, -body_w * 0.4), rot(body_w * 0.5, -body_w * 0.4)]
        pygame.draw.polygon(surf, (235, 90, 60), body_rect_pts)
        pygame.draw.polygon(surf, (40, 40, 46), body_rect_pts, width=2)
        # camera/eye
        nose = rot(body_w * 0.55, 0)
        pygame.draw.circle(surf, (60, 200, 230), (int(nose[0]), int(nose[1])), 4)

        if has_package:
            pkg = rot(0, body_w * 0.55)
            pygame.draw.rect(surf, (222, 184, 135),
                              (int(pkg[0] - 8), int(pkg[1] - 6), 16, 12), border_radius=2)
            pygame.draw.line(surf, (200, 40, 40), (int(pkg[0] - 8), int(pkg[1])), (int(pkg[0] + 8), int(pkg[1])), 2)

    # ------------------------------------------------------------------ #
    # Main render
    # ------------------------------------------------------------------ #
    def handle_events(self, dt):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m or event.key == pygame.K_f:
                    self.camera.toggle_manual(self._last_drone_px)
                elif event.key == pygame.K_ESCAPE:
                    self._quit_requested = True

        keys = pygame.key.get_pressed()
        if self.camera.mode == "manual":
            dx = dy = 0.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                dx -= 1
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                dx += 1
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                dy -= 1
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                dy += 1
            if dx or dy:
                self.camera.manual_pan(dx, dy, dt)
            if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
                self.camera.manual_zoom(0.03)
            if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
                self.camera.manual_zoom(-0.03)

    def render(self, mode="human"):
        self._lazy_init()
        self._ensure_terrain()
        env = self.env

        dt = self.clock.tick(self.env.metadata["render_fps"]) / 1000.0
        dt = min(dt, 0.05)
        self.time += dt
        self._quit_requested = getattr(self, "_quit_requested", False)

        drone_px = self.grid_to_px(env.drone_pos)
        self._last_drone_px = drone_px

        if self.camera.mode == "intro" and self.camera.intro_stage == 0 and self.camera.stage_timer == 0.0 \
                and not getattr(self, "_intro_started", False):
            self.camera.start_intro(self.grid_to_px(env.hospital), self.grid_to_px(env.clinic), drone_px)
            self._intro_started = True

        self.handle_events(dt)
        self.camera.update(dt, drone_px)

        # ---- draw world ----
        self.world_surf.blit(self.terrain_surf, (0, 0))
        for m in env.mountains:
            self._draw_mountain(self.world_surf, self.grid_to_px(m), env.MOUNTAIN_RADIUS * TILE)
        self._draw_hospital(self.world_surf, self.grid_to_px(env.hospital))
        self._draw_clinic(self.world_surf, self.grid_to_px(env.clinic))
        self._draw_charging(self.world_surf, self.grid_to_px(env.charging))
        for s in env.storms:
            self._draw_storm(self.world_surf, self.grid_to_px(s), env.HAZARD_RADIUS * TILE, self.time)
        for z in env.no_fly_zones:
            self._draw_nofly(self.world_surf, self.grid_to_px(z), env.HAZARD_RADIUS * TILE, self.time)

        heading = getattr(self, "_prev_heading", 0.0)
        if self._last_drone_px is not None and hasattr(self, "_prev_drone_px"):
            ddx = drone_px[0] - self._prev_drone_px[0]
            ddy = drone_px[1] - self._prev_drone_px[1]
            if abs(ddx) > 0.01 or abs(ddy) > 0.01:
                heading = math.atan2(ddy, ddx)
        self._prev_heading = heading
        self._prev_drone_px = drone_px
        self._draw_drone(self.world_surf, drone_px, self.time, env.has_package, heading)

        # ---- camera extraction ----
        rect = self.camera.world_to_screen_rect()
        clipped = rect.clip(self.world_surf.get_rect())
        if clipped.width > 0 and clipped.height > 0:
            sub = self.world_surf.subsurface(clipped).copy()
            scale = SCREEN_W / rect.width
            target_size = (max(1, int(clipped.width * scale)), max(1, int(clipped.height * scale)))
            scaled = pygame.transform.smoothscale(sub, target_size)
            self.screen.fill(COL_BG_SKY)
            off_x = int((clipped.x - rect.x) * scale)
            off_y = int((clipped.y - rect.y) * scale)
            self.screen.blit(scaled, (off_x, off_y))
        else:
            self.screen.fill(COL_BG_SKY)

        self._draw_hud()

        if mode == "human":
            pygame.display.flip()
            return None
        else:
            arr = pygame.surfarray.array3d(self.screen)
            return np.transpose(arr, (1, 0, 2))

    def _draw_hud(self):
        env = self.env
        panel = pygame.Surface((SCREEN_W, HUD_HEIGHT), pygame.SRCALPHA)
        panel.fill((12, 15, 22, 195))
        self.screen.blit(panel, (0, 0))

        bx, by, bw, bh = 20, 20, 220, 22
        pygame.draw.rect(self.screen, (50, 54, 62), (bx, by, bw, bh), border_radius=6)
        frac = clamp(env.battery / 100.0, 0, 1)
        col = COL_BATTERY_GOOD if env.battery > 50 else (COL_BATTERY_MID if env.battery > 25 else COL_BATTERY_LOW)
        pygame.draw.rect(self.screen, col, (bx, by, int(bw * frac), bh), border_radius=6)
        pygame.draw.rect(self.screen, (230, 230, 230), (bx, by, bw, bh), width=2, border_radius=6)
        batt_txt = self.font_small.render(f"Battery {env.battery:0.0f}%", True, COL_UI_TEXT)
        self.screen.blit(batt_txt, (bx + 6, by + 2))

        pkg_txt = self.font.render(
            f"Package: {'Carrying' if env.has_package else 'None'}", True, COL_UI_TEXT)
        self.screen.blit(pkg_txt, (bx, by + 32))

        prio_col = (232, 120, 60) if env.priority == 1 else (150, 200, 255)
        prio_txt = self.font.render(
            f"Priority: {'URGENT' if env.priority == 1 else 'Normal'}", True, prio_col)
        self.screen.blit(prio_txt, (bx + 260, by))

        step_txt = self.font.render(f"Step: {env.steps}/{env.MAX_STEPS}", True, COL_UI_TEXT)
        self.screen.blit(step_txt, (bx + 260, by + 32))

        obj_name, _ = env._current_objective()
        obj_txt = self.font.render(f"Objective: {obj_name.title()}", True, (255, 230, 150))
        self.screen.blit(obj_txt, (bx + 460, by))

        rew_txt = self.font.render(f"Last reward: {env.last_reward:+.1f}", True, COL_UI_TEXT)
        self.screen.blit(rew_txt, (bx + 460, by + 32))

        event_txt = self.font.render(env.last_event, True, (200, 220, 255))
        self.screen.blit(event_txt, (bx + 700, by))

        mode_txt = self.font_small.render(
            f"Camera: {self.camera.mode.upper()}   [M] toggle manual   [WASD/Arrows] pan   [+/-] zoom",
            True, (170, 175, 185))
        self.screen.blit(mode_txt, (bx + 700, by + 32))

    def close(self):
        if self._initialized:
            pygame.quit()
            self._initialized = False
