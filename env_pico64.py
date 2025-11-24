"""
Pico Park 6-4 (DON'T PUSH) - Minimal Trainable Environment (Grid/CNN)
---------------------------------------------------------------------
- 2 agents on a tile-map with platforms, deadly buttons, one key, one door
- Single-agent RL with a scripted teammate by default (parameter-sharing later)
- Observations: H x W x C tensor (float32) suitable for CNNPolicy
- Actions: 0 noop, 1 left, 2 right, 3 jump, 4 wait
- Reward shaping aligned to our design note

This is a *training-friendly clone* of the PICO PARK 6-4 mechanics, not the
original game. Physics is simplified but includes gravity and jumps.

Usage with the provided PPO:
----------------------------
from env_pico64 import Pico64Env, make_map_cfg
from ppo_tf2_tensorflow_implementation import PPO, CNNPolicy, CNNValue

cfg = make_map_cfg("C3")  # curriculum level (C0..C3)
env = Pico64Env(cfg, frame_skip=4, max_steps=600, teammate_policy="scripted")

obs_shape = env.observation_shape
n_actions = env.n_actions
policy = CNNPolicy(obs_shape, n_actions)
value  = CNNValue(obs_shape)
ppo = PPO(policy, value, gamma=0.995, lam=0.95, clip_ratio=0.2,
          pi_lr=3e-4, vf_lr=1e-3, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)

rollouts = ppo.collect_rollouts(env, steps=32768, frame_skip=1)
ppo.update(rollouts, epochs=10, batch_size=65536)

Notes:
- For multi-agent parameter sharing, you can create two env copies with different
  teammate snapshots or modify collect_rollouts to push both agents' steps.
- The observation returns a (H, W, C) tensor. Default H=24, W=64, C=6.

Author: ChatGPT (GPT-5 Thinking)
Date: 2025-10-03
License: MIT
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

# -----------------------------
# Tiles & constants
# -----------------------------
EMPTY, SOLID, BUTTON, KEY, DOOR = 0, 1, 2, 3, 4

@dataclass
class Agent:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = False
    has_key: bool = False

class Pico64Env:
    """Grid-based clone of Pico Park 6-4 for RL training."""
    def __init__(self,
                 map_cfg: Dict[str, Any],
                 frame_skip: int = 4,
                 max_steps: int = 600,
                 teammate_policy: str = "scripted",
                 seed: Optional[int] = None):
        self.map_cfg = map_cfg
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.teammate_policy = teammate_policy  # "scripted" or "idle"
        self.rng = np.random.RandomState(seed)
        self.dt = 1.0
        self.g = 0.8
        self.move_speed = 0.8
        self.jump_speed = 3.2
        self.term_vy = 6.0
        self.h, self.w = map_cfg['grid'].shape
        self.channels = 6  # [solid, button, key, door, p1, p2]
        self.observation_shape = (self.h, self.w, self.channels)
        self.n_actions = 5  # noop, left, right, jump, wait
        self.reset()

    # -------------- Gym-like API --------------
    def reset(self):
        self.grid = self.map_cfg['grid'].copy()
        self.spawn1 = tuple(self.map_cfg['spawn1'])
        self.spawn2 = tuple(self.map_cfg['spawn2'])
        self.key_pos = tuple(self.map_cfg['key'])
        self.door_pos = tuple(self.map_cfg['door'])
        self.agents = [Agent(*self.spawn1), Agent(*self.spawn2)]
        self.agents[0].has_key = False
        self.agents[1].has_key = False
        self.t = 0
        self.done = False
        self.reached_high = False
        return self._obs()

    def step(self, action: int):
        """Single-agent control (agent 0), teammate runs scripted policy."""
        if self.done:
            return self._obs(), 0.0, True, {}
        r_total = 0.0
        info = {}
        for _ in range(self.frame_skip):
            # 1) Apply actions (agent 0 from input, agent 1 scripted)
            a0 = action
            a1 = self._scripted_teammate()
            self._apply_action(0, a0)
            self._apply_action(1, a1)

            # 2) Physics & collisions
            self._physics_step()

            # 3) Events: button death, key pickup, goal
            death = self._check_buttons()
            if death:
                r_total += -50.0
                self.done = True
                break
            key_gain = self._check_key_pickup()
            if key_gain:
                r_total += 10.0
            # team formation under key
            if self._is_stack_under_key():
                r_total += 5.0
            # small time penalty
            r_total += -0.01

            agent0 = self.agents[0]
            x = agent0.x
            y = agent0.y
            h = self.grid.shape[0]

            if not getattr(self, "reached_high", False) and y <= (h - 7 + 0.5) and x >= 20:
                r_total += 30.0
                self.reached_high = True

            # distance shaping to key/door
            r_total += 0.02 * self._delta_goal_progress()

            # success
            if self._check_goal():
                r_total += 100.0
                self.done = True
                break

            self.t += 1
            if self.t >= self.max_steps:
                self.done = True
                break
        return self._obs(), float(r_total), self.done, info

    # -------------- Internals --------------
    def _obs(self) -> np.ndarray:
        ch = np.zeros((self.h, self.w, self.channels), dtype=np.float32)
        ch[..., 0] = (self.grid == SOLID).astype(np.float32)
        ch[..., 1] = (self.grid == BUTTON).astype(np.float32)
        # key/door dynamic masks (key vanishes when taken)
        has_key = any(a.has_key for a in self.agents)
        if not has_key:
            ky = int(self.key_pos[1]); kx = int(self.key_pos[0])
            ch[ky, kx, 2] = 1.0
        dy = int(self.door_pos[1]); dx = int(self.door_pos[0])
        ch[dy, dx, 3] = 1.0
        # agents
        ay0 = max(0, min(self.h-1, int(round(self.agents[0].y))))
        ax0 = max(0, min(self.w-1, int(round(self.agents[0].x))))
        ay1 = max(0, min(self.h-1, int(round(self.agents[1].y))))
        ax1 = max(0, min(self.w-1, int(round(self.agents[1].x))))
        ch[ay0, ax0, 4] = 1.0
        ch[ay1, ax1, 5] = 1.0
        return ch

    def _scripted_teammate(self) -> int:
        if self.teammate_policy == "idle":
            return 4  # wait
        # simple rule-based: move toward key until taken; then toward door
        target = self.key_pos
        if any(a.has_key for a in self.agents):
            target = self.door_pos
        a = self.agents[1]
        dx = target[0] - a.x
        if abs(dx) > 0.5:
            return 2 if dx > 0 else 1  # right/left
        # try to jump if below target (crude)
        if (target[1] < a.y) and a.on_ground:
            return 3
        return 4  # wait

    def _apply_action(self, i: int, action: int):
        a = self.agents[i]
        if action == 1:   # left
            a.vx = -self.move_speed
        elif action == 2: # right
            a.vx = self.move_speed
        else:
            a.vx = 0.0
        if action == 3 and a.on_ground:  # jump
            a.vy = -self.jump_speed
        # action 0 noop / 4 wait -> minimal change

    def _physics_step(self):
        for a in self.agents:
            # gravity
            a.vy = min(a.vy + self.g * self.dt, self.term_vy)
            new_x = a.x + a.vx * self.dt
            new_y = a.y + a.vy * self.dt
            # horizontal collision
            if self._is_solid(int(round(new_x)), int(round(a.y))):
                new_x = a.x
            # vertical collision
            if self._is_solid(int(round(a.x)), int(round(new_y))):
                # landing
                if a.vy > 0:
                    a.on_ground = True
                a.vy = 0.0
                new_y = a.y
            else:
                a.on_ground = False
            # stacking (simplified): if two agents overlap vertically, top stands on bottom
            if abs(new_x - self.agents[0 if a is self.agents[1] else 1].x) < 0.6:
                other = self.agents[0 if a is self.agents[1] else 1]
                if new_y < other.y - 1.0 and (other.y - new_y) < 1.2 and other.on_ground:
                    # allow climb by jumping
                    pass
            a.x, a.y = np.clip(new_x, 0, self.w-1), np.clip(new_y, 0, self.h-1)

    def _is_solid(self, x: int, y: int) -> bool:
        x = max(0, min(self.w-1, x)); y = max(0, min(self.h-1, y))
        return self.grid[y, x] == SOLID

    def _check_buttons(self) -> bool:
        for a in self.agents:
            gx, gy = int(round(a.x)), int(round(a.y))
            if self.grid[gy, gx] == BUTTON:
                return True
        return False

    def _check_key_pickup(self) -> bool:
        ky, kx = int(self.key_pos[1]), int(self.key_pos[0])
        if any(ag.has_key for ag in self.agents):
            return False
        for i, a in enumerate(self.agents):
            if int(round(a.x)) == kx and int(round(a.y)) == ky:
                a.has_key = True
                return True
        return False

    def _check_goal(self) -> bool:
        dy, dx = int(self.door_pos[1]), int(self.door_pos[0])
        # both agents at door; one must have the key
        cond = []
        for a in self.agents:
            cond.append(int(round(a.x)) == dx and int(round(a.y)) == dy)
        if all(cond) and any(a.has_key for a in self.agents):
            return True
        return False

    def _is_stack_under_key(self) -> bool:
        ky, kx = int(self.key_pos[1]), int(self.key_pos[0])
        # if agents occupy (kx, ky+1) and (kx, ky+2) within small tolerance
        xs = [int(round(a.x)) for a in self.agents]
        ys = [int(round(a.y)) for a in self.agents]
        if xs[0] == xs[1] == kx and ((ys[0] == ky + 1 and ys[1] == ky + 2) or (ys[1] == ky + 1 and ys[0] == ky + 2)):
            return True
        return False

    def _delta_goal_progress(self) -> float:
        # shaping: average inverse distance decrease to current target (key until taken, then door)
        tgt = self.key_pos if not any(a.has_key for a in self.agents) else self.door_pos
        def d(a):
            return abs(tgt[0]-a.x) + abs(tgt[1]-a.y)
        d0 = d(self.agents[0]); d1 = d(self.agents[1])
        if not hasattr(self, '_prev_d'): self._prev_d = (d0, d1)
        prev0, prev1 = self._prev_d
        delta = (prev0 - d0 + prev1 - d1) * 0.5
        self._prev_d = (d0, d1)
        return float(delta)

# -----------------------------
# Curriculum map generators
# -----------------------------
def make_map_cfg(level: str = "C0") -> Dict[str, Any]:
    """Return a dict with keys: grid(h,w), spawn1(x,y), spawn2(x,y), key(x,y), door(x,y)
    Coordinates are float (x,y) where x->columns (0..w-1), y->rows (0..h-1)."""
    h, w = 24, 64
    grid = np.zeros((h, w), dtype=np.int32)
    # floor
    grid[h-1, :] = SOLID

    if level == "C0":
        platforms = [
            (h-5, 4, 40)
        ]
        for y, x0, x1 in platforms:
            grid[y, x0:x1] = SOLID
        spawn1 = (6.0, h-2.0); spawn2 = (8.0, h-2.0)
        key = (20.0, h-6.0)
        door = (30.0, h-6.0)
        return {'grid': grid, 'spawn1': spawn1, 'spawn2': spawn2, 'key': key, 'door': door}

    elif level == "C1":
        # Sparse buttons far from path
        platforms = [
            (h-5, 4, 20), (h-7, 20, 40)
        ]
        for y, x0, x1 in platforms:
            grid[y, x0:x1] = SOLID
        spawn1 = (6.0, h-6.0); spawn2 = (8.0, h-6.0)
        key = (26.0, h-8.0)
        door = (34.0, h-8.0)
        return {'grid': grid, 'spawn1': spawn1, 'spawn2': spawn2, 'key': key, 'door': door}

    elif level == "C2":
        # Corridor with button clusters
        platforms = [
            (h-5, 4, 20), (h-7, 20, 40)
        ]
        for y, x0, x1 in platforms:
            grid[y, x0:x1] = SOLID
        spawn1 = (6.0, h-6.0); spawn2 = (8.0, h-6.0)
        key = (26.0, h-8.0)
        door = (34.0, h-8.0)
        button_x = 24
        button_y = h-8        # 跟高平台同一高度
        grid[button_y, button_x] = BUTTON
        return {'grid': grid, 'spawn1': spawn1, 'spawn2': spawn2, 'key': key, 'door': door}

    elif level == "EASY":
        platforms = [(h-5, 4, 40)]
        for y, x0, x1 in platforms:
            grid[y, x0:x1] = SOLID
        spawn1 = (6.0,  h-2.0);  spawn2 = (8.0,  h-2.0)
        key    = (14.0, h-6.0)   # 很近
        door   = (22.0, h-6.0)   # 也很近
        return {'grid': grid, 'spawn1': spawn1, 'spawn2': spawn2, 'key': key, 'door': door}

    return {
        'grid': grid,
        'spawn1': spawn1,
        'spawn2': spawn2,
        'key': key,
        'door': door,
    }
