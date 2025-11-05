"""
Training script for Pico Park 6-4 clone with PPO (TensorFlow 2)
----------------------------------------------------------------
- Uses env_pico64.Pico64Env + make_map_cfg
- Uses PPO + CNNPolicy/CNNValue from ppo_tf2_tensorflow_implementation
- Includes simple curriculum (C0 -> C1 -> C2 -> C3) and periodic eval/save

Run:
  python train_pico64.py

Author: ChatGPT (GPT-5 Thinking)
Date: 2025-10-03
"""
import os
import time
import numpy as np
import tensorflow as tf

from env_pico64 import Pico64Env, make_map_cfg
from PPO import PPO, CNNPolicy, CNNValue

# ---------------------
# Config
# ---------------------
LEVELS = ["EASY", "C0", "C1", "C2", "C3"]
FRAMESKIP = 2
MAX_STEPS = 200
STEPS_PER_ITER = 16384
EPOCHS = 10
BATCH_SIZE = 4096
EVAL_EPISODES = 5
SAVE_EVERY = 5
OUTDIR = "runs/pico64"


os.makedirs(OUTDIR, exist_ok=True)

# ---------------------
# Helpers
# ---------------------
def make_env(level: str):
    cfg = make_map_cfg(level)
    return Pico64Env(cfg, frame_skip=FRAMESKIP, max_steps=MAX_STEPS, teammate_policy="scripted")

@tf.function(reduce_retracing=True)
def _noop(x):
    return x


def evaluate(env: Pico64Env, ppo: PPO, episodes=EVAL_EPISODES):
    returns = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ret = 0.0
        while not done:
            a, _, _ = ppo.select_action(obs)
            obs, r, done, _ = env.step(a)
            ret += r
        returns.append(ret)
    return float(np.mean(returns)), float(np.std(returns))

# ---------------------
# Train
# ---------------------
if __name__ == "__main__":
    # Build env & models
    env = make_env(LEVELS[0])
    obs_shape = env.observation_shape  # (H,W,C)
    n_actions = env.n_actions

    policy = CNNPolicy(obs_shape, n_actions)
    value  = CNNValue(obs_shape)

    ppo = PPO(policy, value,
          gamma=0.99, lam=0.95,
          clip_ratio=0.2,
          pi_lr=3e-4, vf_lr=1e-3,
          ent_coef=0.03,   # 原本 0.01 → 0.03
          vf_coef=0.5,
          max_grad_norm=0.5)


    start_time = time.time()
    global_iter = 0
    for lvl in LEVELS:
        print(f"\n===== Start curriculum level: {lvl} =====")
        env = make_env(lvl)

        for local_iter in range(10000):  # up to 100 iters per level (adjust as needed)
            global_iter += 1
            ro = ppo.collect_rollouts(env, steps=STEPS_PER_ITER, frame_skip=1)
            ppo.update(ro, epochs=EPOCHS, batch_size=BATCH_SIZE)

            mean_ret, std_ret = evaluate(env, ppo, episodes=EVAL_EPISODES)
            elapsed = time.time() - start_time
            print(f"[lvl {lvl}] iter {global_iter:04d}  eval_return={mean_ret:.1f}±{std_ret:.1f}  elapsed={elapsed/60:.1f}m")

            # Simple promotion rule: if avg return exceeds 60 at early levels or 120 at final
            threshold = 60.0 if lvl != "C3" else 120.0
            if mean_ret >= threshold:
                print(f"Promote from {lvl} at iter {global_iter}, eval_return={mean_ret:.1f}")
                break

            if global_iter % SAVE_EVERY == 0:
                save_dir = os.path.join(OUTDIR, f"lvl_{lvl}_iter_{global_iter}")
                try:
                    ppo.save(save_dir)
                    print(f"Saved checkpoint: {save_dir}")
                except Exception as e:
                    print(f"Save failed: {e}")

    # Final save
    final_dir = os.path.join(OUTDIR, "final")
    try:
        ppo.save(final_dir)
        print(f"Training done. Saved final to: {final_dir}")
    except Exception as e:
        print(f"Final save failed: {e}")
