"""
PPO (Proximal Policy Optimization) - TensorFlow 2.x / Keras implementation
-----------------------------------------------------------------------------
- Discrete action spaces (Categorical policy)
- Supports visual or vector observations (any tf.Tensor convertible dtype)
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective + value clip + entropy bonus
- Works with single-agent Gym-like env; for parameter-sharing multi-agent,
  simply treat each agent-step as an independent sample (see usage notes).

Author: ChatGPT (GPT-5 Thinking)
Date: 2025-10-03
License: MIT

Quick Start (single-agent):
---------------------------
from ppo_tf2_tensorflow_implementation import PPO, MLPPolicy, MLPValue

env = YourGymEnv()
obs_shape = env.observation_space.shape
n_actions = env.action_space.n

policy = MLPPolicy(obs_shape, n_actions)
value  = MLPValue(obs_shape)
ppo = PPO(policy, value,
          gamma=0.995, lam=0.95,
          clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3,
          ent_coef=0.01, vf_coef=0.5,
          max_grad_norm=0.5, device='/CPU:0')

rollouts = ppo.collect_rollouts(env, steps=32768, frame_skip=1)
ppo.update(rollouts, epochs=10, batch_size=65536)

Notes for parameter-sharing multi-agent:
----------------------------------------
- If env returns dicts {agent_id: obs}, {agent_id: reward}, etc.,
  flatten them into lists and push into the buffer as separate samples.
- You can adapt collect_rollouts() accordingly or feed your own rollouts
  to PPO.update() using build_rollout_dict().

"""
from __future__ import annotations
import os
import dataclasses
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
# ---------------------------------------------------------------
# Utility: set mixed precision optional (comment out if unwanted)
# ---------------------------------------------------------------
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

# ---------------------------------------------------------------
# Simple default networks (MLP / CNN)
# ---------------------------------------------------------------
class MLPPolicy(tf.keras.Model):
    def __init__(self, obs_shape: Tuple[int, ...], n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.flatten = layers.Flatten()
        self.hidden = tf.keras.Sequential([layers.Dense(h, activation='relu') for h in hidden_sizes])
        self.logits = layers.Dense(n_actions, activation=None)

    @tf.function(reduce_retracing=True)
    def call(self, obs, training=False):
        x = tf.cast(obs, tf.float32)
        x = self.flatten(x)
        x = self.hidden(x, training=training)
        return self.logits(x)

class MLPValue(tf.keras.Model):
    def __init__(self, obs_shape: Tuple[int, ...], hidden_sizes=(256, 256)):
        super().__init__()
        self.flatten = layers.Flatten()
        self.hidden = tf.keras.Sequential([layers.Dense(h, activation='relu') for h in hidden_sizes])
        self.v = layers.Dense(1, activation=None)

    @tf.function(reduce_retracing=True)
    def call(self, obs, training=False):
        x = tf.cast(obs, tf.float32)
        x = self.flatten(x)
        x = self.hidden(x, training=training)
        return tf.squeeze(self.v(x), axis=-1)

# ----- REPLACE your CNNPolicy with this -----
class CNNPolicy(tf.keras.Model):
    def __init__(self, obs_shape: Tuple[int, ...], n_actions: int):
        super().__init__()
        # 明確宣告輸入維度，避免 Autograph 亂推
        self.backbone = tf.keras.Sequential([
            layers.InputLayer(input_shape=obs_shape),            # (H,W,C) = (24,64,6)
            layers.Conv2D(32, 8, strides=4, padding='same', activation='relu'),  # -> (6,16,32)
            layers.Conv2D(64, 4, strides=2, padding='same', activation='relu'),  # -> (3,8,64)
            layers.Conv2D(64, 3, strides=1, padding='same', activation='relu'),  # -> (3,8,64)
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
        ])
        self.logits = layers.Dense(n_actions, activation=None)

    @tf.function(reduce_retracing=True)
    def call(self, obs, training=False):
        x = tf.cast(obs, tf.float32)  # NHWC
        x = self.backbone(x, training=training)
        return self.logits(x)

# ----- REPLACE your CNNValue with this -----
class CNNValue(tf.keras.Model):
    def __init__(self, obs_shape: Tuple[int, ...]):
        super().__init__()
        self.backbone = tf.keras.Sequential([
            layers.InputLayer(input_shape=obs_shape),
            layers.Conv2D(32, 8, strides=4, padding='same', activation='relu'),
            layers.Conv2D(64, 4, strides=2, padding='same', activation='relu'),
            layers.Conv2D(64, 3, strides=1, padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
        ])
        self.v = layers.Dense(1, activation=None)

    @tf.function(reduce_retracing=True)
    def call(self, obs, training=False):
        x = tf.cast(obs, tf.float32)
        x = self.backbone(x, training=training)
        return tf.squeeze(self.v(x), axis=-1)


# ---------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------
@dataclass
class Rollouts:
    obs: np.ndarray
    actions: np.ndarray
    logp: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    # computed later
    adv: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None

    def as_tf(self):
        return {k: tf.convert_to_tensor(getattr(self, k)) for k in ['obs','actions','logp','rewards','dones','values','adv','returns']}

# ---------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------
class PPO:
    def __init__(self,
                 policy: tf.keras.Model,
                 value: tf.keras.Model,
                 gamma=0.99, lam=0.95,
                 clip_ratio=0.2,
                 pi_lr=3e-4, vf_lr=1e-3,
                 ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5,
                 device: str = '/CPU:0'):
        self.policy = policy
        self.value  = value
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.pi_opt = optimizers.Adam(learning_rate=pi_lr)
        self.vf_opt = optimizers.Adam(learning_rate=vf_lr)

    # ---------------------------
    # action sampling utilities
    # ---------------------------
    @tf.function(reduce_retracing=True)
    def _policy_logits(self, obs):
        return self.policy(obs, training=False)

    @tf.function(reduce_retracing=True)
    def _value_pred(self, obs):
        return self.value(obs, training=False)

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        logits = self._policy_logits(obs[None, ...])
        dist = tf.random.categorical(logits, num_samples=1)
        action = tf.squeeze(dist, axis=-1).numpy()[0]
        logp = tf.nn.log_softmax(logits)[0, action].numpy()
        v = self._value_pred(obs[None, ...]).numpy()[0]
        return int(action), float(logp), float(v)

    # ---------------------------
    # data collection
    # ---------------------------
    def collect_rollouts(self, env, steps=2048, frame_skip: int = 1) -> Rollouts:
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        obs = env.reset()
        for t in range(steps):
            a, logp, v = self.select_action(obs)
            r_total = 0.0
            done = False
            next_obs = None
            for _ in range(frame_skip):
                next_obs, r, done, info = env.step(a)
                r_total += float(r)
                if done:
                    break
            obs_buf.append(obs)
            act_buf.append(a)
            logp_buf.append(logp)
            rew_buf.append(r_total)
            done_buf.append(float(done))
            val_buf.append(v)
            obs = env.reset() if done else next_obs
        return Rollouts(
            obs=np.array(obs_buf),
            actions=np.array(act_buf),
            logp=np.array(logp_buf, dtype=np.float32),
            rewards=np.array(rew_buf, dtype=np.float32),
            dones=np.array(done_buf, dtype=np.float32),
            values=np.array(val_buf, dtype=np.float32),
        )

    # ---------------------------
    # advantage / return via GAE
    # ---------------------------
    def _compute_gae(self, rollouts: Rollouts, last_value: float = 0.0):
        rewards = rollouts.rewards
        dones = rollouts.dones
        values = np.append(rollouts.values, last_value)
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t+1] * nonterminal - values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae
        returns = adv + rollouts.values
        # normalize adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        rollouts.adv = adv.astype(np.float32)
        rollouts.returns = returns.astype(np.float32)
        return rollouts

    # ---------------------------
    # PPO update
    # ---------------------------
    def update(self, rollouts, epochs=10, batch_size=65536, shuffle=True):
        rollouts = self._compute_gae(rollouts, last_value=0.0)
        data = rollouts.as_tf()
        N = int(data['obs'].shape[0])
        if N == 0:
            return

        for ep in range(epochs):
            idxs = tf.range(N, dtype=tf.int32)
            if shuffle:
                idxs = tf.random.shuffle(idxs)

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_idx = idxs[start:end]
                mb = {k: tf.gather(v, idxs[start:end]) for k, v in data.items()}
                self._update_minibatch(mb)



    @tf.function(reduce_retracing=True)
    def _update_minibatch(self, mb: Dict[str, tf.Tensor]):
        obs = mb['obs']
        actions = tf.cast(mb['actions'], tf.int32)
        old_logp = mb['logp']
        adv = mb['adv']
        returns = mb['returns']
        old_values = mb['values']

        with tf.GradientTape(persistent=True) as tape:
            # policy
            logits = self.policy(obs, training=True)
            log_probs_all = tf.nn.log_softmax(logits)
            logp = tf.gather_nd(log_probs_all, tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1))
            ratio = tf.exp(logp - old_logp)
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            pg_losses = -tf.minimum(ratio * adv, clipped_ratio * adv)
            pg_loss = tf.reduce_mean(pg_losses)
            entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(log_probs_all) * log_probs_all, axis=-1))
            policy_loss = pg_loss - self.ent_coef * entropy

            # value (with clip)
            values = self.value(obs, training=True)
            value_pred_clipped = old_values + tf.clip_by_value(values - old_values, -self.clip_ratio, self.clip_ratio)
            vf_losses1 = tf.square(values - returns)
            vf_losses2 = tf.square(value_pred_clipped - returns)
            vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)) * self.vf_coef

            total_loss = policy_loss + vf_loss

        pi_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        vf_grads = tape.gradient(vf_loss, self.value.trainable_variables)
        del tape

        # gradient clipping
        pi_grads, _ = tf.clip_by_global_norm(pi_grads, self.max_grad_norm)
        vf_grads, _ = tf.clip_by_global_norm(vf_grads, self.max_grad_norm)

        self.pi_opt.apply_gradients(zip(pi_grads, self.policy.trainable_variables))
        self.vf_opt.apply_gradients(zip(vf_grads, self.value.trainable_variables))

    # ---------------------------
    # Save / Load
    # ---------------------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.policy.save(os.path.join(path, "policy.keras"))
        self.value.save(os.path.join(path, "value.keras"))


    def load(self, path: str):
        self.policy = tf.keras.models.load_model(os.path.join(path, 'policy'))
        self.value  = tf.keras.models.load_model(os.path.join(path, 'value'))

# ---------------------------------------------------------------
# Helper to build rollouts from external collectors
# ---------------------------------------------------------------
def build_rollout_dict(obs, actions, logp, rewards, dones, values) -> Rollouts:
    return Rollouts(
        obs=np.asarray(obs),
        actions=np.asarray(actions),
        logp=np.asarray(logp, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32)
    )

# ---------------------------------------------------------------
# Minimal training loop example (commented)
# ---------------------------------------------------------------
"""
# Example usage with Gymnasium-like env
import gymnasium as gym

env = gym.make('CartPole-v1')
obs_shape = env.observation_space.shape
n_actions = env.action_space.n

policy = MLPPolicy(obs_shape, n_actions)
value  = MLPValue(obs_shape)
ppo = PPO(policy, value, gamma=0.995, lam=0.95, clip_ratio=0.2,
          pi_lr=3e-4, vf_lr=1e-3, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5)

for it in range(100):
    ro = ppo.collect_rollouts(env, steps=4096)
    ppo.update(ro, epochs=10, batch_size=4096)
    # quick eval
    o, _ = env.reset()
    done = False; ret = 0
    while not done:
        a, _, _ = ppo.select_action(o)
        o, r, done, _, = env.step(a)
        ret += r
    print(f"iter {it}: return={ret:.1f}")
"""
