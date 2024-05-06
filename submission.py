from __future__ import annotations

import argparse
import copy
import os
import time
from collections import deque
from functools import partial
from typing import Any, Callable, Dict, List, Optional, SupportsFloat, Tuple

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from einops import rearrange
from flax import struct
from flax.training.train_state import TrainState
from gymnasium.core import ActType, ObsType
from jaxtyping import PRNGKeyArray

import wandb

"""
This code is adapted from https://github.com/ShaneFlandermeyer/tdmpc2-jax
TD-MPC2 Algorithm: 
    Nicklas Hansen, Hao Su, and Xiaolong Wang. 
    "Td-mpc2: Scalable, robust world models for continuous control”. 
    In: arXiv preprint arXiv:2310.16828 (2023) 
"""

# You will need to replace this value if you want to run the code, everything else should work as is.
WANDB_ENTITY = "ejex"

# jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_triton_softmax_fusion=true "
#     "--xla_gpu_triton_gemm_any=True "
#     "--xla_gpu_enable_async_collectives=true "
#     "--xla_gpu_enable_latency_hiding_scheduler=true "
#     "--xla_gpu_enable_highest_priority_async_stream=true "
# )

def mish(x: jax.Array) -> jax.Array:
    return x * jnp.tanh(jnp.log(1 + jnp.exp(x)))


def simnorm(x: jax.Array, simplex_dim: int = 8) -> jax.Array:
    x = rearrange(x, "...(L V) -> ... L V", V=simplex_dim)
    x = jax.nn.softmax(x, axis=-1)
    return rearrange(x, "... L V -> ... (L V)")


def huber_loss(x: jax.Array, delta: float = 1.0) -> jax.Array:
    return jnp.where(jnp.abs(x) < delta, 0.5 * x**2, delta * (jnp.abs(x) - 0.5 * delta))


def mse_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((pred - target) ** 2)


def mae_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(jnp.abs(pred - target))


def soft_crossentropy(
    pred_logits: jax.Array, target: jax.Array, low: float, high: float, num_bins: int
) -> jax.Array:
    pred = jax.nn.log_softmax(pred_logits, axis=-1)
    target = two_hot(target, low, high, num_bins)
    return -(pred * target).sum(axis=-1)


def binary_crossentropy(pred_logits: jax.Array, target: jax.Array) -> jax.Array:
    pred = jax.nn.sigmoid(pred_logits)
    return -jnp.mean(target * jnp.log(pred) + (1 - target) * jnp.log(1 - pred))


def percentile(x: jax.Array, q: jax.Array) -> jax.Array:
    x_dtype, x_shape = x.dtype, x.shape
    x = x.reshape(x.shape[0], -1)
    in_sorted = jnp.sort(x, axis=0)
    positions = q * (x.shape[0] - 1) / 100
    floored = jnp.floor(positions)
    ceiled = floored + 1
    # Replace below with jnp.where
    ceiled = jnp.where(ceiled > x.shape[0] - 1, x.shape[0] - 1, ceiled)
    weight_ceiled = positions
    weight_floored = 1.0 - weight_ceiled
    d0 = in_sorted[floored.astype(jnp.int32), :] * weight_floored[:, None]
    d1 = in_sorted[ceiled.astype(jnp.int32), :] * weight_ceiled[:, None]
    return (d0 + d1).reshape((-1, *x_shape[1:])).astype(x_dtype)


# Normalize input values using a running scale of the range between a given range of percentiles.
def percentile_normalization(
    x: jax.Array,
    prev_scale: jax.Array,
    percentile_range: jax.Array = jnp.array([5, 95]),
    tau: float = 0.01,
) -> jax.Array:
    # Compute percentiles for the input values.
    percentiles = percentile(x, percentile_range)
    scale = percentiles[1] - percentiles[0]

    return tau * scale + (1 - tau) * prev_scale


def mean_std_normalization(
    x: jax.Array, prev_scale: jax.Array, tau: float = 0.01
) -> jax.Array:
    mean = jnp.mean(x)
    std = jnp.std(x)
    scale = jnp.array([mean, std])

    return tau * scale + (1 - tau) * prev_scale


def symlog(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def two_hot(x: jax.Array, low: float, high: float, num_bins: int) -> jax.Array:
    """
    Generate two-hot encoded tensor from input tensor.

    Parameters
    ----------
    x : jax.Array
        Input tensor of continuous values. Shape: (*batch_dim, num_values)
        Should **not** have a leading singleton dimension at the end.
    low : float
        Minimum value under consideration in log-space
    high : float
        Maximum value under consideration in log-space
    num_bins : int
        Number of encoding bins

    Returns
    -------
    jax.Array
        _description_
    """
    bin_size = (high - low) / (num_bins - 1)

    x = jnp.clip(symlog(x), low, high)
    bin_index = jnp.floor((x - low) / bin_size).astype(int)
    bin_offset = (x - low) / bin_size - bin_index.astype(float)

    # Two-hot encode
    two_hot = (
        jax.nn.one_hot(bin_index, num_bins) * (1 - bin_offset[..., None])
        + jax.nn.one_hot(bin_index + 1, num_bins) * bin_offset[..., None]
    )

    return two_hot


def two_hot_inv(
    x: jax.Array, low: float, high: float, num_bins: int, apply_softmax: bool = True
) -> jax.Array:
    bins = jnp.linspace(low, high, num_bins)

    if apply_softmax:
        x = jax.nn.softmax(x, axis=-1)

    x = jnp.sum(x * bins, axis=-1)
    return symexp(x)


def sg(x):
    return jax.tree.map(jax.lax.stop_gradient, x)


class EpisodicReplayBuffer:
    def __init__(
        self,
        capacity: int,
        dummy_input: Dict,
        seed: Optional[int] = None,
        respect_episode_boundaries: bool = True,
    ):
        self.capacity = capacity
        self.data = jax.tree.map(
            lambda x: np.empty((capacity,) + np.asarray(x).shape, np.asarray(x).dtype),
            dummy_input,
        )

        self.respect_episode_boundaries = respect_episode_boundaries
        self.last_episode_ind = None
        self.episode_starts = deque()
        self.episode_counts = deque()

        self.size = 0
        self.current_ind = 0

        self.np_random = np.random.RandomState(seed=seed)

    def __len__(self):
        return self.size

    def insert(self, data: Dict, episode_index: int) -> None:
        # Insert the data
        jax.tree.map(lambda x, y: x.__setitem__(self.current_ind, y), self.data, data)

        # Remove the oldest episode information if it gets overwritten
        if self.size == self.capacity:
            self.episode_counts[0] -= 1
            if self.episode_counts[0] == 0:
                self.episode_starts.popleft()
                self.episode_counts.popleft()

        # Increment episode information
        if episode_index == self.last_episode_ind:
            self.episode_counts[-1] += 1
        else:
            self.last_episode_ind = episode_index
            self.episode_starts.append(self.current_ind)
            self.episode_counts.append(1)

        self.current_ind = (self.current_ind + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, sequence_length: int) -> Dict:
        if self.respect_episode_boundaries:
            episode_starts, counts = (
                np.array(self.episode_starts),
                np.array(self.episode_counts),
            )
            valid = counts >= sequence_length
            episode_starts, counts = episode_starts[valid], counts[valid]

            # Sample subsequences uniformly within episodes
            episode_inds = self.np_random.randint(0, len(counts), size=batch_size)
            sequence_starts = np.round(
                self.np_random.rand(batch_size)
                * (counts[episode_inds] - sequence_length)
            ).astype(int)
            buffer_starts = episode_starts[episode_inds] + sequence_starts
        else:
            buffer_starts = self.np_random.randint(0, self.size, size=batch_size)

        sequence_inds = buffer_starts[:, None] + np.arange(sequence_length)
        sequence_inds = sequence_inds % self.capacity

        batch = jax.tree.map(lambda x: np.swapaxes(x[sequence_inds], 0, 1), self.data)

        batch["reward"] = np.nan_to_num(batch["reward"])

        return batch


class Ensemble(nn.Module):
    base_module: nn.Module
    num: int = 2

    @nn.compact
    def __call__(self, *args, **kwargs):
        ensemble = nn.vmap(
            self.base_module,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args, **kwargs)


class NormedLinear(nn.Module):
    features: int
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    dropout_rate: Optional[float] = None

    kernel_init: Callable = partial(nn.initializers.truncated_normal, stddev=0.02)
    dtype: jnp.dtype = jnp.float32  # Switch this to bfloat16 for speed
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        x = nn.Dense(
            features=self.features,
            kernel_init=self.kernel_init(),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)

        x = nn.LayerNorm()(x)
        x = self.activation(x)

        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        return x


class WorldModel(struct.PyTreeNode):
    # Models
    encoder: TrainState
    dynamics_model: TrainState
    reward_model: TrainState
    policy_model: TrainState
    value_model: TrainState
    target_value_model: TrainState
    continue_model: TrainState
    # Spaces
    observation_space: gym.Space = struct.field(pytree_node=False)
    action_space: gym.Space = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    # Architecture
    mlp_dim: int = struct.field(pytree_node=False)
    latent_dim: int = struct.field(pytree_node=False)
    num_value_nets: int = struct.field(pytree_node=False)
    num_bins: int = struct.field(pytree_node=False)
    symlog_min: float
    symlog_max: float
    predict_continues: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        # Spaces
        observation_space: gym.Space,
        action_space: gym.Space,
        # Models
        encoder_module: nn.Module,
        # Architecture
        mlp_dim: int,
        latent_dim: int,
        value_dropout: float,
        num_value_nets: int,
        num_bins: int,
        symlog_min: float,
        symlog_max: float,
        simnorm_dim: int,
        predict_continues: bool,
        # Optimization
        learning_rate: float,
        encoder_learning_rate: float,
        max_grad_norm: float = 10,
        # Misc
        tabulate: bool = False,
        dtype: jnp.dtype = jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        dynamics_key, reward_key, value_key = jax.random.split(key, 3)

        action_dim = np.prod(action_space.shape)

        encoder = TrainState.create(
            apply_fn=encoder_module.apply,
            params=encoder_module.init(key, observation_space.sample())["params"],
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(encoder_learning_rate),
            ),
        )

        # Latent forward dynamics model
        dynamics_module = nn.Sequential(
            [
                NormedLinear(mlp_dim, activation=mish, dtype=dtype),
                NormedLinear(mlp_dim, activation=mish, dtype=dtype),
                NormedLinear(
                    latent_dim,
                    activation=partial(simnorm, simplex_dim=simnorm_dim),
                    dtype=dtype,
                ),
            ]
        )

        dynamics_model = TrainState.create(
            apply_fn=dynamics_module.apply,
            params=dynamics_module.init(
                dynamics_key, jnp.zeros(latent_dim + action_dim)
            )["params"],
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate),
            ),
        )

        # Transition reward model
        reward_module = nn.Sequential(
            [
                NormedLinear(mlp_dim, activation=mish, dtype=dtype),
                NormedLinear(mlp_dim, activation=mish, dtype=dtype),
                nn.Dense(num_bins, kernel_init=nn.initializers.zeros),
            ]
        )
        reward_model = TrainState.create(
            apply_fn=reward_module.apply,
            params=reward_module.init(reward_key, jnp.zeros(latent_dim + action_dim))[
                "params"
            ],
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate),
            ),
        )

        # Policy model
        policy_module = nn.Sequential(
            [
                NormedLinear(mlp_dim, activation=mish, dtype=dtype),
                NormedLinear(mlp_dim, activation=mish, dtype=dtype),
                nn.Dense(
                    2 * action_dim, kernel_init=nn.initializers.truncated_normal(0.02)
                ),
            ]
        )
        policy_model = TrainState.create(
            apply_fn=policy_module.apply,
            params=policy_module.init(key, jnp.zeros(latent_dim))["params"],
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate, eps=1e-5),
            ),
        )

        # Return/value model (ensemble)
        value_param_key, value_dropout_key = jax.random.split(value_key)
        value_base = partial(
            nn.Sequential,
            [
                NormedLinear(
                    mlp_dim, activation=mish, dropout_rate=value_dropout, dtype=dtype
                ),
                NormedLinear(mlp_dim, activation=mish, dtype=dtype),
                nn.Dense(num_bins, kernel_init=nn.initializers.zeros),
            ],
        )
        value_ensemble = Ensemble(value_base, num=num_value_nets)
        value_model = TrainState.create(
            apply_fn=value_ensemble.apply,
            params=value_ensemble.init(
                {"params": value_param_key, "dropout": value_dropout_key},
                jnp.zeros(latent_dim + action_dim),
            )["params"],
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate),
            ),
        )
        target_value_model = TrainState.create(
            apply_fn=value_ensemble.apply,
            params=copy.deepcopy(value_model.params),
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        if predict_continues:
            continue_module = nn.Sequential(
                [
                    NormedLinear(mlp_dim, activation=mish, dtype=dtype),
                    NormedLinear(mlp_dim, activation=mish, dtype=dtype),
                    nn.Dense(1, kernel_init=nn.initializers.zeros),
                ]
            )
            continue_model = TrainState.create(
                apply_fn=continue_module.apply,
                params=continue_module.init(key, jnp.zeros(latent_dim))["params"],
                tx=optax.chain(
                    optax.clip_by_global_norm(max_grad_norm),
                    optax.adam(learning_rate),
                ),
            )
        else:
            continue_model = None

        return cls(
            # Spaces
            observation_space=observation_space,
            action_space=action_space,
            action_dim=action_dim,
            # Models
            encoder=encoder,
            dynamics_model=dynamics_model,
            reward_model=reward_model,
            policy_model=policy_model,
            value_model=value_model,
            target_value_model=target_value_model,
            continue_model=continue_model,
            # Architecture
            mlp_dim=mlp_dim,
            latent_dim=latent_dim,
            num_value_nets=num_value_nets,
            num_bins=num_bins,
            symlog_min=float(symlog_min),
            symlog_max=float(symlog_max),
            predict_continues=predict_continues,
        )

    @jax.jit
    def encode(self, obs: np.ndarray, params: Dict) -> jax.Array:
        return self.encoder.apply_fn(
            {"params": params}, jax.tree.map(lambda x: symlog(x), obs)
        )

    @jax.jit
    def next(self, z: jax.Array, a: jax.Array, params: Dict) -> jax.Array:
        # Apply the dynamics model to calculate change in latent state
        return (
            self.dynamics_model.apply_fn(
                {"params": params}, jnp.concatenate([z, a], axis=-1)
            )
            + z
        )

    @jax.jit
    def reward(
        self, z: jax.Array, a: jax.Array, params: Dict
    ) -> Tuple[jax.Array, jax.Array]:
        z = jnp.concatenate([z, a], axis=-1)
        logits = self.reward_model.apply_fn({"params": params}, z)
        reward = two_hot_inv(logits, self.symlog_min, self.symlog_max, self.num_bins)
        return reward, logits

    @jax.jit
    def sample_actions(
        self,
        z: jax.Array,
        params: Dict,
        min_log_std: float = -10,
        max_log_std: float = 2,
        *,
        key: PRNGKeyArray,
    ) -> Tuple[jax.Array, ...]:
        # Chunk the policy model output to get mean and logstd
        mu, log_std = jnp.split(
            self.policy_model.apply_fn({"params": params}, z), 2, axis=-1
        )
        log_std = min_log_std + 0.5 * (max_log_std - min_log_std) * (
            jnp.tanh(log_std) + 1
        )

        # Sample action and compute logprobs
        eps = jax.random.normal(key, mu.shape)
        x_t = mu + eps * jnp.exp(log_std)
        residual = (-0.5 * eps**2 - log_std).sum(-1)
        log_probs = x_t.shape[-1] * (residual - 0.5 * jnp.log(2 * jnp.pi))

        # Squash tanh
        mean = jnp.tanh(mu)
        action = jnp.tanh(x_t)
        log_probs -= jnp.log((1 - action**2) + 1e-6).sum(-1)

        return action, mean, log_std, log_probs

    @jax.jit
    def Q(
        self, z: jax.Array, a: jax.Array, params: Dict, key: PRNGKeyArray
    ) -> Tuple[jax.Array, jax.Array]:
        z = jnp.concatenate([z, a], axis=-1)
        logits = self.value_model.apply_fn({"params": params}, z, rngs={"dropout": key})

        Q = two_hot_inv(logits, self.symlog_min, self.symlog_max, self.num_bins)
        return Q, logits


class TDMPC2(struct.PyTreeNode):
    model: WorldModel
    scale: jax.Array

    # Planning
    mpc: bool
    horizon: int = struct.field(pytree_node=False)
    mppi_iterations: int = struct.field(pytree_node=False)
    population_size: int = struct.field(pytree_node=False)
    policy_prior_samples: int = struct.field(pytree_node=False)
    num_elites: int = struct.field(pytree_node=False)
    min_plan_std: float
    max_plan_std: float
    temperature: float

    # Optimization
    batch_size: int = struct.field(pytree_node=False)
    discount: float
    rho: float
    consistency_coef: float
    reward_coef: float
    value_coef: float
    continue_coef: float
    entropy_coef: float
    tau: float

    @classmethod
    def create(
        cls,
        world_model: WorldModel,
        # Planning
        mpc: bool,
        horizon: int,
        mppi_iterations: int,
        population_size: int,
        policy_prior_samples: int,
        num_elites: int,
        min_plan_std: float,
        max_plan_std: float,
        temperature: float,
        # Optimization
        discount: float,
        batch_size: int,
        rho: float,
        consistency_coef: float,
        reward_coef: float,
        value_coef: float,
        continue_coef: float,
        entropy_coef: float,
        tau: float,
    ):
        return cls(
            model=world_model,
            mpc=mpc,
            horizon=horizon,
            mppi_iterations=mppi_iterations,
            population_size=population_size,
            policy_prior_samples=policy_prior_samples,
            num_elites=num_elites,
            min_plan_std=min_plan_std,
            max_plan_std=max_plan_std,
            temperature=temperature,
            discount=discount,
            batch_size=batch_size,
            rho=rho,
            consistency_coef=consistency_coef,
            reward_coef=reward_coef,
            value_coef=value_coef,
            continue_coef=continue_coef,
            entropy_coef=entropy_coef,
            tau=tau,
            scale=jnp.array([1.0]),
        )

    def act(
        self,
        obs: np.ndarray,
        prev_plan: jax.Array = None,
        train: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        z = self.model.encode(obs, self.model.encoder.params)
        z = jnp.atleast_2d(z)

        if self.mpc:
            action, plan = self.plan(z, prev_plan=prev_plan, train=train, key=key)
        else:
            action = self.model.sample_actions(
                z, self.model.policy_model.params, key=key
            )[0].squeeze()
            plan = None

        return np.array(action), plan

    @jax.jit
    def plan(
        self,
        z: jax.Array,
        prev_plan: Tuple[jax.Array, jax.Array] = None,
        train: bool = False,
        *,
        key: PRNGKeyArray,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Select next action via MPPI planner

        Parameters
        ----------
        z : jax.Array
            Enncoded environment observation
        key : PRNGKeyArray
            Jax PRNGKey
        prev_mean : jax.Array, optional
            Mean from previous planning interval. If present, MPPI is given a warm start by time-shifting this value by 1 step. If None, the MPPI mean is set to zero, by default None
        train : bool, optional
            If True, inject noise into the final selected action, by default False

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            - Action output from planning
            - Final mean value (for use in warm start)
        """
        # Sample trajectories from policy prior
        key, *prior_keys = jax.random.split(key, self.horizon + 1)
        policy_actions = jnp.empty(
            (self.horizon, self.policy_prior_samples, self.model.action_dim)
        )
        _z = z.repeat(self.policy_prior_samples, axis=0)
        for t in range(self.horizon - 1):
            policy_actions = policy_actions.at[t].set(
                self.model.sample_actions(
                    _z, self.model.policy_model.params, key=prior_keys[t]
                )[0]
            )
            _z = self.model.next(
                _z, policy_actions[t], self.model.dynamics_model.params
            )
        policy_actions = policy_actions.at[-1].set(
            self.model.sample_actions(
                _z, self.model.policy_model.params, key=prior_keys[-1]
            )[0]
        )

        # Initialize population state
        z = z.repeat(self.population_size, axis=0)
        mean = jnp.zeros((self.horizon, self.model.action_dim))
        std = self.max_plan_std * jnp.ones((self.horizon, self.model.action_dim))

        # Warm start MPPI with the previous solution
        if prev_plan is not None:
            prev_mean, prev_std = prev_plan
            mean = mean.at[:-1].set(prev_mean[1:])
        #   std = std.at[:-1].set(prev_std[1:])

        actions = jnp.empty((self.horizon, self.population_size, self.model.action_dim))
        actions = actions.at[:, : self.policy_prior_samples].set(policy_actions)

        # Iterate MPPI
        key, action_noise_key, *value_keys = jax.random.split(
            key, self.mppi_iterations + 1 + 1
        )
        noise = jax.random.normal(
            action_noise_key,
            shape=(
                self.mppi_iterations,
                self.horizon,
                self.population_size - self.policy_prior_samples,
                self.model.action_dim,
            ),
        )

        for i in range(self.mppi_iterations):
            # Sample actions
            actions = actions.at[:, self.policy_prior_samples :].set(
                mean[:, None, :] + std[:, None, :] * noise[i]
            )
            actions = jnp.clip(actions, -1, 1)

            # Compute elite actions
            value = self.estimate_value(z, actions, key=value_keys[i])
            value = jnp.nan_to_num(value)  # Handle nans
            _, elite_inds = jax.lax.top_k(value, self.num_elites)
            elite_values, elite_actions = value[elite_inds], actions[:, elite_inds]

            # Update parameters
            max_value = jnp.max(elite_values)
            score = jnp.exp(self.temperature * (elite_values - max_value))
            score /= jnp.sum(score) + 1e-8

            mean = jnp.sum(score[None, :, None] * elite_actions, axis=1)
            std = jnp.sqrt(
                jnp.sum(
                    score[None, :, None] * (elite_actions - mean[:, None, :]) ** 2,
                    axis=1,
                )
            )
            std = jnp.clip(std, self.min_plan_std, self.max_plan_std)

        # Select action based on the score
        key, *final_action_keys = jax.random.split(key, 3)
        action_ind = jax.random.choice(
            final_action_keys[0], a=jnp.arange(self.num_elites), p=score
        )
        actions = elite_actions[:, action_ind]

        action, action_std = actions[0], std[0]
        action += (
            jnp.array(train, float)
            * action_std
            * jax.random.normal(final_action_keys[1], shape=action.shape)
        )

        action = jnp.clip(action, -1, 1)
        return sg(action), (mean, std)

    @jax.jit
    def update(
        self,
        observations: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        next_observations: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        *,
        key: PRNGKeyArray,
    ) -> Tuple[TDMPC2, Dict[str, Any]]:
        target_dropout, value_dropout_key1, value_dropout_key2, policy_key = (
            jax.random.split(key, 4)
        )

        def world_model_loss_fn(
            encoder_params: Dict,
            dynamics_params: Dict,
            value_params: Dict,
            reward_params: Dict,
            continue_params: Dict,
        ):
            done = jnp.logical_or(terminated, truncated)
            discount = jnp.ones((self.horizon + 1, self.batch_size))
            horizon = jnp.zeros(self.batch_size)

            next_z = sg(self.model.encode(next_observations, encoder_params))
            td_targets = self.td_target(next_z, rewards, terminated, key=target_dropout)

            # Latent rollout (compute latent dynamics + consistency loss)
            zs = jnp.empty((self.horizon + 1, self.batch_size, next_z.shape[-1]))
            z = self.model.encode(
                jax.tree.map(lambda x: x[0], observations), encoder_params
            )
            zs = zs.at[0].set(z)
            consistency_loss = jnp.zeros(self.batch_size)

            for t in range(self.horizon):
                z = self.model.next(z, actions[t], dynamics_params)
                consistency_loss += jnp.mean(
                    (z - next_z[t]) ** 2 * discount[t][:, None], -1
                )
                zs = zs.at[t + 1].set(z)

                horizon += discount[t] > 0
                discount = discount.at[t + 1].set(
                    discount[t] * self.rho * (1 - done[t])
                )

            # Get logits for loss computations
            _, q_logits = self.model.Q(
                zs[:-1], actions, value_params, value_dropout_key1
            )
            _, reward_logits = self.model.reward(zs[:-1], actions, reward_params)

            if self.model.predict_continues:
                continue_logits = self.model.continue_model.apply_fn(
                    {"params": continue_params}, zs[1:]
                ).squeeze(-1)

            reward_loss = jnp.zeros(self.batch_size)
            value_loss = jnp.zeros(self.batch_size)
            continue_loss = jnp.zeros(self.batch_size)

            for t in range(self.horizon):
                reward_loss += (
                    soft_crossentropy(
                        reward_logits[t],
                        rewards[t],
                        self.model.symlog_min,
                        self.model.symlog_max,
                        self.model.num_bins,
                    )
                    * discount[t]
                )

                if self.model.predict_continues:
                    continue_loss += (
                        optax.sigmoid_binary_cross_entropy(
                            continue_logits[t], 1 - terminated[t]
                        )
                        * discount[t]
                    )

                for q in range(self.model.num_value_nets):
                    value_loss += (
                        soft_crossentropy(
                            q_logits[q, t],
                            td_targets[t],
                            self.model.symlog_min,
                            self.model.symlog_max,
                            self.model.num_bins,
                        )
                        * discount[t]
                        / self.model.num_value_nets
                    )

            consistency_loss = (consistency_loss / horizon).mean()
            reward_loss = (reward_loss / horizon).mean()
            value_loss = (value_loss / horizon).mean()
            continue_loss = (continue_loss / horizon).mean()
            total_loss = (
                self.consistency_coef * consistency_loss
                + self.reward_coef * reward_loss
                + self.value_coef * value_loss
                + self.continue_coef * continue_loss
            )

            return total_loss, {
                "consistency_loss": consistency_loss,
                "reward_loss": reward_loss,
                "value_loss": value_loss,
                "continue_loss": continue_loss,
                "total_loss": total_loss,
                "zs": zs,
            }

        # Update world model
        (
            (encoder_grads, dynamics_grads, value_grads, reward_grads, continue_grads),
            model_info,
        ) = jax.grad(world_model_loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)(
            self.model.encoder.params,
            self.model.dynamics_model.params,
            self.model.value_model.params,
            self.model.reward_model.params,
            self.model.continue_model.params if self.model.predict_continues else None,
        )
        zs = model_info.pop("zs")

        new_encoder = self.model.encoder.apply_gradients(grads=encoder_grads)
        new_dynamics_model = self.model.dynamics_model.apply_gradients(
            grads=dynamics_grads
        )
        new_reward_model = self.model.reward_model.apply_gradients(grads=reward_grads)
        new_value_model = self.model.value_model.apply_gradients(grads=value_grads)

        new_target_value_model = self.model.target_value_model.replace(
            params=optax.incremental_update(
                new_value_model.params, self.model.target_value_model.params, self.tau
            )
        )

        if self.model.predict_continues:
            new_continue_model = self.model.continue_model.apply_gradients(
                grads=continue_grads
            )
        else:
            new_continue_model = self.model.continue_model

        # Update policy
        def policy_loss_fn(params: Dict):
            actions, _, _, log_probs = self.model.sample_actions(
                zs, params, key=policy_key
            )

            # Compute Q-values
            Qs, _ = self.model.Q(
                zs, actions, new_value_model.params, value_dropout_key2
            )

            Q = jnp.mean(Qs, axis=0)
            # Update and apply scale
            scale = percentile_normalization(Q[0], self.scale)
            Q /= jnp.clip(scale, 1, None)

            # Compute policy objective (equation 4)
            rho = self.rho ** jnp.arange(self.horizon + 1)
            policy_loss = (
                (self.entropy_coef * log_probs - Q).mean(axis=1) * rho
            ).mean()
            return policy_loss, {"policy_loss": policy_loss, "policy_scale": scale}

        policy_grads, policy_info = jax.grad(policy_loss_fn, has_aux=True)(
            self.model.policy_model.params
        )
        new_policy = self.model.policy_model.apply_gradients(grads=policy_grads)

        # Update model
        new_agent = self.replace(
            model=self.model.replace(
                encoder=new_encoder,
                dynamics_model=new_dynamics_model,
                reward_model=new_reward_model,
                value_model=new_value_model,
                policy_model=new_policy,
                target_value_model=new_target_value_model,
                continue_model=new_continue_model,
            ),
            scale=policy_info["policy_scale"],
        )
        info = {**model_info, **policy_info}

        return new_agent, info

    @jax.jit
    def estimate_value(
        self, z: jax.Array, actions: jax.Array, key: PRNGKeyArray
    ) -> jax.Array:
        G, discount = 0, 1
        for t in range(self.horizon):
            reward, _ = self.model.reward(z, actions[t], self.model.reward_model.params)
            z = self.model.next(z, actions[t], self.model.dynamics_model.params)
            G += discount * reward

            if self.model.predict_continues:
                continues = jax.nn.sigmoid(
                    self.model.continue_model.apply_fn(
                        {"params": self.model.continue_model.params}, z
                    )
                ).squeeze(-1)
            else:
                continues = 1

            discount *= self.discount * continues

        action_key, dropout_key = jax.random.split(key, 2)
        next_action = self.model.sample_actions(
            z, self.model.policy_model.params, key=action_key
        )[0]

        # Sample two Q-values from the ensemble
        Qs, _ = self.model.Q(
            z, next_action, self.model.value_model.params, key=dropout_key
        )
        Q = jnp.mean(Qs, axis=0)
        return sg(G + discount * Q)

    @jax.jit
    def td_target(
        self,
        next_z: jax.Array,
        reward: jax.Array,
        terminal: jax.Array,
        key: PRNGKeyArray,
    ) -> jax.Array:
        action_key, ensemble_key, dropout_key = jax.random.split(key, 3)
        next_action = self.model.sample_actions(
            next_z, self.model.policy_model.params, key=action_key
        )[0]

        # Sample two Q-values from the target ensemble
        all_inds = jnp.arange(0, self.model.num_value_nets)
        inds = jax.random.choice(ensemble_key, a=all_inds, shape=(2,), replace=False)
        Qs, _ = self.model.Q(
            next_z, next_action, self.model.target_value_model.params, key=dropout_key
        )
        Q = jnp.min(Qs[inds], axis=0)
        return sg(reward + (1 - terminal) * self.discount * Q)


class CustomMonitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    """

    def __init__(
        self,
        env: gym.Env,
        log_dir: str = None,
        record_freq: int = 5,
        no_improvement_window: int = 100,
    ):
        super().__init__(env=env)

        self.t_start = time.time()
        self.results_writer = None
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_file_path = os.path.join(log_dir, "agent-log.txt")
        self.log_file = open(self.log_file_path, "w")

        self.rewards: List[float] = []
        self.episode_rewards: List[float] = []
        self.needs_reset = True
        self.episodes = 0
        self.cumulative_reward = 0.0
        self.record_freq = record_freq
        self.recording = False
        self.video_frames = []
        self.ep_since_improvement = 0
        self.best_reward = -np.inf
        self.no_improvement_window = no_improvement_window

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        episode_reward = sum(self.rewards)
        episode_length = len(self.rewards)

        if episode_length > 0:
            self.episodes += 1
            self.cumulative_reward += episode_reward
            self.episode_rewards.append(episode_reward)
            self.log_file.write(f"episode: {self.episodes}, reward: {episode_reward}\n")
            self.log_file.flush()

            log_data = {
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "cumulative_reward": self.cumulative_reward,
                "average_reward (last 10 episodes)": np.mean(
                    self.episode_rewards[-10:]
                ),
                "average_reward (last 100 episodes)": np.mean(
                    self.episode_rewards[-100:]
                ),
            }

            if self.recording and len(self.video_frames) > 0:
                video_array = np.stack(self.video_frames)
                # (time, height, width, channel) -> (time, channel, height, width)
                video_array = np.transpose(video_array, (0, 3, 1, 2))

                video = wandb.Video(
                    video_array,
                    caption=f"Episode {self.episodes}, Reward: {episode_reward:.0f}",
                    fps=30,
                )

                log_data["video"] = video
                self.video_frames = []

            wandb.log(log_data, step=self.episodes)

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.ep_since_improvement = 0
            else:
                self.ep_since_improvement += 1

                assert (
                    self.no_improvement_window <= 0
                    or self.episodes < 2 * self.no_improvement_window
                    or sum(self.episode_rewards[-self.no_improvement_window :])
                    > sum(
                        self.episode_rewards[
                            -2
                            * self.no_improvement_window : -self.no_improvement_window
                        ]
                    )
                ), f"Mean reward did not improve for {self.no_improvement_window} episodes"

        self.rewards = []
        self.needs_reset = False
        self.recording = self.record_freq > 0 and self.episodes % self.record_freq == 0

        return self.env.reset(**kwargs)

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))

        if self.recording:
            self.video_frames.append(self.env.render())

        if terminated or truncated:
            self.needs_reset = True

        return observation, reward, terminated, truncated, info

    def __del__(self):
        self.log_file.close()
        # upload the log file to wandb
        wandb.save(self.log_file_path)


def train(cfg: dict):
    seed = 42
    max_episodes = cfg["max_episodes"]
    encoder_config = cfg["encoder"]
    model_config = cfg["world_model"]
    tdmpc_config = cfg["tdmpc2"]
    seed_steps = int(cfg["seed_steps"])
    seed_update_ratio = cfg["seed_update_ratio"]
    buffer_size = cfg["buffer_size"]
    no_improvement_window = cfg["no_improvement_window"]
    
    wandb.init(
        project="tdmpc2-jax",
        entity=WANDB_ENTITY,
        config=cfg
    )

    env = gym.make(
        "BipedalWalker-v3", render_mode="rgb_array", hardcore=cfg["hardcore"]
    )

    env = CustomMonitor(
        env, log_dir="logs", record_freq=10, no_improvement_window=no_improvement_window
    )

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    dtype = jnp.dtype(model_config["dtype"])
    rng, model_key = jax.random.split(rng, 2)
    encoder = nn.Sequential(
        [
            NormedLinear(encoder_config["encoder_dim"], activation=mish, dtype=dtype)
            for _ in range(encoder_config["num_encoder_layers"] - 1)
        ]
        + [
            NormedLinear(
                model_config["latent_dim"],
                activation=partial(simnorm, simplex_dim=model_config["simnorm_dim"]),
                dtype=dtype,
            )
        ]
    )

    model = WorldModel.create(
        observation_space=env.observation_space,
        action_space=env.action_space,
        encoder_module=encoder,
        **model_config,
        key=model_key,
    )
    agent = TDMPC2.create(world_model=model, **tdmpc_config)

    replay_buffer = EpisodicReplayBuffer(
        capacity=buffer_size,
        dummy_input=dict(
            observation=env.observation_space.sample(),
            action=env.action_space.sample(),
            reward=1.0,
            next_observation=env.observation_space.sample(),
            terminated=True,
            truncated=True,
        ),
        seed=seed,
        respect_episode_boundaries=False,
    )

    # Training loop
    prev_plan = None
    observation, _ = env.reset(seed=seed)
    done = False
    step_count = 0

    for ep_count in tqdm.tqdm(range(max_episodes), smoothing=0.1):
        while not done:
            if step_count <= seed_steps:
                action = env.action_space.sample()
            else:
                rng, action_key = jax.random.split(rng)

                action, prev_plan = agent.act(
                    observation, prev_plan, train=True, key=action_key
                )

            next_observation, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            step_count += 1

            replay_buffer.insert(
                dict(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    truncated=truncated,
                ),
                episode_index=ep_count,
            )

            observation = next_observation

            if step_count >= seed_steps:
                if step_count == seed_steps:
                    print("Pre-training on seed data...")
                    num_updates = int(seed_update_ratio * seed_steps)
                else:
                    num_updates = 1

                rng, *update_keys = jax.random.split(rng, num_updates + 1)
                for j in range(num_updates):
                    batch = replay_buffer.sample(agent.batch_size, agent.horizon)

                    agent, train_info = agent.update(
                        observations=batch["observation"],
                        actions=batch["action"],
                        rewards=batch["reward"],
                        next_observations=batch["next_observation"],
                        terminated=batch["terminated"],
                        truncated=batch["truncated"],
                        key=update_keys[j],
                    )

        observation, _ = env.reset()
        prev_plan = None
        done = False


basic_config = {
    "max_episodes": 500,
    "seed_steps": 15600,
    "seed_update_ratio": 0.25,
    "buffer_size": 1000000,
    "hardcore": False,
    "no_improvement_window": 0,
    "encoder": {"encoder_dim": 48, "num_encoder_layers": 2},
    "world_model": {
        "mlp_dim": 96,
        "latent_dim": 208,
        "value_dropout": 0.005,
        "num_value_nets": 5,
        "num_bins": 101,
        "symlog_min": -10,
        "symlog_max": 10,
        "simnorm_dim": 8,
        "learning_rate": 6e-4,
        "encoder_learning_rate": 1e-4,
        "predict_continues": True,
        "dtype": "bfloat16",
    },
    "tdmpc2": {
        "mpc": True,
        "horizon": 5,
        "mppi_iterations": 6,
        "population_size": 1024,
        "policy_prior_samples": 32,
        "num_elites": 64,
        "min_plan_std": 0.05,
        "max_plan_std": 2,
        "temperature": 0.32,
        "batch_size": 256,
        "discount": 0.99,
        "rho": 0.5,
        "consistency_coef": 20,
        "reward_coef": 0.1,
        "continue_coef": 0.1,
        "value_coef": 0.1,
        "entropy_coef": 1e-5,
        "tau": 0.01,
    },
}

hardcore_config = {}

parser = argparse.ArgumentParser()
parser.add_argument("--hardcore", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    hardcore = args.hardcore
    config = hardcore_config if hardcore else basic_config
    train(config)
