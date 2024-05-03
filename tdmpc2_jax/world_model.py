import copy
from functools import partial
from typing import Dict, Tuple, Callable, Optional
import flax.linen as nn
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from networks import NormedLinear, BatchNorm
from common.activations import mish, simnorm
from jaxtyping import PRNGKeyArray
import jax
import jax.numpy as jnp
import optax
from networks import Ensemble
import gymnasium as gym
from common.util import symlog, two_hot_inv


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x


class WorldModel(struct.PyTreeNode):
    # Models
    encoder: TrainState
    dynamics_model: TrainState
    reward_model: TrainState
    policy_model: TrainState
    value_model: TrainState
    target_value_model: Optional[TrainState]
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
    crossq: bool = struct.field(pytree_node=False)

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
        crossq: bool = False,
        max_grad_norm: float = 10,
        # Misc
        dtype: jnp.dtype = jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        dynamics_key, reward_key, value_key = jax.random.split(key, 3)

        NormedLinearPartial: Callable = partial(
            NormedLinear,
            activation=mish,
            dtype=dtype,
            use_layer_norm=not crossq,
            use_batch_norm=crossq,
        )

        maybe_batch_norm = partial(BatchNorm, dtype=dtype) if crossq else Identity

        action_dim = np.prod(action_space.shape)

        variables = encoder_module.init(key, observation_space.sample())

        encoder = TrainState.create(
            apply_fn=encoder_module.apply,
            params=variables["params"],
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(encoder_learning_rate),
            ),
        )

        # Latent forward dynamics model
        dynamics_module = nn.Sequential(
            [
                maybe_batch_norm(),
                NormedLinearPartial(
                    mlp_dim,
                ),
                NormedLinearPartial(
                    mlp_dim,
                ),
                NormedLinearPartial(
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
                maybe_batch_norm(),
                NormedLinearPartial(
                    mlp_dim,
                ),
                NormedLinearPartial(
                    mlp_dim,
                ),
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
                maybe_batch_norm(),
                NormedLinearPartial(
                    mlp_dim,
                ),
                NormedLinearPartial(
                    mlp_dim,
                ),
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

        layers = [
            maybe_batch_norm(),
            NormedLinearPartial(mlp_dim, dropout_rate=value_dropout),
            NormedLinearPartial(mlp_dim),
            nn.Dense(num_bins, kernel_init=nn.initializers.zeros),
        ]

        value_base = partial(nn.Sequential, layers)

        value_ensemble = Ensemble(value_base, num=num_value_nets)

        value_param_key, value_dropout_key = jax.random.split(value_key)
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

        if crossq:
            target_value_model = None
        else:
            target_value_model = TrainState.create(
                apply_fn=value_ensemble.apply,
                params=copy.deepcopy(value_model.params),
                tx=optax.GradientTransformation(lambda _: None, lambda _: None),
            )

        if predict_continues:
            continue_module = nn.Sequential(
                [
                    maybe_batch_norm(),
                    NormedLinearPartial(
                        mlp_dim,
                    ),
                    NormedLinearPartial(
                        mlp_dim,
                    ),
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
            crossq=crossq,
        )

    @jax.jit
    def encode(self, obs: np.ndarray, params: Dict) -> jax.Array:
        return self.encoder.apply_fn(
            {"params": params}, jax.tree.map(lambda x: symlog(x), obs)
        )

    @jax.jit
    def next(self, z: jax.Array, a: jax.Array, params: Dict) -> jax.Array:
        z = jnp.concatenate([z, a], axis=-1)
        return self.dynamics_model.apply_fn({"params": params}, z)

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

        logits = self.value_model.apply_fn(
            {"params": params},
            z,
            rngs={"dropout": key},
        )

        Q = two_hot_inv(logits, self.symlog_min, self.symlog_max, self.num_bins)
        return Q, logits
