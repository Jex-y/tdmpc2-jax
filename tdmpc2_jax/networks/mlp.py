from functools import partial
from typing import (
    Any,
    Callable,
    Optional,
)

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from jax._src import core
from jax.nn import initializers

Array = jax._src.typing.Array
KeyArray = Array
Shape = core.Shape


class BatchNorm(nn.Module):
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 0.001
    warmup_steps: int = 100_000
    dtype: Optional[jnp.dtype] = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[KeyArray, Shape, jnp.dtype], Array] = initializers.zeros
    scale_init: Callable[[KeyArray, Shape, jnp.dtype], Array] = initializers.ones
    axis_index_groups: Any = None
    # This parameter was added in flax.linen 0.7.2 (08/2023)
    # commented out to be compatible with a wider range of jax versions
    # TODO: re-activate in some months (04/2024)
    use_fast_variance: bool = False

    @nn.compact
    def __call__(self, x):
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)

        mean, var = _compute_stats(
            x,
            reduction_axes,
            dtype=self.dtype,
            axis_name=self.axis_name if not self.is_initializing() else None,
            axis_index_groups=self.axis_index_groups,
            use_fast_variance=self.use_fast_variance,
        )

        def raise_if_nan_input(x):
            if jnp.any(jnp.isnan(x)):
                raise ValueError("BatchNorm has NaN inputs.")

        def raise_if_nan_output(x):
            if jnp.any(jnp.isnan(x)):
                raise ValueError("BatchNorm has NaN outputs.")

        jax.debug.callback(raise_if_nan_input, x)

        x = _normalize(
            self,
            x,
            mean,
            var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )

        jax.debug.callback(raise_if_nan_output, x)

        return x


class NormedLinear(nn.Module):
    features: int
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    dropout_rate: Optional[float] = None
    use_layer_norm: Optional[bool] = True
    use_batch_norm: Optional[bool] = False
    batch_norm_momentum: Optional[float] = 0.99

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

        if self.use_layer_norm:
            x = nn.LayerNorm()(x)

        x = self.activation(x)

        if self.use_batch_norm:
            x = BatchNorm(dtype=self.dtype)(x)

        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        return x
