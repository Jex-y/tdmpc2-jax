from functools import partial
from typing import (
    Callable,
    Optional,
)

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import dtypes
from flax.linen.normalization import _canonicalize_axes
from jax import lax
from jax._src import core
from jax.nn import initializers

Array = jax._src.typing.Array
KeyArray = Array
Shape = core.Shape

class Affine(nn.Module):
    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', initializers.ones, (x.shape[-1],))
        shift = self.param('shift', initializers.zeros, (x.shape[-1],))
        
        return x * scale + shift


class BatchNorm(nn.Module):
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 0.001
    dtype: Optional[jnp.dtype] = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x):    
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)

        # Cast to float 32:
        x = jnp.asarray(x, dtype=jnp.float32)

        # Caculate means and variances for each feature
        mu = jnp.mean(x, axis=reduction_axes, keepdims=True)
        # sig_square = jnp.var(x, axis=reduction_axes, keepdims=True)

        # Use trainable parameters for scale and bias
        # scale = self.param('scale', self.scale_init, (1, x.shape[-1]), self.param_dtype)
        # bias = self.param('bias', self.bias_init, (1, x.shape[-1]), self.param_dtype)
        
        def nan_check(name):
            def check(x):
                if jnp.any(jnp.isnan(x)):
                    raise ValueError(f'NaNs detected in {name}')
            return check
          
        # jax.debug.callback(nan_check('mu'), mu)
        # jax.debug.callback(nan_check('sig_square'), sig_square)
        # jax.debug.callback(nan_check('scale'), scale)
        # jax.debug.callback(nan_check('bias'), bias)

        # Apply batch norm
        y = x - mu
        jax.debug.callback(nan_check('x - mu'), y)
        
        # y *= lax.rsqrt(sig_square + self.epsilon) # * scale
        # jax.debug.callback(nan_check('y *= lax.rsqrt(sig_square + self.epsilon) * scale'), y)
        
        # y += bias
        # jax.debug.callback(nan_check('y += bias'), y)

        dtype = dtypes.canonicalize_dtype(x, dtype=self.dtype)
        return jnp.asarray(y, dtype=dtype)

class NormedLinear(nn.Module):
    features: int
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    dropout_rate: Optional[float] = None
    use_layer_norm: Optional[bool] = True
    use_batch_norm: Optional[bool] = False
    batch_norm_momentum: Optional[float] = 0.99

    kernel_init: Callable = partial(nn.initializers.truncated_normal, stddev=0.02)

    dtype: jnp.dtype = jnp.float32
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
