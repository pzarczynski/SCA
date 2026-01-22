import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from flax.training.train_state import TrainState
from sca import helpers as h

import logging
logging.basicConfig(level=logging.INFO)


class MLP(nn.Module):
    dims: tuple = (64, 64)
    num_classes: int = 256

    @nn.compact
    def __call__(self, x, train: bool):
        for dim in self.dims:
            x = nn.Dense(dim)(x)
            # x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.relu(x)

        x = nn.Dense(self.num_classes)(x)
        return x


def PI(proba, pts, ks, eps=1e-15):
    pts, ks = pts[:, 2], ks[:, 2]
    log_proba = jnp.log2(jnp.maximum(proba, eps))
    return jnp.array([h.pi_score(log_proba[ks == k], pts[ks == k], k) 
                     for k in range(256)])


def batchify(*X, batch_size):
    n_batches = X[0].shape[0] // batch_size
    size = n_batches * batch_size
    return [jnp.reshape(Y[:size], (n_batches, batch_size, -1)) for Y in X]


def shuffle(*X, key):
    shuffle_key, key = jax.random.split(key)
    idx = jax.random.permutation(shuffle_key, X[0].shape[0])
    return [Y[idx] for Y in X], key

     
def create_train_state(key): 
    init_key, dropout_key, key = jax.random.split(key, 3)
    model = MLP()

    dummy = jnp.ones((1, 1400), dtype=jnp.float32)
    var = model.init(
        {'params': init_key, 'dropout': dropout_key}, dummy, key
    )
    state = TrainState.create(
        apply_fn=model.apply, 
        params=var['params'],
        tx=optax.rmsprop(1e-5),
    )
    return state, key


@jax.jit
def train_step(statekey, batch):
    state, key = statekey
    key, drop_key = jax.random.split(key)
    X, y = batch

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, X, train=True, rngs={'dropout': drop_key}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return (state, key), loss


@jax.jit
def val_step(state, batch):
    X, y, pts, ks = batch
    logits = state.apply_fn({'params': state.params}, X, train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y[..., 0])
    pi = jnp.mean(PI(X, pts, ks))
    return state, (loss, pi)


if __name__ == '__main__':
    from sklearn.model_selection import StratifiedKFold as SKFold

    key = jax.random.PRNGKey(42)
    state, key = create_train_state(key)

    X, y, pts, ks = h.load_data('data/processed/ascadv_clean.h5')
    prof_idx, atk_idx = next(SKFold(n_splits=2).split(X, ks[:, 2]))

    train_dataset = X[prof_idx], y[prof_idx]
    val_dataset = X[atk_idx], y[atk_idx], pts[atk_idx], ks[atk_idx]
    val_dataset = batchify(*val_dataset, batch_size=64)

    for epoch in range(20):
        train_dataset, key = shuffle(*train_dataset, key=key)

        (state, key), train_losses = jax.lax.scan(train_step, (state, key), train_dataset)
        _, (val_losses, val_pi) = jax.lax.scan(val_step, state, val_dataset) 
        
        train_loss = jnp.mean(train_losses)
        val_loss, pi = jnp.mean(val_loss), jnp.mean(val_pi)
        logging.info(f"{epoch=}: \t{train_loss=:.4f}; \t{val_loss=:.4f}; \t{pi=:.2e}")

