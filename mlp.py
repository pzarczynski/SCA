import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from flax.training.train_state import TrainState
from jax.ops import segment_sum
from sca import helpers as h

import logging
logging.basicConfig(level=logging.INFO)


SBOX = jnp.array(h.SBOX)


@jax.jit(static_argnums=(3,))
def compute_pi(log_proba, pts, ks, num_classes=256):
    correct_idx = SBOX[pts[:, 2] ^ ks[:, 2]]                
    correct_proba = log_proba[jnp.arange(log_proba.shape[0]), correct_idx]
    correct_proba /= jnp.log(2)

    data = correct_proba                           
    seg_ids = ks[:, 2]
    num = segment_sum(data, seg_ids, num_segments=num_classes)
    den = segment_sum(jnp.ones_like(data), seg_ids,
                      num_segments=num_classes)  
    mean_correct = num / jnp.maximum(den, 1.0)     

    return 8.0 + mean_correct                       


def compute_pi_from_batches(log_proba, pts, ks, num_classes=256):
    log_proba = log_proba.reshape(-1, num_classes)
    pts = pts[:log_proba.shape[0]]
    ks = ks[:log_proba.shape[0]]
    score = compute_pi(log_proba, pts, ks, num_classes)
    return jnp.mean(score)


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
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y[..., 0])
        log_proba = jax.nn.log_softmax(logits, axis=-1)
        return jnp.mean(loss), log_proba

    (loss, log_proba), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return (state, key), (loss, log_proba)


@jax.jit
def val_step(state, batch):
    X, y = batch
    logits = state.apply_fn({'params': state. params}, X, train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y[..., 0])
    log_proba = jax.nn.log_softmax(logits, axis=-1)
    return state, (jnp.mean(loss), log_proba)


if __name__ == '__main__':
    from sklearn.model_selection import StratifiedKFold as SKFold
    from sklearn.preprocessing import StandardScaler

    key = jax.random.PRNGKey(42)
    state, key = create_train_state(key)

    X, y, pts, ks = h.load_data('data/processed/ascadv_clean.h5')
    prof_idx, atk_idx = next(SKFold(n_splits=2).split(X, ks[:, 2]))

    scaler = StandardScaler()
    X_prof, y_prof = scaler.fit_transform(X[prof_idx]), y[prof_idx]
    pts_prof, ks_prof = pts[prof_idx], ks[prof_idx]

    val_dataset = scaler.transform(X[atk_idx]), y[atk_idx]
    val_dataset = batchify(*val_dataset, batch_size=64)
    pts_atk, ks_atk = pts[atk_idx], ks[atk_idx]

    for epoch in range(20):
        train_dataset, key = shuffle(X_prof, y_prof, key=key)
        train_dataset = batchify(*train_dataset, batch_size=64)

        (state, key), (train_losses, train_log_proba) = jax.lax.scan(train_step, (state, key), train_dataset)
        train_loss = jnp.mean(train_losses)
        train_pi = compute_pi_from_batches(train_log_proba, pts_prof, ks_prof)

        _, (val_losses, val_log_proba) = jax.lax.scan(val_step, state, val_dataset) 
        val_loss = jnp.mean(val_losses)
        val_pi = compute_pi_from_batches(val_log_proba, pts_atk, ks_atk)

        logging.info(f"{epoch=}: \t{train_loss=:.4f}; \t{train_pi=:.6f}; \t{val_loss=:.4f}; \t{val_pi=:.6f}")

