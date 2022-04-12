# -*- coding: utf-8 -*-


import numpy as np
import jax
import jax.numpy as jnp
import jax.random as rnd
import jax.nn as jnn
import augmax
import tqdm

# Gaussian kernel
def sdca_gaussian_norm(gamma=2.0):
  def fun(Xi, X):
    Xi = Xi / jnp.linalg.norm(Xi, axis=1, keepdims=True)
    X = X / jnp.linalg.norm(X, axis=1, keepdims=True)
    return jnp.exp(-gamma * (2 - 2*Xi@X.T))
  return jax.jit(fun)

def sdca_poly_norm(deg=2, c=1.):
  def fun(Xi, X):
    Xi = Xi / jnp.linalg.norm(Xi, axis=1, keepdims=True)
    X = X / jnp.linalg.norm(X, axis=1, keepdims=True)
    return jnp.power((c + Xi@X.T), deg)
  return jax.jit(fun)

def SDCA(X, y, key, kernel=sdca_poly_norm(), C=100, E=10, batch_size=1024):
  n, d = y.shape
  alpha = jnp.zeros_like(y)

  for e in range(E):
    key, skey = rnd.split(key)
    indices = rnd.permutation(skey, n)
    for b in tqdm.trange(n//batch_size, desc="epoch: {}/{}".format(e, E)):
      i = indices[b*batch_size:(b+1)*batch_size]
      Xi = X[i,:]
      Ki = kernel(Xi, X)
      alpha = sdca_update_alpha(alpha, y, i, Ki, C)
    # last incomplete batch
    i = indices[(b+1)*batch_size:]
    Xi = X[i,:]
    Ki = kernel(Xi, X)
    alpha = sdca_update_alpha(alpha, y, i, Ki, C)
    Ki = None # free mem
  
  # pruning
  i = alpha.sum(axis=1) > 1e-7
  return alpha[i,:], X[i, :], y[i,:], kernel

@jax.jit
def sdca_update_alpha(alpha, y, i, Ki, C):
  err = 1 - y[i] * jnp.matmul(Ki, alpha*y)
  ai = jnp.clip(alpha[i] + err, 0, C)
  return alpha.at[i].set(ai)

def sdca_predict(X, alpha_i, X_i, y_i, kernel, batch_size=256):
  n, d = X.shape
  _, c = y_i.shape
  ypred = jnp.empty((0, c))
  for b in range(n//batch_size):
    Xb = X[b*batch_size:(b+1)*batch_size, :]
    K = kernel(Xb, X_i)
    yb = K @ (alpha_i * y_i)
    ypred = jnp.concatenate([ypred, yb], axis=0)
  # last batch
  Xb = X[(b+1)*batch_size:, :]
  K = kernel(Xb, X_i)
  yb = K @ (alpha_i * y_i)
  ypred = jnp.concatenate([ypred, yb], axis=0)
  return ypred

import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')

import tensorflow_datasets as tfds

data_dir = './tfds'

# Fetch full datasets for evaluation
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
# You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
cifar_data, info = tfds.load(name="cifar10", batch_size=-1, data_dir=data_dir, with_info=True)
cifar_data = tfds.as_numpy(cifar_data)
train_data, test_data = cifar_data['train'], cifar_data['test']
num_labels = info.features['label'].num_classes
h, w, c = info.features['image'].shape
num_pixels = h * w * c

# Full train set
train_images, train_labels = train_data['image'], train_data['label']
# convert to 0-1
X_train = train_images/255.
mean_train = X_train.mean(axis=(0,1,2), keepdims=True)
std_train = X_train.std(axis=(0,1,2), keepdims=True)
X_train = (X_train - mean_train ) /std_train
y_train = train_labels
print('nb train images: {}'.format(len(train_images)))
# validation set
val_images, val_labels = test_data['image'], test_data['label']
X_val = val_images/255.
X_val = (X_val - mean_train)/std_train
y_val = val_labels

print(mean_train)
print(std_train)

"""# PCA with whitening

The following function computes the projectors for a PCA with whitening (i.e., such that all dimensions in the projected space have unit variance). It uses a batch stratgegy to compute the covariance matrix to avoid out of memory error and should take no more than about 10 seconds.
"""

def pcaw(X, dim, key, transform, E=5):
  n = len(X)
  mu = 0.
  cov = 0.
  batch_size=8192
  for e in range(E):
    key, skey = rnd.split(key)
    sub_rng = jax.random.split(skey, train_images.shape[0])
    x = jnp.reshape(transform(sub_rng, X), (n, -1))
    mu = mu + x.mean(axis=0, keepdims=True)
    for b in tqdm.trange(n//batch_size, desc="epoch {}/{}".format(e, E)):
      xb = x[b*batch_size:(b+1)*batch_size, :]
      cov = cov + xb.T @ xb
    x = x[(b+1)*batch_size:, :]
    cov = cov + x.T @ x
  mu = mu / E
  cov = cov / (E*n - 1) - mu.T @ mu

  L, U = jnp.linalg.eigh(cov)
  P = U[:, -dim:] / jnp.sqrt(L[None, -dim:])
  return mu, P

def pcaw_project(X, mu, P):
  return (X - mu) @ P

def accuracy(y_pred, y):
  return (jnp.argmax(y_pred, axis=1) == jnp.argmax(y, axis=1)).mean()

"""# Single test

First we launch a single test with the following main hyperparameters to get an idea of the time the algorithm takes and its accuracy in a somewhat default setting.
"""

dim = 128
E = 5
C = 100
deg = 4
c = 0.1
nb_aug = 100
nb_val_aug = 20
batch_size = 128
key = rnd.PRNGKey(3407) # magic seed value stolen from pytorch!

# DA by imax

transform = augmax.Chain(
  augmax.RandomGrayscale(p=0.3),
  augmax.Solarization(p=0.3),
  augmax.GaussianBlur(p=0.3),
  augmax.HorizontalFlip(),
  augmax.RandomCrop(28, 28)
)
augment = jax.jit(jax.vmap(transform))

vtransform = augmax.Chain(
  augmax.CenterCrop(28, 28)
)
vaugment = jax.jit(jax.vmap(vtransform))

"""PCA"""
print("doing PCA")

key, skey = rnd.split(key)
mu, P = pcaw(X_train, dim, key, augment, E=nb_aug)

images = jnp.empty((0, dim))
labels = jnp.empty((0, num_labels))
n = len(X_train)
for e in tqdm.trange(nb_aug):
  key, skey = rnd.split(key)
  sub_rng = jax.random.split(skey, X_train.shape[0])
  transformed_images = jnp.reshape(augment(sub_rng, X_train), (n, -1))
  transformed_images = pcaw_project(transformed_images, mu, P)
  images = jnp.concatenate([images, transformed_images], axis=0)
  y = jnn.one_hot(y_train, 10) * 2 - 1
  labels = jnp.concatenate([labels, y], axis=0)
X_pca = images
Y = labels

"""Training one vs all"""
print("Training...")

key, skey = rnd.split(key)
alpha, Xi, yi, kernel = SDCA(X_pca, Y, key, kernel=sdca_poly_norm(deg=deg, c=c), C=C, E=E, batch_size=batch_size)

"""Validation accuracy"""
print("Evaluating...")

images = jnp.empty((0, dim))
n = len(X_val)
for e in range(nb_val_aug):
  key, skey = rnd.split(key)
  sub_rng = jax.random.split(skey, X_val.shape[0])
  transformed_images = jnp.reshape(augment(sub_rng, X_val), (n, -1))
  transformed_images = pcaw_project(transformed_images, mu, P)
  images = jnp.concatenate([images, transformed_images], axis=0)
Xv = images
Yv = jnn.one_hot(y_val, num_labels)*2 - 1

y_pred = sdca_predict(Xv, alpha, Xi, yi, kernel)

y_pred = jnp.reshape(y_pred, (nb_val_aug, n, num_labels)).mean(axis=0)
acc = accuracy(y_pred, Yv)
print('accuracy: {}'.format(acc))

print(alpha[0:10, :])
