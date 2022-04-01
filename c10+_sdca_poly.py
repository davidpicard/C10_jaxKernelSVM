# -*- coding: utf-8 -*-

import argparse
import jax
import jax.numpy as jnp
import jax.random as rnd
import jax.nn as jnn
from tqdm import trange

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
    for b in trange(n//batch_size, desc='Epoch {}/{}'.format(e, E)):
      i = indices[b*batch_size:(b+1)*batch_size]
      Xi = X[i,:]
      yi = y[i]
      Ki = kernel(Xi, X)
      alpha = sdca_update_alpha(alpha, y, i, Ki, C)
    # last incomplete batch
    i = indices[(b+1)*batch_size:]
    Xi = X[i,:]
    Ki = kernel(Xi, X)
    alpha = sdca_update_alpha(alpha, y, i, Ki, C)
  
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
  for b in trange(n//batch_size):
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

def pcaw(X, dim):
  n = len(X)
  mu = 0.
  cov = 0.
  batch_size=8192
  # normal case
  for i in range(4):
    for j in range(4):
      x = jnp.reshape(X[:, i:i+28, j:j+28, :], (n, -1)) # crop + flatten
      mu = mu + x.mean(axis=0, keepdims=True)
      for b in trange(n // batch_size, desc='{},{}'.format(i,j)):
        xb = x[b * batch_size:(b + 1) * batch_size, :]
        cov = cov + xb.T @ xb
      x = x[(b + 1) * batch_size:, :]
      cov = cov + x.T @ x
  # flipped case
  for i in range(4):
    for j in range(4):
      x = X[:, i:i + 28, j:j + 28, :] # crop
      x = jnp.reshape(x[:,:,::-1,:], (n, -1)) # flip + flatten
      mu = mu + x.mean(axis=0, keepdims=True)
      for b in trange(n // batch_size, desc='{},{}'.format(i,j)):
        xb = x[b * batch_size:(b + 1) * batch_size, :]
        cov = cov + xb.T @ xb
      x = x[(b + 1) * batch_size:, :]
      cov = cov + x.T @ x

  mu = mu / 16
  cov = cov / (16*n - 1) - mu.T @ mu

  L, U = jnp.linalg.eigh(cov)
  P = U[:, -dim:] / jnp.sqrt(L[None, -dim:])
  return mu, P

def pcaw_project(X, mu, P):
  return (X - mu) @ P

def accuracy(y_pred, y):
  return (jnp.argmax(y_pred, axis=1) == jnp.argmax(y, axis=1)).mean()

parser = argparse.ArgumentParser()
parser.add_argument("--configuration", type=int)
args = parser.parse_args()

configurations = []
for d in [64, 128, 256, 512, 1024]:
  for g in [1, 2, 3, 4, 5]:
    for c in [0., 0.1, 0.2, 0.5, 1.0]:
      configurations.append({ 'dim': d,
                              'deg': g,
                              'c': c})
conf = configurations[args.configuration]
print(conf)

dim = conf['dim']
E = 5
C = 100
deg = conf['deg']
c = conf['c']
batch_size = 512
key = rnd.PRNGKey(3407) # magic seed value stolen from pytorch!

"""PCA"""
print('PCA...')
key, skey = rnd.split(key)
mu, P = pcaw(X_train, dim)

images = jnp.empty((0, dim))
labels = jnp.empty((0, num_labels))
n = len(X_train)
# normal
for i in range(4):
  for j in range(4):
    x = jnp.reshape(X_train[:, i:i + 28, j:j + 28, :], (n, -1))  # crop + flatten
    im = pcaw_project(x, mu, P)
    images = jnp.concatenate([images, im], axis=0)
    y = jnn.one_hot(y_train, 10) * 2 - 1
    labels = jnp.concatenate([labels, y], axis=0)

# flipped
for i in range(4):
  for j in range(4):
    x = X_train[:, i:i + 28, j:j + 28, :] # crop
    x = jnp.reshape(x[:,:,::-1,:], (n, -1)) # flip + flatten
    im = pcaw_project(x, mu, P)
    images = jnp.concatenate([images, im], axis=0)
    y = jnn.one_hot(y_train, 10) * 2 - 1
    labels = jnp.concatenate([labels, y], axis=0)

X_pca = images
Y = labels

"""Training one vs all"""
print('Training...')
key, skey = rnd.split(key)
alpha, Xi, yi, kernel = SDCA(X_pca, Y, key, kernel=sdca_poly_norm(deg=deg, c=c), C=C, E=E, batch_size=batch_size)

"""Validation accuracy"""
print('Evaluating...')

images = jnp.empty((0, dim))
n = len(X_val)
# normal
for i in range(4):
  for j in range(4):
    x = jnp.reshape(X_val[:, i:i + 28, j:j + 28, :], (n, -1))  # crop + flatten
    im = pcaw_project(x, mu, P)
    images = jnp.concatenate([images, im], axis=0)

# flipped
for i in range(4):
  for j in range(4):
    x = X_val[:, i:i + 28, j:j + 28, :] # crop
    x = jnp.reshape(x[:,:,::-1,:], (n, -1)) # flip + flatten
    im = pcaw_project(x, mu, P)
    images = jnp.concatenate([images, im], axis=0)

Xv = images
Yv = jnn.one_hot(y_val, num_labels)*2 - 1

y_pred = sdca_predict(Xv, alpha, Xi, yi, kernel)

y_pred = jnp.reshape(y_pred, (16, n, num_labels)).mean(axis=0)
acc = accuracy(y_pred, Yv)
print('Accuracy: {}'.format(acc))
