# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions and classes related to optimization (weight updates)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from absl import logging
# import gin
import tensorflow as tf
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
import sys
# import tensorflow_addons.optimizers as tfa_optimizers


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a given learning rate decay schedule."""

  def __init__(self,
               initial_learning_rate,
               decay_schedule_fn,
               warmup_steps,
               power=1.0,
               name=None):
    super(WarmUp, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or 'WarmUp') as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = (
          self.initial_learning_rate *
          tf.math.pow(warmup_percent_done, self.power))
      return tf.cond(
          global_step_float < warmup_steps_float,
          lambda: warmup_learning_rate,
          lambda: self.decay_schedule_fn(step),
          name=name)

  def get_config(self):
    return {
        'initial_learning_rate': self.initial_learning_rate,
        'decay_schedule_fn': self.decay_schedule_fn,
        'warmup_steps': self.warmup_steps,
        'power': self.power,
        'name': self.name
    }


# # @gin.configurable
def create_optimizer(init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     lr_multipliers,
                     end_lr=0.0,
                     optimizer_type='sgd',
                     momentum=None):
  """Creates an optimizer with learning rate schedule."""
  # Implements linear decay of the learning rate.
  lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=init_lr,
      decay_steps=num_train_steps,
      end_learning_rate=end_lr)
  if num_warmup_steps:
    lr_schedule = WarmUp(
        initial_learning_rate=init_lr,
        decay_schedule_fn=lr_schedule,
        warmup_steps=num_warmup_steps)

  if optimizer_type == 'sgd':
    logging.info('using layer-wise sgd optimizer')
    optimizer = LayerwiseSGD(
        learning_rate=lr_schedule,
        lr_multipliers=lr_multipliers)
  elif optimizer_type == 'sgd_momentum':
    logging.info('using layer-wise sgd with momentum optimizer')
    optimizer = LayerwiseSGD(
        learning_rate=lr_schedule,
        momentum=momentum,
        nesterov=True,
        lr_multipliers=lr_multipliers)
  # elif optimizer_type == 'lamb':
  #   logging.info('using Lamb optimizer')
  #   optimizer = tfa_optimizers.LAMB(
  #       learning_rate=lr_schedule,
  #       weight_decay_rate=0.01,
  #       beta_1=0.9,
  #       beta_2=0.999,
  #       epsilon=1e-6,
  #       exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
  else:
    raise ValueError('Unsupported optimizer type: ', optimizer_type)

  return optimizer


class LayerwiseSGD(tf.keras.optimizers.SGD):
  def __init__(self,
               learning_rate=1.0,
               lr_multipliers=None,
               momentum=0.0,
               nesterov=False,
               clipvalue=1.0,
               name='LayerwiseSGD',
               **kwargs):
    super(LayerwiseSGD, self).__init__(learning_rate, momentum=momentum,
                                       nesterov=nesterov, name=name, **kwargs)
    self.lr_multipliers = lr_multipliers
  
  def _get_lr(self, name):
      return self.lr_multipliers[name]

  def _resource_apply_dense(self, grad, var, apply_state=None):
    lr_mult = self._get_lr(var.name)
    # print_op = tf.print(var.name, lr_mult)
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # with tf.control_dependencies([print_op]):
    if self._momentum:
      momentum_var = self.get_slot(var, "momentum")
      return training_ops.resource_apply_keras_momentum(
            var.handle,
            momentum_var.handle,
            coefficients["lr_t"] * lr_mult,
            grad,
            coefficients["momentum"],
            use_locking=self._use_locking,
            use_nesterov=self.nesterov)
    else:
      return training_ops.resource_apply_gradient_descent(
            var.handle, coefficients["lr_t"] * lr_mult, grad, use_locking=self._use_locking)
    

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    lr_mult = self._get_lr(var.name)
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    momentum_var = self.get_slot(var, "momentum")
    return training_ops.resource_sparse_apply_keras_momentum(
        var.handle,
        momentum_var.handle,
        coefficients["lr_t"] * lr_mult,
        grad,
        indices,
        coefficients["momentum"],
        use_locking=self._use_locking,
        use_nesterov=self.nesterov)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices,
                                               **kwargs):
    lr_mult = self._get_lr(var.name)
    if self._momentum:
      return super(LayerwiseSGD, self)._resource_apply_sparse_duplicate_indices(
          grad, var, indices, **kwargs)
    else:
      var_device, var_dtype = var.device, var.dtype.base_dtype
      coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
                      or self._fallback_apply_state(var_device, var_dtype))

      return resource_variable_ops.resource_scatter_add(
          var.handle, indices, -grad * coefficients["lr_t"] * lr_mult)