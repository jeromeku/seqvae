from datetime import datetime
import functools
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def image_summary(*, seqs, name, step, num=None):
    """Visualizes sequences as TensorBoard summaries.

  Args:
    seqs: A tensor of shape [n, t, h, w, c].
    name: String name of this summary.
    num: Integer for the number of examples to visualize. Defaults to
      all examples.
  """
    seqs = tf.clip_by_value(seqs, 0.0, 1.0)
    seqs = tf.unstack(seqs[:num])
    joined_seqs = [tf.concat(tf.unstack(seq), 1) for seq in seqs]
    joined_seqs = tf.expand_dims(tf.concat(joined_seqs, 0), 0)
    tf.compat.v2.summary.image(name, joined_seqs, max_outputs=1, step=step)


def visualize_reconstruction(inputs, reconstruct, step, num=3, name="reconstruction"):
    """Visualizes the reconstruction of inputs in TensorBoard.

  Args:
    inputs: A tensor of the original inputs, of shape [batch, timesteps,
      h, w, c].
    reconstruct: A tensor of a reconstruction of inputs, of shape
      [batch, timesteps, h, w, c].
    num: Integer for the number of examples to visualize.
    name: String name of this summary.
  """
    reconstruct = tf.clip_by_value(reconstruct, 0.0, 1.0)
    inputs_and_reconstruct = tf.concat((inputs[:num], reconstruct[:num]), axis=0)
    image_summary(seqs=inputs_and_reconstruct, name=name, step=step)


def visualize_qualitative_analysis(
    inputs, model, step, samples=1, batch_size=3, length=8
):
    """Visualizes a qualitative analysis of a given model.

  Args:
    inputs: A tensor of the original inputs, of shape [batch, timesteps,
      h, w, c].
    model: A DisentangledSequentialVAE model.
    samples: Number of samples to draw from the latent distributions.
    batch_size: Number of sequences to generate.
    length: Number of timesteps to generate for each sequence.
  """
    average = lambda dist: tf.reduce_mean(
        input_tensor=dist.mean(), axis=0
    )  # avg over samples
    with tf.compat.v1.name_scope("val_reconstruction"):
        reconstruct = functools.partial(
            model.reconstruct, inputs=inputs, samples=samples
        )
        visualize_reconstruction(inputs, average(reconstruct()), step)
        visualize_reconstruction(
            inputs, average(reconstruct(sample_static=True)), step, name="static_prior"
        )
        visualize_reconstruction(
            inputs,
            average(reconstruct(sample_dynamic=True)),
            step,
            name="dynamic_prior",
        )
        visualize_reconstruction(
            inputs, average(reconstruct(swap_static=True)), step, name="swap_static"
        )
        visualize_reconstruction(
            inputs, average(reconstruct(swap_dynamic=True)), step, name="swap_dynamic"
        )

    with tf.compat.v1.name_scope("generation"):
        generate = functools.partial(
            model.generate, batch_size=batch_size, length=length, samples=samples
        )
        image_summary(
            seqs=average(generate(fix_static=True)), name="fix_static", step=step
        )
        image_summary(
            seqs=average(generate(fix_dynamic=True)), name="fix_dynamic", step=step
        )


def summarize_dist_params(dist, name, step, name_scope="dist_params"):
    """Summarize the parameters of a distribution.

  Args:
    dist: A Distribution object with mean and standard deviation
      parameters.
    name: The name of the distribution.
    name_scope: The name scope of this summary.
  """
    with tf.compat.v1.name_scope(name_scope):
        tf.compat.v2.summary.histogram(
            name="{}/{}".format(name, "mean"), data=dist.mean(), step=step
        )
        tf.compat.v2.summary.histogram(
            name="{}/{}".format(name, "stddev"), data=dist.stddev(), step=step
        )


def summarize_mean_in_nats_and_bits(
    inputs, units, name, step, nats_name_scope="nats", bits_name_scope="bits_per_dim"
):
    """Summarize the mean of a tensor in nats and bits per unit.

  Args:
    inputs: A tensor of values measured in nats.
    units: The units of the tensor with which to compute the mean bits
      per unit.
    name: The name of the tensor.
    nats_name_scope: The name scope of the nats summary.
    bits_name_scope: The name scope of the bits summary.
  """
    mean = tf.reduce_mean(input_tensor=inputs)
    with tf.compat.v1.name_scope(nats_name_scope):
        tf.compat.v2.summary.scalar(name, mean, step=step)
    with tf.compat.v1.name_scope(bits_name_scope):
        tf.compat.v2.summary.scalar(name, mean / units / tf.math.log(2.0), step=step)
