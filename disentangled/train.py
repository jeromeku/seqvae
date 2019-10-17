from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm.notebook import trange

import tensorflow as tf
import tensorflow_probability as tfp
from '.summary' import *
tfd = tfp.distributions


def train_step(*, model, optimizer, train_ds, test_ds, FLAGS, writer, checkpoint_manager):
    step = tf.Variable(0, dtype=tf.int64, name="step")

    for inputs in train_ds.prefetch(buffer_size=None):
        with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(0, step % FLAGS.log_steps)
        ):
            tf.compat.v2.summary.histogram("image", data=inputs, step=step)

        with tf.GradientTape() as tape:
            features = model.compressor(inputs)  # (batch, timesteps, hidden)
            static_sample, static_posterior = model.sample_static_posterior(
                features, FLAGS.num_samples
            )  # (samples, batch, latent)
            dynamic_sample, dynamic_posterior = model.sample_dynamic_posterior(
                features, FLAGS.num_samples, static_sample
            )  # (sampl, N, T, latent)
            likelihood = model.decoder((dynamic_sample, static_sample))

            reconstruction = tf.reduce_mean(  # integrate samples
                input_tensor=likelihood.mean()[: FLAGS.num_reconstruction_samples],
                axis=0,
            )
            visualize_reconstruction(
                inputs, reconstruction, step, name="train_reconstruction"
            )

            static_prior = model.static_prior()
            _, dynamic_prior = model.sample_dynamic_prior(
                FLAGS.num_samples, FLAGS.batch_size, sprites.length
            )

            if FLAGS.enable_debug_logging:
                summarize_dist_params(static_prior, "static_prior", step)
                summarize_dist_params(static_posterior, "static_posterior", step)
                summarize_dist_params(dynamic_prior, "dynamic_prior", step)
                summarize_dist_params(dynamic_posterior, "dynamic_posterior", step)
                summarize_dist_params(likelihood, "likelihood", step)

            static_prior_log_prob = static_prior.log_prob(static_sample)
            static_posterior_log_prob = static_posterior.log_prob(static_sample)
            dynamic_prior_log_prob = tf.reduce_sum(
                input_tensor=dynamic_prior.log_prob(dynamic_sample), axis=-1
            )  # sum time
            dynamic_posterior_log_prob = tf.reduce_sum(
                input_tensor=dynamic_posterior.log_prob(dynamic_sample), axis=-1
            )  # sum time
            likelihood_log_prob = tf.reduce_sum(
                input_tensor=likelihood.log_prob(inputs), axis=-1
            )  # sum time

            if FLAGS.enable_debug_logging:
                with tf.compat.v1.name_scope("log_probs"):
                    summarize_mean_in_nats_and_bits(
                        static_prior_log_prob, FLAGS.latent_size_static, "static_prior", step
                    )
                    summarize_mean_in_nats_and_bits(
                        static_posterior_log_prob,
                        FLAGS.latent_size_static,
                        "static_posterior", step
                    )
                    summarize_mean_in_nats_and_bits(
                        dynamic_prior_log_prob,
                        FLAGS.latent_size_dynamic * sprites.length,
                        "dynamic_prior", step
                    )
                    summarize_mean_in_nats_and_bits(
                        dynamic_posterior_log_prob,
                        FLAGS.latent_size_dynamic * sprites.length,
                        "dynamic_posterior", step
                    )
                    summarize_mean_in_nats_and_bits(
                        likelihood_log_prob,
                        sprites.frame_size ** 2
                        * sprites.channels
                        * sprites.length,
                        "likelihood", step
                    )

            elbo = tf.reduce_mean(
                input_tensor=static_prior_log_prob
                - static_posterior_log_prob
                + dynamic_prior_log_prob
                - dynamic_posterior_log_prob
                + likelihood_log_prob
            )
            loss = -elbo
            tf.compat.v2.summary.scalar("elbo", elbo, step=step)

        grads = tape.gradient(loss, model.variables)
        grads, global_norm = tf.clip_by_global_norm(grads, FLAGS.clip_norm)
        grads_and_vars = list(zip(grads, model.variables))  # allow reuse in py3
        if FLAGS.enable_debug_logging:
            with tf.compat.v1.name_scope("grads"):
                tf.compat.v2.summary.scalar("global_norm_grads", global_norm, step=step)
                tf.compat.v2.summary.scalar(
                    "global_norm_grads_clipped", tf.linalg.global_norm(grads), step=step
                )
            for grad, var in grads_and_vars:
                with tf.compat.v1.name_scope("grads"):
                    tf.compat.v2.summary.histogram(
                        "{}/grad".format(var.name), data=grad, step=step
                    )
                with tf.compat.v1.name_scope("vars"):
                    tf.compat.v2.summary.histogram(var.name, data=var, step=step)
        optimizer.apply_gradients(grads_and_vars)

        is_log_step = step.numpy() % FLAGS.log_steps == 0
        is_final_step = step.numpy() == FLAGS.max_steps
        if is_log_step or is_final_step:
            checkpoint_manager.save()
            print("ELBO ({}/{}): {}".format(step.numpy(), FLAGS.max_steps, elbo.numpy()))
            with tf.compat.v2.summary.record_if(True):
                val_data = test_ds.take(20)
                inputs = next(iter(val_data.shuffle(20).batch(3)))
                visualize_qualitative_analysis(
                    inputs, model, step, FLAGS.num_reconstruction_samples
                )
        step.assign_add(1)
        writer.flush()
