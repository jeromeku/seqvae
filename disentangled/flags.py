from absl import flags

flags.DEFINE_integer("batch_size", default=32, help="Batch size during training.")
flags.DEFINE_float(
    "clip_norm", default=1e10, help="Threshold for global norm gradient clipping."
)
flags.DEFINE_boolean(
    "enable_debug_logging",
    default=None,
    help="Whether or not to include extra TensorBoard logging for debugging.",
)
flags.DEFINE_boolean(
    "fake_data", default=None, help="Whether or not to train with synthetic data."
)
flags.DEFINE_integer(
    "hidden_size", default=512, help="Dimensionality of the model intermediates."
)
flags.DEFINE_enum(
    "latent_posterior",
    default="factorized",
    enum_values=["factorized", "full"],
    help="The formulation for the latent posterior `q`.",
)
flags.DEFINE_integer(
    "latent_size_dynamic",
    default=32,
    help="Dimensionality of each dynamic, time-variant latent variable `z_t`.",
)
flags.DEFINE_integer(
    "latent_size_static",
    default=256,
    help="Dimensionality of the static, time-invariant latent variable `f`.",
)
flags.DEFINE_float(
    "learning_rate", default=0.0001, help="Learning rate during training."
)
flags.DEFINE_string(
    "logdir",  # `log_dir` is already defined by absl
    default="/tmp/disentangled_vae/logs/{timestamp}",
    help="Directory in which to write TensorBoard logs.",
)
flags.DEFINE_integer(
    "log_steps", default=100, help="Frequency, in steps, of TensorBoard logging."
)
flags.DEFINE_integer(
    "max_steps", default=10000, help="Number of steps over which to train."
)
flags.DEFINE_string(
    "model_dir",
    default="/tmp/disentangled_vae/models/{timestamp}",
    help="Directory in which to save model checkpoints.",
)
flags.DEFINE_integer(
    "num_reconstruction_samples",
    default=1,
    help="Number of samples to use during reconstruction evaluation.",
)
flags.DEFINE_integer(
    "num_samples", default=4, help="Number of samples to use during training."
)
flags.DEFINE_integer("seed", default=42, help="Random seed.")

FLAGS = flags.FLAGS
