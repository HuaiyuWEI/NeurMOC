"""Keras model builders and custom layers for the dual-branch NN (DBNN)."""

from __future__ import annotations

import logging
import os

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, Add, Dense, Dropout, LeakyReLU, PReLU
from tensorflow.keras.models import Model

_RUNTIME_CONFIGURED = False


def tensorflow_runtime_info() -> dict[str, object]:
    """Collect concise TensorFlow build and accelerator diagnostics."""
    build_info = tf.sysconfig.get_build_info()
    physical_gpus = tf.config.list_physical_devices("GPU")
    gpu_descriptions = []
    for device in physical_gpus:
        details = tf.config.experimental.get_device_details(device)
        description = str(details.get("device_name", device.name))
        capability = details.get("compute_capability")
        if capability is not None:
            description += f" (compute capability {capability})"
        gpu_descriptions.append(description)
    return {
        "tensorflow_version": tf.__version__,
        "eager": bool(tf.executing_eagerly()),
        "cuda_build": bool(tf.test.is_built_with_cuda()),
        "cuda_version": build_info.get("cuda_version", "not reported"),
        "cudnn_version": build_info.get("cudnn_version", "not reported"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"),
        "cpu_count": len(tf.config.list_physical_devices("CPU")),
        "gpu_count": len(physical_gpus),
        "gpus": gpu_descriptions,
    }


def _log_tensorflow_runtime(info: dict[str, object]) -> None:
    logger = logging.getLogger(__name__)
    logger.info(
        "TensorFlow %s | eager=%s | CUDA build=%s | physical CPUs=%d",
        info["tensorflow_version"],
        info["eager"],
        info["cuda_build"],
        info["cpu_count"],
    )
    logger.info(
        "CUDA version=%s | cuDNN version=%s | CUDA_VISIBLE_DEVICES=%s",
        info["cuda_version"],
        info["cudnn_version"],
        info["cuda_visible_devices"],
    )
    if info["gpu_count"]:
        logger.info(
            "TensorFlow physical GPUs (%d): %s",
            info["gpu_count"],
            "; ".join(info["gpus"]),
        )
    else:
        logger.warning("TensorFlow sees no physical GPU; neural-network work will use CPU.")


def configure_tensorflow_runtime(
    seed: int = 0,
    *,
    require_fresh_console: bool = False,
) -> dict[str, object]:
    """Set deterministic seeds and graph mode, and report accelerator status.

    ``require_fresh_console`` is used by Stage 08 training: graph mode is a
    process-wide TensorFlow setting, so training must start in a new Python
    process.
    """
    import warnings

    global _RUNTIME_CONFIGURED

    warnings.filterwarnings("ignore", category=UserWarning, module="keras.engine.training_v1")
    info = tensorflow_runtime_info()
    _log_tensorflow_runtime(info)

    if require_fresh_console and (_RUNTIME_CONFIGURED or not info["eager"]):
        raise RuntimeError(
            "Stage 08 training must run in a new Python process "
            "(TensorFlow graph mode is already enabled in this one).")

    tf.keras.utils.set_random_seed(seed)
    if tf.executing_eagerly():
        try:
            tf.compat.v1.disable_eager_execution()
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "TensorFlow could not enter graph mode; start a new Python process."
            ) from exc
    _RUNTIME_CONFIGURED = True
    return info


_SIMPLE_ACTIVATIONS = {"relu", "sigmoid", "tanh", "elu", "linear", "gelu",
                       "swish"}


def build_dbnn(
    n_inputs: int,
    n_outputs: int,
    neurons: list[int],
    activation: str = "leaky_relu",
    reg_strength: float = 0.01,
    dropout_rate: float = 0.2,
    use_resnet: bool = True,
) -> Model:
    """Build the dual-branch (deep + linear skip) fully connected network.

    The deep branch stacks `neurons` dense layers; when `use_resnet` is set,
    a single linear layer taps the first hidden layer and its output is
    added to the deep branch's output (the "dual-branch NN" of the paper).
    """

    def activate(x):
        if activation == "leaky_relu":
            return LeakyReLU(alpha=0.2)(x)
        if activation == "prelu":
            return PReLU()(x)
        if activation in _SIMPLE_ACTIVATIONS:
            return Activation(activation)(x)
        raise ValueError(f"Unsupported activation function: {activation}")

    inputs_raw = tf.keras.Input(shape=(n_inputs,))
    first_dense = Dense(neurons[0],
                        kernel_regularizer=regularizers.l2(reg_strength))(inputs_raw)
    out = activate(first_dense)
    if dropout_rate:
        out = Dropout(dropout_rate)(out)

    for width in neurons[1:]:
        out = Dense(width, kernel_regularizer=regularizers.l2(reg_strength))(out)
        out = activate(out)
        if dropout_rate:
            out = Dropout(dropout_rate)(out)

    if use_resnet:
        skip = Dropout(dropout_rate)(first_dense) if dropout_rate else first_dense
        skip = Dense(n_outputs, activation="linear",
                     kernel_regularizer=regularizers.l2(reg_strength))(skip)
        out = Dense(n_outputs, activation="linear")(out)
        out = Add(name="add_layer")([out, skip])
    else:
        out = Dense(n_outputs, activation="linear")(out)

    return Model(inputs=inputs_raw, outputs=out)
