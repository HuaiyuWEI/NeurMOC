"""Branch-wise layer-wise relevance propagation for the NeurMOC DBNN.

Maps PCA scores to physical MOC targets before attribution. Input relevance,
network-bias relevance, propagation remainder, and inverse-transform offset
are accounted for separately. LRP-0 rejects singular denominators; supported
pointwise activations use identity pass-through.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Absolute threshold rejects singular LRP-0 denominators; the relative
# threshold records near-cancellation without changing the rule.
LRP0_DENOMINATOR_ATOL = 1e-12
LRP0_RELATIVE_DENOMINATOR_THRESHOLD = 1e-7


def lrp_method_name(epsilon: float, propagation_rule: str | None = None) -> str:
    """Validate and return the unambiguous selected propagation method."""
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"epsilon must be finite and >= 0, got {epsilon!r}")
    if propagation_rule is None:
        propagation_rule = "lrp0" if epsilon == 0.0 else "epsilon"
    propagation_rule = str(propagation_rule).strip().lower()
    if propagation_rule not in {"epsilon", "lrp0"}:
        raise ValueError(
            "propagation_rule must be 'epsilon' or 'lrp0', got "
            f"{propagation_rule!r}"
        )
    if propagation_rule == "epsilon" and epsilon <= 0.0:
        raise ValueError("epsilon propagation requires epsilon > 0")
    if propagation_rule == "lrp0" and epsilon != 0.0:
        raise ValueError("LRP-0 propagation requires epsilon == 0")
    return (
        "branchwise_lrp0_z_rule"
        if propagation_rule == "lrp0"
        else "branchwise_epsilon_lrp"
    )


@dataclass(frozen=True)
class PhysicalOutputHead:
    """Affine map from model PCA scores to one signed physical MOC target."""

    weights: np.ndarray
    offset: float
    output_index: int
    sign: float


@dataclass(frozen=True)
class BranchExplanation:
    """LRP result for one sequential branch of the dual-branch network."""

    relevance: np.ndarray
    output: np.ndarray
    centered_score: np.ndarray
    internal_bias_relevance: np.ndarray
    stabilizer_remainder: np.ndarray
    minimum_absolute_denominator: float
    minimum_relative_denominator: float
    maximum_absolute_message: float
    inactive_zero_denominator_count: int
    low_relative_denominator_count: int


@dataclass(frozen=True)
class LRPExplanation:
    """Combined relevance and accounting diagnostics for both DBNN branches."""

    relevance: np.ndarray
    model_output: np.ndarray
    centered_score: np.ndarray
    physical_prediction: np.ndarray
    branch_scores: np.ndarray
    internal_bias_relevance: np.ndarray
    stabilizer_remainder: np.ndarray
    feature_conservation_residual: np.ndarray
    accounted_conservation_residual: np.ndarray
    minimum_absolute_denominator: float
    minimum_relative_denominator: float
    maximum_absolute_message: float
    inactive_zero_denominator_count: int
    low_relative_denominator_count: int


@dataclass(frozen=True)
class _DenseStep:
    name: str
    inputs: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    preactivation: np.ndarray


@dataclass(frozen=True)
class _PreparedDenseLayer:
    """Weights and bias of one Dense layer plus its pointwise activation."""

    layer: object
    weights: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class _PreparedBranch:
    """Static branch operations cached once for repeated LRP batches."""

    name: str
    operations: tuple[object, ...]


@dataclass(frozen=True)
class PreparedLRPExplainer:
    """A physical-target DBNN explainer reusable across input batches.

    Keras branch graphs and Dense parameters are captured once when this
    object is prepared.  Reusing it is important for long model-test records:
    rebuilding two branch ``Model`` objects for every batch can grow the
    TensorFlow graph and add substantial Python overhead.
    """

    branches: tuple[_PreparedBranch, _PreparedBranch]
    head: PhysicalOutputHead
    epsilon: float
    propagation_rule: str

    def trace_inputs(self, x: np.ndarray) -> tuple:
        """Forward-trace every branch once for reuse across targets.

        The trace is a property of the branch and the batch alone, so the
        same one explains any number of physical targets (see ``explain``).
        """
        return tuple(_trace_sequential_branch(branch, x)
                     for branch in self.branches)

    def explain(
        self,
        x: np.ndarray,
        *,
        traces: tuple | None = None,
        head: PhysicalOutputHead | None = None,
    ) -> LRPExplanation:
        """Explain one standardized input batch with the selected LRP rule.

        ``head`` overrides the prepared target and ``traces`` supplies a
        forward trace from :meth:`trace_inputs`; passing both explains many
        cells from one forward pass without re-preparing the branches.
        """
        head = self.head if head is None else head
        if traces is not None and len(traces) != len(self.branches):
            raise ValueError(
                f"expected {len(self.branches)} branch traces, got {len(traces)}"
            )
        explained = [
            _explain_branch(
                branch,
                x,
                head.weights,
                self.epsilon,
                self.propagation_rule,
                None if traces is None else traces[index],
            )
            for index, branch in enumerate(self.branches)
        ]
        relevance = explained[0].relevance + explained[1].relevance
        model_output = explained[0].output + explained[1].output
        branch_scores = np.stack(
            [item.centered_score for item in explained], axis=1
        )
        centered_score = branch_scores.sum(axis=1)
        internal_bias = sum(
            (item.internal_bias_relevance for item in explained),
            start=np.zeros(centered_score.shape, dtype=float),
        )
        stabilizer = sum(
            (item.stabilizer_remainder for item in explained),
            start=np.zeros(centered_score.shape, dtype=float),
        )
        feature_residual = relevance.sum(axis=1) - centered_score
        accounted_residual = (
            relevance.sum(axis=1)
            + internal_bias
            + stabilizer
            - centered_score
        )

        return LRPExplanation(
            relevance=relevance,
            model_output=model_output,
            centered_score=centered_score,
            physical_prediction=centered_score + head.offset,
            branch_scores=branch_scores,
            internal_bias_relevance=internal_bias,
            stabilizer_remainder=stabilizer,
            feature_conservation_residual=feature_residual,
            accounted_conservation_residual=accounted_residual,
            minimum_absolute_denominator=min(
                item.minimum_absolute_denominator for item in explained
            ),
            maximum_absolute_message=max(
                item.maximum_absolute_message for item in explained
            ),
            minimum_relative_denominator=min(
                item.minimum_relative_denominator for item in explained
            ),
            inactive_zero_denominator_count=sum(
                item.inactive_zero_denominator_count for item in explained
            ),
            low_relative_denominator_count=sum(
                item.low_relative_denominator_count for item in explained
            ),
        )


def physical_output_head(
    scaler_y,
    pca_y,
    output_index: int,
    *,
    sign: float = 1.0,
) -> PhysicalOutputHead:
    """Return the exact affine head for one inverse-transformed output cell.

    Training first standardizes the physical target and then fits PCA.  If
    ``q`` is the network's PCA-score output, physical cell ``j`` is

    ``(q @ components[:, j] + pca.mean_[j]) * scale_y[j] + mean_y[j]``.

    ``sign=-1`` is useful for a diagnostic whose positive direction is
    defined as ``-Psi``.  Both the weights and offset are then sign-flipped.
    """
    sign = float(sign)
    if sign not in (-1.0, 1.0):
        raise ValueError(f"sign must be +1 or -1, got {sign!r}")

    scale = np.asarray(getattr(scaler_y, "scale_", None), dtype=float)
    mean = np.asarray(getattr(scaler_y, "mean_", None), dtype=float)
    if scale.ndim != 1 or mean.shape != scale.shape:
        raise TypeError(
            "LRP requires a fitted StandardScaler-like target transform with "
            "one-dimensional mean_ and scale_ arrays"
        )
    output_index = int(output_index)
    if not 0 <= output_index < scale.size:
        raise IndexError(f"physical output index {output_index} is outside 0..{scale.size - 1}")

    if pca_y is None:
        weights = np.zeros(scale.size, dtype=float)
        weights[output_index] = scale[output_index]
        offset = mean[output_index]
    else:
        components = np.asarray(getattr(pca_y, "components_", None), dtype=float)
        pca_mean = np.asarray(getattr(pca_y, "mean_", None), dtype=float)
        if components.ndim != 2 or components.shape[1] != scale.size:
            raise TypeError(
                "PCA-Y components are incompatible with the target scaler: "
                f"components={components.shape}, target_width={scale.size}"
            )
        if pca_mean.shape != scale.shape:
            raise TypeError(f"PCA-Y mean has shape {pca_mean.shape}; expected {scale.shape}")
        weights = components[:, output_index] * scale[output_index]
        offset = pca_mean[output_index] * scale[output_index] + mean[output_index]

    return PhysicalOutputHead(
        weights=sign * np.asarray(weights, dtype=float),
        offset=sign * float(offset),
        output_index=output_index,
        sign=sign,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function used by NumPy Swish."""
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _apply_activation(layer, values: np.ndarray) -> np.ndarray:
    """Evaluate the small set of pointwise activations used by NeurMOC."""
    class_name = layer.__class__.__name__
    config = layer.get_config()
    if class_name == "Activation":
        name = str(config.get("activation", "linear"))
    elif class_name == "LeakyReLU":
        alpha = float(config.get("alpha", config.get("negative_slope", 0.2)))
        return np.where(values >= 0, values, alpha * values)
    elif class_name == "PReLU":
        learned = layer.get_weights()
        if len(learned) != 1:
            raise RuntimeError(f"{layer.name}: malformed PReLU weights")
        alpha = np.asarray(learned[0], dtype=float)
        return np.where(values >= 0, values, alpha * values)
    else:
        name = str(config.get("activation", "linear"))

    if name == "linear":
        return values
    if name == "swish":
        return values * _sigmoid(values)
    if name == "relu":
        return np.maximum(values, 0.0)
    if name == "sigmoid":
        return _sigmoid(values)
    if name == "tanh":
        return np.tanh(values)
    if name == "elu":
        return np.where(values >= 0, values, np.expm1(values))
    if name == "gelu":
        # TensorFlow/Keras' default approximate=False definition.
        from scipy.special import erf

        return 0.5 * values * (1.0 + erf(values / np.sqrt(2.0)))
    raise TypeError(f"{layer.name}: activation {name!r} is not supported by NeurMOC LRP")


def _prepare_branch(
    branch_model,
    dense_cache: dict[int, _PreparedDenseLayer] | None = None,
) -> _PreparedBranch:
    """Freeze the supported operations of one Keras branch."""
    dense_cache = {} if dense_cache is None else dense_cache
    operations: list[object] = []
    for layer in branch_model.layers[1:]:  # skip InputLayer
        class_name = layer.__class__.__name__
        if class_name == "Dense":
            prepared = dense_cache.get(id(layer))
            if prepared is None:
                params = layer.get_weights()
                if len(params) != 2:
                    raise RuntimeError(
                        f"{layer.name}: LRP requires a Dense kernel and bias"
                    )
                weights, bias = (
                    np.asarray(value, dtype=float) for value in params
                )
                prepared = _PreparedDenseLayer(layer, weights, bias)
                dense_cache[id(layer)] = prepared
            operations.append(prepared)
        elif class_name in {"Activation", "LeakyReLU", "PReLU"}:
            operations.append(layer)
        elif class_name == "Dropout":
            # Dropout is exactly the identity during inference.
            continue
        else:
            raise TypeError(
                f"{branch_model.name}: unsupported layer {layer.name} "
                f"({class_name}); this LRP implementation is intentionally "
                "restricted to the validated NeurMOC DBNN architecture"
            )
    return _PreparedBranch(
        name=str(branch_model.name), operations=tuple(operations)
    )


def _trace_sequential_branch(
    branch: _PreparedBranch,
    x: np.ndarray,
) -> tuple[np.ndarray, list[_DenseStep]]:
    """Reproduce one prepared branch and retain its Dense inputs."""
    current = np.asarray(x, dtype=float)
    if current.ndim != 2:
        raise ValueError(f"LRP input must be [sample, feature], got {current.shape}")
    dense_steps: list[_DenseStep] = []

    for operation in branch.operations:
        if isinstance(operation, _PreparedDenseLayer):
            layer = operation.layer
            weights = operation.weights
            bias = operation.bias
            if current.shape[1] != weights.shape[0]:
                raise RuntimeError(
                    f"{layer.name}: activation width {current.shape[1]} does not "
                    f"match Dense kernel {weights.shape}"
                )
            preactivation = current @ weights + bias
            dense_steps.append(
                _DenseStep(
                    name=str(layer.name),
                    inputs=current,
                    weights=weights,
                    bias=bias,
                    preactivation=preactivation,
                )
            )
            current = _apply_activation(layer, preactivation)
        else:
            current = _apply_activation(operation, current)
    return current, dense_steps


def _stabilize(values: np.ndarray, epsilon: float) -> np.ndarray:
    sign = np.where(values >= 0.0, 1.0, -1.0)
    return values + epsilon * sign


def _explain_branch(
    branch: _PreparedBranch,
    x: np.ndarray,
    head_weights: np.ndarray,
    epsilon: float,
    propagation_rule: str,
    trace: tuple[np.ndarray, tuple] | None = None,
) -> BranchExplanation:
    # Reuse the forward trace when explaining several outputs for one batch.
    output, dense_steps = (
        trace if trace is not None else _trace_sequential_branch(branch, x)
    )
    head_weights = np.asarray(head_weights, dtype=float).reshape(-1)
    if output.shape[1] != head_weights.size:
        raise RuntimeError(
            f"{branch.name}: output width {output.shape[1]} does not "
            f"match physical-head width {head_weights.size}"
        )

    # The physical head has zero bias here: its inverse-transform offset is
    # intentionally retained as a separate, non-spatial term.
    relevance = output * head_weights[None, :]
    centered_score = relevance.sum(axis=1)
    bias_total = np.zeros(output.shape[0], dtype=float)
    stabilizer_total = np.zeros(output.shape[0], dtype=float)
    minimum_absolute_denominator = np.inf
    minimum_relative_denominator = np.inf
    maximum_absolute_message = 0.0
    inactive_zero_denominator_count = 0
    low_relative_denominator_count = 0

    for step in reversed(dense_steps):
        active = relevance != 0.0
        if propagation_rule == "lrp0":
            absolute = np.abs(step.preactivation)
            contribution_scale = (
                np.abs(step.inputs) @ np.abs(step.weights)
                + np.abs(step.bias)[None, :]
            )
            relative = np.full(absolute.shape, np.inf, dtype=float)
            np.divide(
                absolute,
                contribution_scale,
                out=relative,
                where=contribution_scale > 0.0,
            )
            active_absolute = absolute[active]
            active_relative = relative[active]
            if active_absolute.size:
                step_minimum = float(np.min(active_absolute))
                step_relative_minimum = float(np.min(active_relative))
            else:
                step_minimum = np.inf
                step_relative_minimum = np.inf
            low_relative = active & (
                relative <= LRP0_RELATIVE_DENOMINATOR_THRESHOLD
            )
            low_relative_denominator_count += int(
                np.count_nonzero(low_relative)
            )
            unsafe = active & (absolute <= LRP0_DENOMINATOR_ATOL)
            if np.any(unsafe):
                first_sample, first_unit = np.argwhere(unsafe)[0]
                first_z = float(step.preactivation[first_sample, first_unit])
                first_relevance = float(relevance[first_sample, first_unit])
                first_scale = float(
                    contribution_scale[first_sample, first_unit]
                )
                implied_message = (
                    first_relevance / first_z if first_z != 0.0 else np.inf
                )
                raise FloatingPointError(
                    f"{branch.name}/{step.name}: LRP-0 z-rule is undefined or "
                    f"numerically singular for {int(np.count_nonzero(unsafe))} "
                    "relevance-active preactivation(s); "
                    f"minimum |z|={step_minimum:.3e}, minimum relative "
                    f"denominator={step_relative_minimum:.3e}, first at "
                    f"sample={int(first_sample)}, unit={int(first_unit)}; "
                    f"z={first_z:.3e}, contribution scale={first_scale:.3e}, "
                    f"incoming relevance={first_relevance:.3e}, implied "
                    f"message={implied_message:.3e}. "
                    f"The absolute hard limit is |z|>"
                    f"{LRP0_DENOMINATOR_ATOL:.1e}; relative cancellation below "
                    f"{LRP0_RELATIVE_DENOMINATOR_THRESHOLD:.1e} is recorded "
                    "separately. Inspect the event and compare a positive-"
                    "epsilon sensitivity product."
                )
            denominator = step.preactivation
            inactive_zero_denominator_count += int(
                np.count_nonzero((~active) & (denominator == 0.0))
            )
            message = np.zeros_like(relevance, dtype=float)
            np.divide(
                relevance,
                denominator,
                out=message,
                where=denominator != 0.0,
            )
        else:
            denominator = _stabilize(step.preactivation, epsilon)
            active_absolute = np.abs(denominator)[active]
            step_minimum = (
                float(np.min(active_absolute)) if active_absolute.size else np.inf
            )
            step_relative_minimum = np.inf
            message = relevance / denominator
        minimum_absolute_denominator = min(
            minimum_absolute_denominator, step_minimum
        )
        minimum_relative_denominator = min(
            minimum_relative_denominator, step_relative_minimum
        )
        maximum_absolute_message = max(
            maximum_absolute_message, float(np.max(np.abs(message)))
        )
        relevance_in = step.inputs * (message @ step.weights.T)

        # Account for bias and stabilizer relevance outside input maps.
        bias_part = (message * step.bias[None, :]).sum(axis=1)
        stabilizer_part = relevance.sum(axis=1) - relevance_in.sum(axis=1) - bias_part
        bias_total += bias_part
        stabilizer_total += stabilizer_part
        relevance = relevance_in

    if not (
        np.isfinite(relevance).all()
        and np.isfinite(bias_total).all()
        and np.isfinite(stabilizer_total).all()
        and np.isfinite(maximum_absolute_message)
    ):
        method = "LRP-0" if propagation_rule == "lrp0" else "epsilon LRP"
        raise FloatingPointError(
            f"{branch.name}: {method} produced non-finite values; inspect "
            "the signed activations and compare a positive-epsilon sensitivity"
        )
    return BranchExplanation(
        relevance=relevance,
        output=output,
        centered_score=centered_score,
        internal_bias_relevance=bias_total,
        stabilizer_remainder=stabilizer_total,
        minimum_absolute_denominator=minimum_absolute_denominator,
        minimum_relative_denominator=minimum_relative_denominator,
        maximum_absolute_message=maximum_absolute_message,
        inactive_zero_denominator_count=inactive_zero_denominator_count,
        low_relative_denominator_count=low_relative_denominator_count,
    )


def prepare_dbnn_physical_target_explainer(
    model,
    head: PhysicalOutputHead,
    *,
    epsilon: float = 1e-6,
    propagation_rule: str = "epsilon",
) -> PreparedLRPExplainer:
    """Prepare a reusable branch-wise explainer for one physical target.

    The current DBNN has a deep and a linear branch whose PCA outputs are
    summed by one final ``Add`` layer.  Preparing the two branches once avoids
    rebuilding Keras graph objects and re-reading Dense weights for every
    time batch.
    """
    epsilon = float(epsilon)
    lrp_method_name(epsilon, propagation_rule)
    propagation_rule = str(propagation_rule).strip().lower()

    add_layers = [layer for layer in model.layers if layer.__class__.__name__ == "Add"]
    if len(add_layers) != 1:
        raise TypeError(
            f"NeurMOC branch-wise LRP requires exactly one final Add layer; found {len(add_layers)}"
        )
    add_layer = add_layers[0]
    branch_tensors = list(add_layer.input)
    if len(branch_tensors) != 2:
        raise TypeError(f"{add_layer.name}: expected two branches, got {len(branch_tensors)}")

    # Import lazily so the core package remains usable without the optional
    # neural-network dependency.
    from tensorflow.keras import Model

    branches = [
        Model(model.input, tensor, name=f"{model.name}_lrp_branch_{number}")
        for number, tensor in enumerate(branch_tensors, start=1)
    ]
    dense_cache: dict[int, _PreparedDenseLayer] = {}
    return PreparedLRPExplainer(
        branches=tuple(
            _prepare_branch(branch, dense_cache) for branch in branches
        ),
        head=head,
        epsilon=epsilon,
        propagation_rule=propagation_rule,
    )


def explain_dbnn_physical_target(
    model,
    x: np.ndarray,
    head: PhysicalOutputHead,
    *,
    epsilon: float = 1e-6,
    propagation_rule: str = "epsilon",
) -> LRPExplanation:
    """Apply branch-wise epsilon LRP or exact LRP-0.

    This convenience form prepares an explainer for one call.  Long-running
    batched workflows should call :func:`prepare_dbnn_physical_target_explainer`
    once per network member and reuse its ``explain`` method.
    """
    return prepare_dbnn_physical_target_explainer(
        model,
        head,
        epsilon=epsilon,
        propagation_rule=propagation_rule,
    ).explain(x)
