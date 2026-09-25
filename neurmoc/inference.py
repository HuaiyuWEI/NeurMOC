"""Load trained ensembles and produce plain or Monte-Carlo predictions."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from .io_utils import require_file


def _resolve_ensemble_counts(
    training_config: dict | None,
    n_folds: int | None,
    n_ensembles: int | None,
) -> tuple[int, int]:
    """Resolve requested counts and reject disagreement with provenance."""
    recorded_folds = (
        int(training_config["num_folds"])
        if training_config is not None and "num_folds" in training_config
        else None
    )
    recorded_ensembles = (
        int(training_config["nn_repeats"])
        if training_config is not None and "nn_repeats" in training_config
        else None
    )
    for label, requested, recorded in (
        ("n_folds", n_folds, recorded_folds),
        ("n_ensembles", n_ensembles, recorded_ensembles),
    ):
        if requested is not None and recorded is not None and requested != recorded:
            raise ValueError(
                f"requested {label}={requested}, but training_config.json "
                f"records {recorded}"
            )
    folds = n_folds if n_folds is not None else (recorded_folds or 5)
    ensembles = (
        n_ensembles if n_ensembles is not None else (recorded_ensembles or 5)
    )
    if folds < 1 or ensembles < 1:
        raise ValueError("ensemble fold/member counts must be positive")
    return int(folds), int(ensembles)


def _check_sklearn_version(
    training_config: dict | None,
    nn_path: Path,
    allow_mismatch: bool,
) -> None:
    """Fail before unpickling estimators under a different sklearn version."""
    recorded = None
    if training_config is not None:
        recorded = training_config.get("software_versions", {}).get("scikit-learn")
    if not recorded or recorded == "not installed":
        return
    try:
        current = version("scikit-learn")
    except PackageNotFoundError as exc:
        raise RuntimeError("scikit-learn is required to load trained artifacts") from exc
    if current == recorded:
        return
    message = (
        f"{nn_path}: artifacts were serialized with scikit-learn {recorded}, "
        f"but this process uses {current}. Cross-version estimator loading is "
        "unsupported; use the recorded environment"
    )
    if not allow_mismatch:
        raise RuntimeError(message)
    warnings.warn(message + " (explicit override enabled)", RuntimeWarning, stacklevel=2)


def _load_joblib(path: Path, allow_sklearn_version_mismatch: bool):
    """Load one artifact and promote sklearn's pickle warning to an error."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = joblib.load(require_file(path))
    mismatch = [
        warning for warning in caught
        if "Trying to unpickle estimator" in str(warning.message)
        and "version" in str(warning.message)
    ]
    if mismatch and not allow_sklearn_version_mismatch:
        raise RuntimeError(
            f"{path}: {mismatch[0].message} Use the scikit-learn version that "
            "created the artifact, or retrain it."
        )
    for warning in caught:
        warnings.warn(
            warning.message, warning.category, stacklevel=2
        )
    return value


@dataclass
class TrainedEnsemble:
    """All artifacts of one cross-validated ensemble, preloaded in memory."""

    models: list          # models[fold][ens]
    scalers_x: list
    scalers_y: list
    pcas_x: list | None   # pcas_x[fold][block], None without input PCA
    pcas_y: list | None   # None when the run did not use target PCA
    n_folds: int
    n_ensembles: int
    training_config: dict | None = None

    @classmethod
    def load(cls, nn_path: Path | str, n_folds: int | None = None,
             n_ensembles: int | None = None,
             use_pca: bool | None = None, use_pca_x: bool | None = None,
             use_pca_y: bool | None = None,
             require_complete: bool = True,
             allow_sklearn_version_mismatch: bool = False) -> "TrainedEnsemble":
        """Load a completed ensemble using its saved configuration.

        PCA settings and member counts are inferred from artifacts. Serialized
        sklearn estimators require the recorded version unless explicitly
        overridden.
        """
        nn_path = Path(nn_path)
        config_path = nn_path / "training_config.json"
        training_config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else None
        )
        n_folds, n_ensembles = _resolve_ensemble_counts(
            training_config, n_folds, n_ensembles
        )
        _check_sklearn_version(
            training_config, nn_path, allow_sklearn_version_mismatch
        )
        if require_complete and not (nn_path / "08_train_nn.py").is_file():
            raise RuntimeError(
                f"{nn_path}: 08_train_nn.py is missing, so training did not "
                "finish; pass require_complete=False to load it anyway.")
        found_pca_y = (nn_path / "pca_y_fold1.pkl").is_file()
        found_pca_x = any(nn_path.glob("pca_x_fold1_block*.pkl"))
        if use_pca_x is None:
            use_pca_x = found_pca_x
        if use_pca_y is None:
            use_pca_y = found_pca_y
            # A PCA hint without PCA artifacts is invalid; PCA-X-only runs
            # remain valid.
            if use_pca is True and not found_pca_x and not found_pca_y:
                use_pca_y = True

        def pca_x_paths(fold: int) -> list[Path]:
            paths = list(nn_path.glob(f"pca_x_fold{fold}_block*.pkl"))

            def block_number(path: Path) -> int:
                match = re.fullmatch(
                    rf"pca_x_fold{fold}_block(\d+)\.pkl", path.name
                )
                if match is None:
                    raise ValueError(f"Malformed PCA-X artifact name: {path.name}")
                return int(match.group(1))

            paths.sort(key=block_number)
            numbers = [block_number(path) for path in paths]
            if numbers and numbers != list(range(1, len(numbers) + 1)):
                raise ValueError(
                    f"PCA-X blocks for fold {fold} are not contiguous: {numbers}"
                )
            return paths


        expected_models = {
            (fold, ens)
            for fold in range(1, n_folds + 1)
            for ens in range(1, n_ensembles + 1)
        }
        found_models = set()
        for path in nn_path.glob("model_fold*_ens*.h5"):
            match = re.fullmatch(r"model_fold(\d+)_ens(\d+)\.h5", path.name)
            if match is None:
                raise ValueError(f"Malformed model artifact name: {path.name}")
            found_models.add((int(match.group(1)), int(match.group(2))))
        if found_models != expected_models:
            missing = sorted(expected_models - found_models)
            extra = sorted(found_models - expected_models)
            raise RuntimeError(
                f"{nn_path}: model artifact grid is incomplete or mixed; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        models, scalers_x, scalers_y = [], [], []
        pcas_x = [] if use_pca_x else None
        pcas_y = [] if use_pca_y else None
        expected_x_blocks = None
        for fold in range(1, n_folds + 1):
            scalers_x.append(_load_joblib(
                nn_path / f"scaler_x_fold{fold}.pkl",
                allow_sklearn_version_mismatch,
            ))
            scalers_y.append(_load_joblib(
                nn_path / f"scaler_y_fold{fold}.pkl",
                allow_sklearn_version_mismatch,
            ))
            if use_pca_x:
                paths = pca_x_paths(fold)
                if not paths:
                    raise FileNotFoundError(
                        f"No PCA-X artifacts found for fold {fold} in {nn_path}"
                    )
                if expected_x_blocks is None:
                    expected_x_blocks = len(paths)
                elif len(paths) != expected_x_blocks:
                    raise ValueError(
                        f"Fold {fold} has {len(paths)} PCA-X blocks; expected "
                        f"{expected_x_blocks}"
                    )
                pcas_x.append([
                    _load_joblib(path, allow_sklearn_version_mismatch)
                    for path in paths
                ])
            if use_pca_y:
                pcas_y.append(_load_joblib(
                    nn_path / f"pca_y_fold{fold}.pkl",
                    allow_sklearn_version_mismatch,
                ))
            fold_models = []
            for ens in range(1, n_ensembles + 1):
                fold_models.append(load_model(
                    require_file(nn_path / f"model_fold{fold}_ens{ens}.h5")))
            models.append(fold_models)
        return cls(
            models=models,
            scalers_x=scalers_x,
            scalers_y=scalers_y,
            pcas_x=pcas_x,
            pcas_y=pcas_y,
            n_folds=n_folds,
            n_ensembles=n_ensembles,
            training_config=training_config,
        )

    # -----------------------------------------------------------------
    def transform_inputs(self, fold: int, x_raw: np.ndarray) -> np.ndarray:
        """Apply one fold's fitted scaler and optional blockwise PCA-X."""
        x = self.scalers_x[fold].transform(x_raw)
        if self.pcas_x is None:
            return x

        parts = []
        start = 0
        for block_no, pca in enumerate(self.pcas_x[fold], start=1):
            stop = start + int(pca.n_features_in_)
            if stop > x.shape[1]:
                raise ValueError(
                    f"PCA-X block {block_no} extends past input width "
                    f"({stop} > {x.shape[1]})"
                )
            parts.append(pca.transform(x[:, start:stop]))
            start = stop
        if start != x.shape[1]:
            raise ValueError(
                f"PCA-X artifacts cover {start} features, but input has {x.shape[1]}"
            )
        return np.concatenate(parts, axis=1)

    def predict_fold(self, fold: int, x_raw: np.ndarray) -> np.ndarray:
        """Ensemble-mean prediction of one fold, in physical units."""
        x = self.transform_inputs(fold, x_raw)
        preds = []
        for model in self.models[fold]:
            y = model.predict(x, verbose=0)
            if self.pcas_y is not None:
                y = self.pcas_y[fold].inverse_transform(y)
            preds.append(y)
        return self.scalers_y[fold].inverse_transform(np.mean(preds, axis=0))

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        """Mean over folds of the ensemble-mean prediction (`[time, n_out]`)."""
        return np.mean([self.predict_fold(f, x_raw) for f in range(self.n_folds)], axis=0)

    def predict_all_members(self, x_raw: np.ndarray) -> np.ndarray:
        """Every member's prediction: `[n_folds * n_ensembles, time, n_out]`.

        Used for the real-world reconstruction, where the member spread is
        the epistemic uncertainty.
        """
        out = []
        for fold in range(self.n_folds):
            x = self.transform_inputs(fold, x_raw)
            for model in self.models[fold]:
                y = model.predict(x, verbose=0)
                if self.pcas_y is not None:
                    y = self.pcas_y[fold].inverse_transform(y)
                out.append(self.scalers_y[fold].inverse_transform(y))
        return np.asarray(out)
