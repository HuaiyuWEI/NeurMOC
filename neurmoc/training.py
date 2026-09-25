"""Load training data and fit cross-validated DBNN ensembles."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np

from .config import BASELINE_YEARS
from .io_utils import ensure_dir, require_file, save_mat
from .naming import TrainingConfig

logger = logging.getLogger("neurmoc.training")


def _software_versions() -> dict[str, str]:
    """Versions needed to reproduce serialized training artifacts."""
    packages = ("numpy", "scipy", "scikit-learn", "joblib", "tensorflow")
    found = {"python": platform.python_version()}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "not installed"
    return found


def _normalize_training_months(
    time,
    realization_index,
    n_samples: int,
    label: str,
) -> np.ndarray:
    """Validate a monthly axis, allowing dates to repeat across members.

    Stage-07 data are ordered by experiment blocks, so the global time axis
    repeats once per realization.  Within each realization, however, rows
    must form one strictly contiguous monthly sequence.
    """
    raw = np.asarray(time).reshape(-1)
    if raw.size != n_samples:
        raise ValueError(f"{label}: {raw.size} timestamps for {n_samples} samples")
    try:
        months = raw.astype("datetime64[M]")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid time_month values") from exc
    if np.isnat(months).any():
        raise ValueError(f"{label}: time_month contains NaT")

    if realization_index is None:
        groups = [np.arange(n_samples)]
    else:
        members = np.asarray(realization_index).reshape(-1)
        if members.size != n_samples:
            raise ValueError(
                f"{label}: {members.size} realization labels for {n_samples} samples"
            )
        # Preserve first-appearance order; sorting mixed string/numeric IDs
        # is unnecessary and can fail for object arrays.
        _, first = np.unique(members, return_index=True)
        groups = [np.flatnonzero(members == members[i]) for i in np.sort(first)]

    for indices in groups:
        member_months = months[indices]
        if member_months.size > 1:
            steps = np.diff(member_months).astype(int)
            if np.any(steps != 1):
                bad = int(np.flatnonzero(steps != 1)[0])
                raise ValueError(
                    f"{label}: time_month is not contiguous within a realization "
                    f"at {member_months[bad]} -> {member_months[bad + 1]}"
                )
    return months


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class TrainingData:
    """Flattened predictors and targets for one CMIP experiment."""

    x: np.ndarray                 # [time, n_features]
    y: np.ndarray                 # [time, n_valid_outputs]
    psi_mask: np.ndarray          # [n_lev * n_lat] validity of the MOC plane
    lat: np.ndarray               # [n_lat]
    rho2: np.ndarray              # [n_lev]
    input_block_sizes: np.ndarray  # features per covariate, in order
    mascon_lon: np.ndarray
    mascon_lat: np.ndarray
    realization_index: np.ndarray | None = None  # stable member ID per sample
    time_month: np.ndarray | None = None  # calendar month per sample (YYYY-MM)
    moc_baseline: dict | None = None  # anomaly-target provenance (see loader)

    @property
    def n_lev(self) -> int:
        return self.rho2.size

    @property
    def n_lat(self) -> int:
        return self.lat.size

    @property
    def block_offsets(self) -> np.ndarray:
        """Cumulative feature offsets, starting at 0 (length n_blocks + 1)."""
        return np.concatenate([[0], np.cumsum(self.input_block_sizes)]).astype(int)


def load_training_data(
    tc: TrainingConfig,
    data_dir: Path | str,
    realization_tag: str = "_r1_r35",
) -> TrainingData:
    """Load MOC targets and mascon predictors saved by the preprocessing stage."""
    data_dir = Path(data_dir)
    lpf_key = tc.lpf_data_key

    moc_file = require_file(data_dir / f"MOC{realization_tag}.npz", "MOC training data")
    with np.load(moc_file) as data:
        rho2 = data["rho2_full"]
        lat = data["lat_psi"]
        psi = np.transpose(data[f"MOC{lpf_key}"], (0, 2, 1))  # [time, lev, lat]
        realization_index = (
            np.asarray(data["realization_index"]) if "realization_index" in data.files
            else None
        )
        target_time = (
            np.asarray(data["time_month"]) if "time_month" in data.files
            else None
        )
        # Without an explicit convention, the target is absolute.
        moc_convention = (
            str(np.asarray(data["moc_convention"]).item())
            if "moc_convention" in data.files else "absolute"
        )
        # Save the anomaly reference state with the trained model.
        moc_baseline = None
        if "MOC_baseline_mean" in data.files:
            moc_baseline = {
                "MOC_baseline_mean": np.asarray(data["MOC_baseline_mean"]),
                "moc_convention": moc_convention,
            }
            for key in ("moc_baseline_period", "moc_baseline_realizations",
                        "moc_baseline_experiment"):
                if key in data.files:
                    moc_baseline[key] = np.asarray(data[key])
    if moc_convention != tc.moc_convention:
        raise ValueError(
            f"{moc_file}: moc_convention is {moc_convention!r}, but "
            f"TrainingConfig expects {tc.moc_convention!r}. Rebuild the MOC "
            "target with stage 06 under the matching convention.")
    if moc_convention == "anomaly_2004_2009":
        if moc_baseline is None:
            raise ValueError(
                f"{moc_file}: anomaly target is missing MOC_baseline_mean; "
                "rerun stage 06 to store the reference state with the target"
            )
        baseline = np.asarray(moc_baseline["MOC_baseline_mean"])
        expected_on_disk = (lat.size, rho2.size)  # stage-06 [lat, lev]
        if baseline.shape != expected_on_disk:
            raise ValueError(
                f"{moc_file}: MOC_baseline_mean has shape {baseline.shape}; "
                f"expected on-disk [lat, lev] {expected_on_disk} "
                f"(model-facing [lev, lat] {(rho2.size, lat.size)})"
            )
        if "moc_baseline_period" not in moc_baseline:
            raise ValueError(
                f"{moc_file}: anomaly baseline is missing moc_baseline_period"
            )
        period = np.asarray(moc_baseline["moc_baseline_period"]).astype(int).reshape(-1)
        expected_period = np.asarray(BASELINE_YEARS, dtype=int)
        if not np.array_equal(period, expected_period):
            raise ValueError(
                f"{moc_file}: MOC baseline period is {period.tolist()}, "
                f"expected {expected_period.tolist()}"
            )

    n_samples = psi.shape[0]

    if realization_index is not None:
        realization_index = np.asarray(realization_index).reshape(-1)
        if realization_index.size != n_samples:
            raise ValueError(
                f"{moc_file}: {realization_index.size} realization labels for "
                f"{n_samples} samples"
            )
    if target_time is not None:
        target_time = _normalize_training_months(
            target_time, realization_index, n_samples, str(moc_file)
        )

    psi_flat = psi.reshape(n_samples, -1)
    psi_mask = ~np.isnan(psi_flat).any(axis=0)
    y = psi_flat[:, psi_mask]

    blocks, sizes = [], []
    mascon_lon = mascon_lat = np.array([])
    for name in tc.covariates.names:
        predictor_file = require_file(
            data_dir / f"{name}{realization_tag}.npz", f"Predictor data for {name}"
        )
        with np.load(predictor_file) as data:
            block = data[f"{name}{lpf_key}"]
            predictor_lon = np.asarray(data["mascon_lon"])
            predictor_lat = np.asarray(data["mascon_lat"])
            predictor_time = (
                np.asarray(data["time_month"])
                if "time_month" in data.files else None
            )
            predictor_realizations = (
                np.asarray(data["realization_index"])
                if "realization_index" in data.files else None
            )
            predictor_wind_convention = (
                str(np.asarray(data["wind_convention"]).item())
                if "wind_convention" in data.files else None
            )
        if block.shape[0] != n_samples:
            raise ValueError(
                f"{predictor_file}: {block.shape[0]} samples do not match the "
                f"MOC target ({n_samples})"
            )
        # Predictors and target must carry the same monthly time coordinate.
        if (target_time is None) != (predictor_time is None):
            missing_side = "predictor" if predictor_time is None else "MOC target"
            raise ValueError(
                f"{predictor_file}: missing time_month on the {missing_side}; "
                "predictor and target must both carry aligned month labels"
            )
        if predictor_time is not None:
            predictor_time = _normalize_training_months(
                predictor_time,
                predictor_realizations,
                block.shape[0],
                str(predictor_file),
            )
            if not np.array_equal(predictor_time, target_time):
                raise ValueError(
                    f"{predictor_file}: time_month is not aligned with the "
                    "MOC target"
                )

        # predictors must share one mascon grid/order
        if mascon_lon.size == 0:
            mascon_lon, mascon_lat = predictor_lon, predictor_lat
        elif not (np.array_equal(predictor_lon, mascon_lon)
                  and np.array_equal(predictor_lat, mascon_lat)):
            raise ValueError(
                f"{predictor_file}: mascon coordinates/order differ from the "
                "other predictors")
        if name.startswith("uas_mascon"):
            if predictor_wind_convention is None:
                raise ValueError(
                    f"{predictor_file}: missing wind_convention provenance. "
                    "Rebuild this predictor with stages 04/06/07 before training."
                )
            if predictor_wind_convention != tc.wind_convention:
                raise ValueError(
                    f"{predictor_file}: wind_convention is "
                    f"{predictor_wind_convention!r}, but TrainingConfig expects "
                    f"{tc.wind_convention!r}"
                )
        if realization_index is not None:
            if predictor_realizations is None:
                raise ValueError(
                    f"{predictor_file}: missing realization_index provenance"
                )
            if not np.array_equal(realization_index, predictor_realizations):
                raise ValueError(
                    f"{predictor_file}: realization_index is not aligned with "
                    "the MOC target"
                )
        blocks.append(block)
        sizes.append(block.shape[1])

    x = np.concatenate(blocks, axis=1) if blocks else np.empty((n_samples, 0))

    logger.info("samples: %d | input blocks: %s | outputs: %d",
                x.shape[0], sizes, y.shape[1])
    return TrainingData(
        x=x, y=y, psi_mask=psi_mask, lat=lat, rho2=rho2,
        input_block_sizes=np.asarray(sizes, dtype=int),
        mascon_lon=mascon_lon, mascon_lat=mascon_lat,
        realization_index=realization_index,
        time_month=target_time,
        moc_baseline=moc_baseline,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _make_scalers(tc: TrainingConfig):
    from sklearn.preprocessing import (
        FunctionTransformer,
        MinMaxScaler,
        RobustScaler,
        StandardScaler,
    )

    scaler_x = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }[tc.scaler_x]()
    scaler_y = StandardScaler() if tc.use_scaler_y else FunctionTransformer()
    return scaler_x, scaler_y


_PCA_SEED_DOMAINS = {"pca_x": 1, "pca_y": 2}


def _pca_seed(base_seed: int, kind: str, fold_no: int, block_no: int = 0) -> int:
    """Derive a stable, domain-separated uint32 seed for one fitted PCA."""
    if kind not in _PCA_SEED_DOMAINS:
        raise ValueError(f"Unknown PCA seed domain: {kind!r}")
    if fold_no < 1 or block_no < 0:
        raise ValueError("PCA fold numbers start at 1 and block numbers at 0")
    sequence = np.random.SeedSequence(
        [int(base_seed), _PCA_SEED_DOMAINS[kind], int(fold_no), int(block_no)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _member_seed(tc: TrainingConfig, fold_no: int, ensemble_no: int) -> int:
    """Unique seed for a fold and ensemble member."""
    return tc.random_seed + (fold_no - 1) * tc.nn_repeats + ensemble_no - 1


def _seed_map(
    tc: TrainingConfig,
    fold_numbers: list[int],
    n_input_blocks: int,
    split_strategy: str,
) -> dict:
    """Return the exact random schedule used by one training invocation."""
    split_seed = None if split_strategy == "GroupKFold" else tc.random_seed
    folds = []
    for fold_no in fold_numbers:
        folds.append({
            "fold": fold_no,
            "pca_x": [
                {
                    "block": block_no,
                    "seed": _pca_seed(tc.random_seed, "pca_x", fold_no, block_no),
                }
                for block_no in range(1, n_input_blocks + 1)
            ] if tc.use_pca_x else [],
            "pca_y": (
                {"seed": _pca_seed(tc.random_seed, "pca_y", fold_no)}
                if tc.use_pca_y else None
            ),
            "members": [
                {
                    "ensemble": ensemble_no,
                    "seed": _member_seed(tc, fold_no, ensemble_no),
                }
                for ensemble_no in range(1, tc.nn_repeats + 1)
            ],
        })
    return {
        "schema_version": 1,
        "base_random_seed": tc.random_seed,
        "split": {
            "strategy": split_strategy,
            "random_state": split_seed,
        },
        "derivation": {
            "pca": "numpy.random.SeedSequence([base, domain, fold, block])",
            "pca_domains": _PCA_SEED_DOMAINS,
            "member": "base + (fold - 1) * nn_repeats + ensemble - 1",
        },
        "folds": folds,
    }


def _fold_indices(
    tc: TrainingConfig,
    n_samples: int,
    realization_index: np.ndarray | None = None,
):
    """Yield deterministic splits that keep each realization in one partition.

    Historical and scenario continuations use the same realization identifier,
    so grouping also prevents those two blocks from leaking across a fold. A
    realization index is mandatory for multi-fold training.
    """
    if tc.num_folds > 1:
        from sklearn.model_selection import GroupKFold

        if realization_index is None:
            raise ValueError(
                "Multi-fold training requires TrainingData.realization_index; "
                "rebuild the stage-06/07 dataset with realization metadata."
            )
        groups = np.asarray(realization_index)
        if groups.shape != (n_samples,):
            raise ValueError(
                "realization_index must have one value per sample "
                f"({groups.shape} != ({n_samples},))"
            )
        n_groups = np.unique(groups).size
        if n_groups < tc.num_folds:
            raise ValueError(
                f"num_folds={tc.num_folds} requires at least that many "
                f"realizations, but the dataset has {n_groups}"
            )

        for fold_no, (tr, te) in enumerate(
            GroupKFold(n_splits=tc.num_folds).split(
                np.arange(n_samples), groups=groups
            ),
            start=1,
        ):
            yield fold_no, tr, te
    else:
        groups = None if realization_index is None else np.asarray(realization_index)
        if groups is not None and groups.shape != (n_samples,):
            raise ValueError(
                "realization_index must have one value per sample "
                f"({groups.shape} != ({n_samples},))"
            )
        if groups is not None and np.unique(groups).size >= 2:
            from sklearn.model_selection import GroupShuffleSplit

            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=1 / 3,
                random_state=tc.random_seed,
            )
            tr, te = next(splitter.split(np.arange(n_samples), groups=groups))
            yield 1, tr, te
        else:
            # A single-realization experiment cannot make a group-independent
            # validation set. Use a deterministic sample split in this
            # explicitly non-CV mode.
            rng = np.random.default_rng(tc.random_seed)
            idx = rng.permutation(n_samples)
            split = int(round(n_samples * 2 / 3))
            yield 1, idx[:split], idx[split:]


def train_experiment(tc: TrainingConfig, data: TrainingData, output_dir: Path) -> dict:
    """Cross-validated ensemble training. Returns a summary dict.

    For each fold: fit scalers (and optional PCA) on the training split
    only, then train `tc.nn_repeats` independently initialized networks.
    """
    import tensorflow as tf
    from sklearn.decomposition import PCA
    from tensorflow.keras.callbacks import Callback, EarlyStopping
    from tensorflow.keras.optimizers import Adam

    from .model import build_dbnn

    output_dir = ensure_dir(output_dir)
    log_path = output_dir / "training_logs.txt"

    # self-describing provenance: the folder name encodes the config, but the
    # JSON dump survives renames and is machine-readable
    config_record = {
        "experiment_name": tc.experiment_name(),
        "neurmoc_version": __import__("neurmoc").__version__,
        "software_versions": _software_versions(),
        "seed_map_file": "seed_map.json",
        **asdict(tc),
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config_record, indent=2))

    class EpochLogger(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 10 == 0:
                msg = (f"Epoch {epoch:>4} | loss {logs.get('loss', 0):.2e} | "
                       f"val_loss {logs.get('val_loss', 0):.2e}")
                print(msg)
                with open(log_path, "a") as fh:
                    fh.write(msg + "\n")

    fold_splits = list(_fold_indices(
        tc, data.x.shape[0], realization_index=data.realization_index
    ))

    def json_scalar(value):
        return value.item() if isinstance(value, np.generic) else value

    split_records = []
    groups = data.realization_index
    for fold_no, train_idx, test_idx in fold_splits:
        record = {
            "fold": fold_no,
            "n_train_samples": int(train_idx.size),
            "n_validation_samples": int(test_idx.size),
        }
        if groups is not None:
            record["training_realizations"] = [
                json_scalar(v) for v in np.unique(groups[train_idx])
            ]
            record["validation_realizations"] = [
                json_scalar(v) for v in np.unique(groups[test_idx])
            ]
        split_records.append(record)
    split_strategy = (
        "GroupKFold" if tc.num_folds > 1 else
        "GroupShuffleSplit" if groups is not None and np.unique(groups).size >= 2 else
        "deterministic_sample_split"
    )
    seed_schedule = _seed_map(
        tc,
        [fold_no for fold_no, _, _ in fold_splits],
        len(data.input_block_sizes),
        split_strategy,
    )
    (output_dir / "seed_map.json").write_text(
        json.dumps(seed_schedule, indent=2)
    )
    fold_seed_records = {
        record["fold"]: record for record in seed_schedule["folds"]
    }
    logger.info("seed map: %s", output_dir / "seed_map.json")
    (output_dir / "cv_splits.json").write_text(json.dumps({
        "strategy": split_strategy,
        "group_key": "realization_index" if groups is not None else None,
        "random_state": seed_schedule["split"]["random_state"],
        "folds": split_records,
    }, indent=2))

    save_mat(output_dir / "inputs_info.mat", {
        "mascon_lon": data.mascon_lon, "mascon_lat": data.mascon_lat,
        "Nsamps": data.x.shape[0], "lat_psi": data.lat,
        "InputNumIndCum": data.block_offsets,
        "covariate_names": tc.covariates.covariate_names,
    })
    save_mat(output_dir / "Psi_mask.mat", {"Psi_mask": data.psi_mask})
    if data.moc_baseline is not None:
        # Store the training target's anomaly reference state.
        np.savez(output_dir / "moc_baseline.npz",
                 lat_psi=data.lat, rho2_full=data.rho2,
                 **data.moc_baseline)
        logger.info("MOC baseline provenance: %s",
                    output_dir / "moc_baseline.npz")

    histories, skills = [], []
    model_summary_written = False

    for fold_no, train_idx, test_idx in fold_splits:
        logger.info("--- fold %d ---", fold_no)
        fold_seed_record = fold_seed_records[fold_no]
        scaler_x, scaler_y = _make_scalers(tc)
        x_train = scaler_x.fit_transform(data.x[train_idx])
        x_test = scaler_x.transform(data.x[test_idx])

        y_train = scaler_y.fit_transform(data.y[train_idx])
        y_test = scaler_y.transform(data.y[test_idx])

        joblib.dump(scaler_x, output_dir / f"scaler_x_fold{fold_no}.pkl")
        joblib.dump(scaler_y, output_dir / f"scaler_y_fold{fold_no}.pkl")

        pcas_x = None
        if tc.use_pca_x:
            pcas_x = []
            parts_train, parts_test = [], []
            offsets = data.block_offsets
            for b in range(1, len(offsets)):
                sl = slice(offsets[b - 1], offsets[b])
                pca_seed = fold_seed_record["pca_x"][b - 1]["seed"]
                pca = PCA(
                    n_components=tc.pca_x_num or tc.pca_x_variability,
                    random_state=pca_seed,
                )
                parts_train.append(pca.fit_transform(x_train[:, sl]))
                parts_test.append(pca.transform(x_test[:, sl]))
                pcas_x.append(pca)
                joblib.dump(
                    pca, output_dir / f"pca_x_fold{fold_no}_block{b}.pkl"
                )
                logger.info(
                    "input block %d: %d -> %d features (PCA seed %d)",
                    b,
                    sl.stop - sl.start,
                    parts_train[-1].shape[1],
                    pca_seed,
                )
            x_train = np.concatenate(parts_train, axis=1)
            x_test = np.concatenate(parts_test, axis=1)

        if tc.use_pca_y:
            pca_y_seed = fold_seed_record["pca_y"]["seed"]
            pca_y = PCA(
                n_components=tc.pca_y_num or tc.pca_y_variability,
                random_state=pca_y_seed,
            )
            y_train = pca_y.fit_transform(y_train)
            y_test = pca_y.transform(y_test)
            joblib.dump(pca_y, output_dir / f"pca_y_fold{fold_no}.pkl")
            logger.info(
                "target PCA: %d components, %.2f%% variance (seed %d)",
                y_train.shape[1],
                100 * np.sum(pca_y.explained_variance_ratio_),
                pca_y_seed,
            )

        for ens_no in range(1, tc.nn_repeats + 1):
            member_seed = fold_seed_record["members"][ens_no - 1]["seed"]
            tf.keras.utils.set_random_seed(member_seed)
            logger.info(
                "training fold %d ensemble %d (seed %d)",
                fold_no,
                ens_no,
                member_seed,
            )
            model = build_dbnn(
                n_inputs=x_train.shape[1], n_outputs=y_train.shape[1],
                neurons=tc.neurons, activation=tc.activation,
                reg_strength=tc.reg_strength, dropout_rate=tc.dropout_rate,
                use_resnet=tc.use_resnet,
            )
            if not model_summary_written:
                with open(output_dir / "model_summary.txt", "w") as fh:
                    model.summary(print_fn=lambda line: fh.write(line + "\n"))
                model_summary_written = True

            model.compile(loss=tc.loss_function, optimizer=Adam(learning_rate=tc.learning_rate))
            early = EarlyStopping(patience=tc.patience, monitor="val_loss",
                                  mode="min", restore_best_weights=True, verbose=1)
            history = model.fit(
                x_train, y_train,
                epochs=tc.epoch_max, validation_data=(x_test, y_test),
                callbacks=[early, EpochLogger()], verbose=0,
                batch_size=tc.batch_size,
            )
            skill = model.evaluate(x_test, y_test, verbose=0)
            model.save(output_dir / f"model_fold{fold_no}_ens{ens_no}.h5")
            histories.append(history.history)
            skills.append(skill)

        tf.keras.backend.clear_session()

    skills = np.asarray(skills)
    summary = {
        "skills": skills,
        "median_skill": float(np.median(skills)),
        "min_skill": float(np.min(skills)),
        "std_skill": float(np.std(skills)),
        "spread_pct": float(np.std(skills) / np.median(skills) * 100),
        "histories": histories,
    }
    logger.info("test losses: %s", np.round(skills, 4).tolist())
    logger.info("median %.4f | min %.4f | spread %.1f%%",
                summary["median_skill"], summary["min_skill"], summary["spread_pct"])
    if summary["spread_pct"] > 30:
        warning = (f"Warning: model-skill spread across folds/ensembles is large "
                   f"(std/median = {summary['spread_pct']:.2f}%)")
        logger.warning(warning)
        (output_dir / f"Warning_{tc.covariates.input_var}.txt").write_text(warning)
    return summary


def plot_training_losses(summary: dict, tc: TrainingConfig, output_dir: Path) -> None:
    """One panel per fold, all ensemble members' train/validation curves."""
    import matplotlib.pyplot as plt

    from .plotting.style import apply_style, save_figure

    apply_style()
    histories = summary["histories"]
    all_losses = [v for h in histories for v in h["loss"] + h["val_loss"]]
    lo, hi = min(all_losses), max(all_losses)

    fig, axes = plt.subplots(tc.num_folds, 1, figsize=(4.5, 1.6 * tc.num_folds),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for fold in range(tc.num_folds):
        ax = axes[fold]
        for ens in range(tc.nn_repeats):
            h = histories[fold * tc.nn_repeats + ens]
            ax.plot(h["loss"], color="0.15", lw=0.8)
            ax.plot(h["val_loss"], color="#D55E00", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.set_ylabel("Loss")
        ax.text(0.98, 0.92, f"fold {fold + 1}", transform=ax.transAxes,
                ha="right", va="top")
    axes[0].legend(["Train", "Validation"], loc="upper center", ncol=2, frameon=False)
    axes[-1].set_xlabel("Epoch")
    save_figure(fig, Path(output_dir) / f"TrainingLoss_{tc.covariates.input_var}",
                formats=("png",))
