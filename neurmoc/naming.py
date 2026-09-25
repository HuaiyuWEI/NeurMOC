"""Experiment names shared by training and evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CovariateConfig:
    """Normalized covariate configuration derived from a comma-separated string."""

    covariate_names: str  # comma-separated versioned mascon variables
    input_var: str        # the same variables joined with "+"

    @property
    def names(self) -> list[str]:
        return [n for n in self.covariate_names.split(",") if n]


def prepare_covariate_config(covariate_names: str) -> CovariateConfig:
    """Normalize the covariate string and derive the results-folder suffix."""
    covariate_list = [name.strip() for name in covariate_names.split(",") if name.strip()]
    return CovariateConfig(
        covariate_names=",".join(covariate_list),
        input_var="+".join(covariate_list),
    )


@dataclass
class TrainingConfig:
    """All knobs of one training run; also the source of the experiment name."""

    cmip_name: str = "ACCESS_hist+SSP585"
    covariate_names: str = "obp_mascon_V7,ssh_mascon_V7,uas_mascon_V7"
    wind_convention: str = "anomaly_2004_2009"
    moc_convention: str = "anomaly_2004_2009"
    lpf_months: int = 24                 # 24 -> use *_LPF_ALL arrays; 0 -> raw
    use_resnet: bool = True

    num_folds: int = 5
    nn_repeats: int = 5
    random_seed: int = 0

    loss_function: str = "mse"
    activation: str = "swish"
    epoch_max: int = 2000
    batch_size: int = 600
    patience: int = 30
    learning_rate: float = 1e-3

    use_scaler_y: bool = True
    scaler_x: str = "standard"           # "standard" | "minmax" | "robust"

    pca_x_variability: float = 0.0
    pca_x_num: int = 0
    pca_y_variability: float = 0.0
    pca_y_num: int = 64

    dropout_rate: float = 0.2
    reg_strength: float = 0.01
    neurons: list[int] = field(default_factory=lambda: [192, 96, 48])

    # ---- derived properties -------------------------------------------------
    def __post_init__(self):
        if not self.wind_convention.strip():
            raise ValueError("wind_convention must be a non-empty provenance label")
        if self.pca_x_variability and self.pca_x_num:
            raise ValueError("pca_x_variability and pca_x_num are mutually exclusive")
        if self.pca_y_variability and self.pca_y_num:
            raise ValueError("pca_y_variability and pca_y_num are mutually exclusive")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")

    @property
    def covariates(self) -> CovariateConfig:
        return prepare_covariate_config(self.covariate_names)

    @property
    def use_pca_x(self) -> bool:
        return bool(self.pca_x_variability or self.pca_x_num)

    @property
    def use_pca_y(self) -> bool:
        return bool(self.pca_y_variability or self.pca_y_num)

    @property
    def lpf_data_key(self) -> str:
        """Suffix of the npz array to load ('_LPF_ALL' for 2-year data)."""
        return "_LPF_ALL" if self.lpf_months == 24 else "_ALL"

    @property
    def lpf_tag(self) -> str:
        """Suffix of the results folder ('_LPF2Year')."""
        return f"_LPF{self.lpf_months // 12}Year" if self.lpf_months else ""

    # ---- experiment name ----------------------------------------------------
    def _model_family(self) -> str:
        return "ResNet" if self.use_resnet else "MLP"

    def _pca_tag(self, include_x: bool, include_y: bool) -> str:
        def fmt(v):
            return int(v) if float(v).is_integer() else v

        x_val = fmt(self.pca_x_num + self.pca_x_variability)
        y_val = fmt(self.pca_y_num + self.pca_y_variability)
        if include_x and include_y:
            return f"PCAinX{x_val}Y{y_val}"
        if include_y:
            return f"PCAinY{y_val}"
        if include_x:
            return f"PCAinX{x_val}"
        return ""

    def experiment_name(self) -> str:
        """The results-folder name encoding this configuration."""
        family = self._model_family()

        # the target is always the full-depth MOC plane, hence the fixed
        # "FullDepth_" stem prefix of every results folder
        inline_pca = (self._pca_tag(self.use_pca_x, self.use_pca_y)
                      if self.use_pca_x else "")
        core = f"{family}{inline_pca}"

        prefix = "FullDepth_"
        if self.use_pca_y:
            prefix += f"{self._pca_tag(self.use_pca_x, self.use_pca_y)}_"
        stem = f"{prefix}{core}"

        activation_tag = f"_{self.activation}Activation" if self.activation != "leaky_relu" else ""
        loss_tag = f"_{self.loss_function}loss" if self.loss_function != "mse" else ""
        scaler_tag = "" if self.use_scaler_y else "NoYScaler_"
        if self.scaler_x == "minmax":
            scaler_tag += "MinMaxXScaler_"
        elif self.scaler_x == "robust":
            scaler_tag += "RobustXScaler_"
        neurons_tag = "x".join(map(str, self.neurons))
        reg_tag = f"Reg{self.reg_strength}" + (
            f"Drop{self.dropout_rate}" if self.dropout_rate else ""
        )
        batch_tag = f"BS{self.batch_size}_" if self.batch_size != 600 else ""
        # Tag settings that distinguish training artifacts.
        ensemble_tag = f"Ens{self.nn_repeats}_" if self.nn_repeats != 5 else ""
        epoch_tag = f"Epoch{self.epoch_max}_" if self.epoch_max != 2000 else ""
        lr_tag = f"LR{self.learning_rate:g}_" if self.learning_rate != 1e-3 else ""
        lr_tag += f"Pat{self.patience}_" if self.patience != 30 else ""
        lr_tag += f"Seed{self.random_seed}_" if self.random_seed != 0 else ""
        cv_tag = f"{self.num_folds}foldCV_" if self.num_folds > 1 else ""

        return (
            f"{stem}_{scaler_tag}Neur{neurons_tag}_"
            f"{batch_tag}{ensemble_tag}{epoch_tag}{lr_tag}{cv_tag}"
            f"{reg_tag}{loss_tag}{activation_tag}{self.lpf_tag}"
        )


def training_config_from_scientific(settings: dict) -> TrainingConfig:
    """Build the default training configuration from the scientific settings."""
    return TrainingConfig(
        cmip_name=str(settings["training_dataset"]),
        covariate_names=str(settings["default_covariates"]),
        wind_convention=str(settings["wind_convention"]),
        moc_convention=str(settings["moc_convention"]),
        lpf_months=int(settings["training_lowpass_months"]),
        pca_y_num=int(settings["target_pca_components"]),
        neurons=[int(n) for n in settings["nn_neurons"]],
        activation=str(settings["nn_activation"]),
        num_folds=int(settings["num_folds"]),
        nn_repeats=int(settings["ensemble_members_per_fold"]),
        random_seed=int(settings["random_seed"]),
    )


def model_output_dir(results_root: Path, experiment_name: str, input_var: str) -> Path:
    """`<results_root>/<experiment>/<covariates>` (matches historical layout)."""
    return Path(results_root) / experiment_name / input_var


def lpf_tag_from_name(experiment_name: str) -> str:
    """Results-folder LPF suffix encoded in an experiment name ('' if none)."""
    match = re.search(r"_LPF\d+Year", experiment_name)
    return match.group(0) if match else ""
