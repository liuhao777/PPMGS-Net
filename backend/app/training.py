from __future__ import annotations

"""Standalone implementation of the final PPMGS-Net workflow.

This module contains only the method reported in the manuscript:

* trait-specific GWAS--LASSO prior construction;
* a genome-wide path and a trait-specific prior-gated path with learned fusion;
* independently trained, frozen trait anchors;
* bounded low-rank asymmetric directional residual transfer;
* cross-fitted, calibrated and confidence-filtered PDAE supervision;
* TPE tuning, repeated cross-validation, OOF prediction intervals and SHAP.

``transfer_mode='none'`` is retained only for the anchor-only ablation of the
same final architecture.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge as SklearnRidge
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from .gwas_prior import build_tassel_mlm_prior

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:  # pragma: no cover
    optuna = None
    TrialPruned = RuntimeError


NA_VALUES = ["", "NA", "NaN", "nan", "null", "None"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
TRAINING_BASE_SEED = 42
JOB_STORE: dict[str, object] = {}
TASK_TYPES = {"single_trait": "single-trait", "multi_trait": "multi-trait"}

DIRECTIONAL_ANCHOR_MODE = "directional_anchor"
PRIOR_MARKER_ATTENTION_MODES = {"prior_marker_pearson"}
PRIOR_REQUIRED_ATTENTION_MODES = set(PRIOR_MARKER_ATTENTION_MODES)
PRIOR_RELIABILITY_ATTENTION_MODES = set(PRIOR_MARKER_ATTENTION_MODES)
PRIOR_COMPATIBLE_ATTENTION_MODES = set(PRIOR_MARKER_ATTENTION_MODES)
ATTENTION_AUXILIARY_MODES = set(PRIOR_MARKER_ATTENTION_MODES)
LR_SCHEDULER_MODES = {"none", "plateau", "cosine"}
INDIVIDUALIZED_CONFORMAL_MC_PASSES = 20

DIRECTIONAL_ANCHOR_LR_MULTIPLIER = 5.0
DIRECTIONAL_ANCHOR_PRESERVATION_WEIGHT = 0.50
DIRECTIONAL_ANCHOR_GATE_PENALTY = 0.01
DIRECTIONAL_ANCHOR_MAX_TRANSFER_EPOCHS = 80
DIRECTIONAL_ANCHOR_TRANSFER_PATIENCE = 12
DIRECTIONAL_ANCHOR_MIN_ACTIVE_FOLD_FRACTION = 0.25

PDAE_CONFIDENCE_LEVEL = 0.95
PDAE_CONFIDENCE_KEEP_QUANTILE = 0.50
PDAE_MC_PASSES = 20
PDAE_MIN_COMPLETE_SAMPLES = 10
PDAE_CROSSFIT_FOLDS = 3
PDAE_INTERNAL_VALIDATION_FRACTION = 0.20
PDAE_MIN_CALIBRATION_SAMPLES_PER_TRAIT = 20
PDAE_MIN_RECONSTRUCTION_PEARSON = 0.20
PDAE_MIN_RELATIVE_MEAN_SKILL = 0.0
PDAE_MAX_SCALED_RMSE = 1.0
PDAE_AFFINE_RIDGE = 1e-3
PDAE_AFFINE_SLOPE_MIN = 0.25
PDAE_AFFINE_SLOPE_MAX = 4.0
PDAE_PSEUDO_WARMUP_EPOCHS = 3
PDAE_PSEUDO_RAMP_EPOCHS = 7
PDAE_MIN_MAIN_TRAINING_EPOCHS = 15
PDAE_GENOMIC_TEACHER_RIDGE_ALPHA = 10.0
PDAE_GENOMIC_TEACHER_MAX_MARKERS = 512
PDAE_PEARSON_BLEND_GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
PDAE_MIN_PEARSON_GAIN = 0.002
PDAE_MAX_PSEUDO_TO_OBSERVED_LOSS_RATIO = 0.10

PRIOR_SPARSITY_MODES = {
    "none": "none",
    "top_0_5pct": "top_0_5pct",
    "top_0.5pct": "top_0_5pct",
    "top0.5": "top_0_5pct",
    "top_1pct": "top_1pct",
    "top_1%": "top_1pct",
    "top1": "top_1pct",
    "top_5pct": "top_5pct",
    "top_5%": "top_5pct",
    "top5": "top_5pct",
    "p_1e-3": "p_1e-3",
    "p<1e-3": "p_1e-3",
    "p_lt_1e-3": "p_1e-3",
    "p_1e-4": "p_1e-4",
    "p<1e-4": "p_1e-4",
    "p_lt_1e-4": "p_1e-4",
}


def hidden_units_from_markers(marker_count: int) -> int:
    return max(64, min(1024, int(round(marker_count * 0.7))))


def activation_layer(name: str = "relu") -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "gelu":
        return nn.GELU()
    return nn.ReLU()


class DenseDropoutBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, activation: str = "relu") -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def dense_dropout_stack(
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    hidden_layers: int,
    activation: str = "relu",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for layer_index in range(max(1, hidden_layers)):
        layers.append(
            DenseDropoutBlock(
                input_dim if layer_index == 0 else hidden_dim,
                hidden_dim,
                dropout,
                activation=activation,
            )
        )
    return nn.Sequential(*layers)


class GenomeWideGSNet(nn.Module):
    """Genotype-only anchor used when the prior path is disabled."""

    attention_architecture = "genome_wide_anchor"

    def __init__(
        self,
        marker_count: int,
        trait_count: int,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
        hidden_layers: int = 4,
        activation: str = "relu",
        **_: object,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or hidden_units_from_markers(marker_count))
        self.encoder = dense_dropout_stack(marker_count, hidden_dim, dropout, hidden_layers, activation)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(trait_count)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        representation = self.encoder(x)
        return torch.cat([head(representation) for head in self.heads], dim=1)


class GenomicRidgeTeacher:
    """Per-trait ridge teacher used only for PDAE residual calibration."""

    def __init__(self, models: list[SklearnRidge | None], trait_count: int) -> None:
        self.models = models
        self.trait_count = int(trait_count)

    def predict_scaled(self, x: np.ndarray) -> np.ndarray:
        prediction = np.zeros((x.shape[0], self.trait_count), dtype=np.float32)
        for trait_index, model in enumerate(self.models):
            if model is not None:
                prediction[:, trait_index] = model.predict(x).astype(np.float32)
        return prediction


def _fit_ridge_per_trait(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    alpha: float = 10.0,
) -> GenomicRidgeTeacher:
    models: list[SklearnRidge | None] = []
    for trait_index in range(y.shape[1]):
        observed = mask[:, trait_index] > 0.5
        if observed.sum() < 2:
            models.append(None)
            continue
        model = SklearnRidge(alpha=float(alpha))
        model.fit(x[observed], y[observed, trait_index])
        models.append(model)
    return GenomicRidgeTeacher(models, y.shape[1])


class PhenotypeDenoisingAutoencoder(nn.Module):
    def __init__(self, trait_count: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(16, min(128, trait_count * 16))
        self.trait_count = trait_count
        self.net = nn.Sequential(
            nn.Linear(trait_count * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, trait_count),
        )
        self.observed_skip_logit = nn.Parameter(torch.tensor(-1.2))

    def forward(self, y_values: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        observed_mask = observed_mask.float()
        masked_values = y_values * observed_mask
        reconstructed = self.net(torch.cat([masked_values, observed_mask], dim=-1))
        return reconstructed + torch.sigmoid(self.observed_skip_logit) * masked_values


class TraitSelfRefinement(nn.Module):
    """Refine each trait representation without borrowing from other traits."""

    def __init__(
        self,
        hidden_dim: int,
        trait_count: int,
        dropout: float,
        activation: str,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.trait_count = int(trait_count)
        if hidden_dim % heads != 0:
            heads = 1
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, trait_representations: torch.Tensor) -> torch.Tensor:
        if self.trait_count <= 1:
            return trait_representations
        identity = torch.eye(
            self.trait_count,
            device=trait_representations.device,
            dtype=trait_representations.dtype,
        )
        attention_bias = torch.log(identity.clamp_min(1e-6))
        attended, _ = self.attention(
            trait_representations,
            trait_representations,
            trait_representations,
            attn_mask=attention_bias,
            need_weights=False,
        )
        refined = self.norm1(trait_representations + self.dropout(attended))
        updated = self.feed_forward(refined)
        return self.norm2(refined + self.dropout(updated))


class PriorMarkerAttentionGSNet(nn.Module):
    """Genome-wide anchor with a trait-specific prior-gated auxiliary path."""

    attention_architecture = "trait_specific_prior_informed_anchor_final"

    def __init__(
        self,
        marker_count: int,
        trait_count: int,
        prior_scores: np.ndarray | torch.Tensor | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
        hidden_layers: int = 4,
        use_prior_reliability_gate: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.marker_count = int(marker_count)
        self.trait_count = int(trait_count)
        hidden_dim = int(hidden_dim or hidden_units_from_markers(marker_count))
        attention_dim = max(32, min(128, hidden_dim // 4))

        self.marker_norm = nn.LayerNorm(marker_count)
        self.marker_gate_logits = nn.Parameter(torch.zeros(marker_count))
        self.trait_marker_gate_logits = nn.Parameter(torch.zeros(trait_count, marker_count))
        self.prior_strength_raw = nn.Parameter(torch.zeros(trait_count))
        self.use_prior_reliability_gate = bool(use_prior_reliability_gate)
        if self.use_prior_reliability_gate:
            self.prior_reliability_logit = nn.Parameter(torch.full((trait_count,), -1.0))
        else:
            self.register_parameter("prior_reliability_logit", None)

        values = torch.zeros(trait_count, marker_count, dtype=torch.float32)
        if prior_scores is not None:
            values = torch.as_tensor(prior_scores, dtype=torch.float32)
            if values.ndim == 1:
                values = values[None, :].expand(trait_count, -1).clone()
            elif values.shape == (marker_count, trait_count):
                values = values.T.contiguous()
            if values.shape != (trait_count, marker_count):
                raise ValueError("prior_scores must align with traits and markers.")
        self.register_buffer("prior_scores", values)

        self.attention_input_dropout = nn.Dropout(min(0.2, dropout))
        self.main_encoder = dense_dropout_stack(marker_count, hidden_dim, dropout, hidden_layers, activation)
        self.attention_encoder = nn.Sequential(
            DenseDropoutBlock(marker_count, attention_dim, dropout, activation),
            DenseDropoutBlock(attention_dim, attention_dim, dropout, activation),
            nn.Linear(attention_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
        )
        self.attention_fusion_logit = nn.Parameter(torch.tensor(-4.0))
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        trait_dim = min(32, hidden_dim)
        self.trait_embedding = nn.Embedding(trait_count, trait_dim)
        self.trait_conditioning_layer = nn.Sequential(
            nn.Linear(hidden_dim + trait_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
        )
        self.trait_self_refinement = TraitSelfRefinement(
            hidden_dim=hidden_dim,
            trait_count=trait_count,
            dropout=dropout,
            activation=activation,
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(trait_count)])
        self.attention_runtime_scale = 1.0
        self.prior_dropout_rate = 0.0

    def configure_transfer_mode(self, mode: str) -> None:
        if str(mode).strip().lower() != "none":
            raise ValueError("A prior-aware anchor has no cross-trait transfer stage.")

    def attention_fusion_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.attention_fusion_logit)

    def prior_strength(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.prior_strength_raw)

    def prior_reliability(self) -> torch.Tensor:
        if self.prior_reliability_logit is None:
            return torch.ones_like(self.prior_strength_raw)
        return torch.sigmoid(self.prior_reliability_logit)

    def effective_prior_strength(self) -> torch.Tensor:
        return self.prior_strength() * self.prior_reliability()

    def set_attention_runtime_scale(self, scale: float) -> None:
        self.attention_runtime_scale = float(np.clip(scale, 0.0, 1.0))

    def set_prior_dropout_rate(self, rate: float) -> None:
        self.prior_dropout_rate = float(np.clip(rate, 0.0, 0.8))

    def trait_marker_gates(self) -> torch.Tensor:
        trait_delta = 0.25 * torch.tanh(self.trait_marker_gate_logits)
        prior_scores = self.prior_scores
        if self.training and self.prior_dropout_rate > 0:
            keep_probability = 1.0 - self.prior_dropout_rate
            prior_scores = prior_scores * (
                (torch.rand_like(prior_scores) < keep_probability).float()
                / max(keep_probability, 1e-6)
            )
        logits = (
            self.marker_gate_logits[None, :]
            + trait_delta
            + self.effective_prior_strength()[:, None] * prior_scores
        )
        return 1.0 + 0.5 * torch.tanh(logits)

    def trait_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        deviations = torch.abs(self.trait_marker_gates() - 1.0)
        denominators = deviations.sum(dim=1, keepdim=True)
        uniform = torch.full_like(deviations, 1.0 / max(self.marker_count, 1))
        weights = torch.where(
            denominators <= 1e-8,
            uniform,
            deviations / denominators.clamp_min(1e-8),
        )
        return weights[None, :, :].expand(x.shape[0], -1, -1)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.trait_attention_weights(x).mean(dim=1)

    def attention_regularization(self) -> torch.Tensor:
        gates = self.trait_marker_gates()
        prior_weight = 1.0 / (1.0 + torch.clamp(self.prior_scores, min=0.0))
        gate_penalty = torch.mean(((gates - 1.0) ** 2) * prior_weight)
        strength_penalty = torch.mean(self.prior_strength() ** 2)
        reliability_penalty = (
            torch.mean(self.prior_reliability() ** 2)
            if self.prior_reliability_logit is not None
            else gates.new_tensor(0.0)
        )
        return (
            gate_penalty
            + 0.01 * strength_penalty
            + 0.01 * reliability_penalty
            + 0.05 * self.attention_fusion_alpha() ** 2
        )

    def base_prediction_representations(
        self,
        x: torch.Tensor,
        attention_scale: float | None = None,
    ) -> torch.Tensor:
        scale = self.attention_runtime_scale if attention_scale is None else float(attention_scale)
        normalized = self.marker_norm(x)
        main_representation = self.main_encoder(normalized)
        alpha = scale * self.attention_fusion_alpha()
        gates = self.trait_marker_gates()
        trait_ids = torch.arange(self.trait_count, device=x.device)
        embeddings = self.trait_embedding(trait_ids)
        representations = []
        for trait_index in range(self.trait_count):
            prior_representation = self.attention_encoder(
                self.attention_input_dropout(normalized * gates[trait_index])
            )
            fused = self.fusion_norm(main_representation + alpha * prior_representation)
            trait_embedding = embeddings[trait_index].expand(x.shape[0], -1)
            representations.append(
                self.trait_conditioning_layer(torch.cat([fused, trait_embedding], dim=1))
            )
        return self.trait_self_refinement(torch.stack(representations, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        representations = self.base_prediction_representations(x)
        return torch.cat(
            [head(representations[:, index, :]) for index, head in enumerate(self.heads)],
            dim=1,
        )


class DirectionalAnchorGSNet(nn.Module):
    """Frozen single-trait anchors with bounded asymmetric residual transfer."""

    MODE = DIRECTIONAL_ANCHOR_MODE
    RANK = 8
    MAX_GATE = 0.25
    INITIAL_GATE = 0.10
    attention_architecture = "directional_single_trait_anchor_residual_transfer_final"

    def __init__(
        self,
        marker_count: int,
        trait_count: int,
        prior_scores: np.ndarray | torch.Tensor | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
        hidden_layers: int = 4,
        activation: str = "relu",
        use_prior_reliability_gate: bool = False,
        anchor_models: list[PriorMarkerAttentionGSNet] | None = None,
    ) -> None:
        super().__init__()
        if trait_count < 2:
            raise ValueError("directional_anchor requires at least two traits.")
        self.marker_count = int(marker_count)
        self.trait_count = int(trait_count)
        self.hidden_dim = int(hidden_dim or hidden_units_from_markers(marker_count))
        self.rank = min(self.RANK, self.hidden_dim)
        self.transfer_mode = self.MODE
        self.attention_runtime_scale = 1.0
        values = self._normalize_prior_scores(prior_scores)
        self.register_buffer("prior_scores", values)

        if anchor_models is None:
            anchor_models = [
                PriorMarkerAttentionGSNet(
                    marker_count=self.marker_count,
                    trait_count=1,
                    prior_scores=values[index : index + 1],
                    hidden_dim=self.hidden_dim,
                    dropout=dropout,
                    hidden_layers=hidden_layers,
                    activation=activation,
                    use_prior_reliability_gate=use_prior_reliability_gate,
                )
                for index in range(self.trait_count)
            ]
        if len(anchor_models) != self.trait_count:
            raise ValueError("One independently trained anchor is required per trait.")
        self.anchor_models = nn.ModuleList(anchor_models)

        self.directional_adapters = nn.ModuleDict()
        for source_index in range(self.trait_count):
            for target_index in range(self.trait_count):
                if source_index == target_index:
                    continue
                adapter = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.rank, bias=False),
                    nn.GELU(),
                    nn.Linear(self.rank, 1, bias=False),
                    nn.Tanh(),
                )
                nn.init.zeros_(adapter[3].weight)
                self.directional_adapters[f"{source_index}_to_{target_index}"] = adapter

        probability = self.INITIAL_GATE / self.MAX_GATE
        initial_logit = float(np.log(probability / (1.0 - probability)))
        logits = torch.full((self.trait_count, self.trait_count), initial_logit)
        logits.fill_diagonal_(-20.0)
        self.directional_gate_logits = nn.Parameter(logits)
        self.register_buffer(
            "directional_off_diagonal",
            1.0 - torch.eye(self.trait_count, dtype=torch.float32),
        )
        self.freeze_anchors()

    def _normalize_prior_scores(self, prior_scores: np.ndarray | torch.Tensor | None) -> torch.Tensor:
        if prior_scores is None:
            return torch.zeros(self.trait_count, self.marker_count, dtype=torch.float32)
        values = torch.as_tensor(prior_scores, dtype=torch.float32)
        if values.ndim == 1:
            values = values[None, :].expand(self.trait_count, -1).clone()
        elif values.shape == (self.marker_count, self.trait_count):
            values = values.T.contiguous()
        if values.shape != (self.trait_count, self.marker_count):
            raise ValueError("prior_scores must align with traits and markers.")
        return values.clone()

    def freeze_anchors(self) -> None:
        for anchor in self.anchor_models:
            for parameter in anchor.parameters():
                parameter.requires_grad_(False)
            anchor.eval()

    def set_anchors_eval(self) -> None:
        for anchor in self.anchor_models:
            anchor.eval()

    def configure_transfer_mode(self, mode: str) -> None:
        if str(mode).strip().lower() != self.MODE:
            raise ValueError(f"Only {self.MODE} is valid for this model.")
        self.freeze_anchors()

    def directional_parameters(self) -> list[nn.Parameter]:
        values = list(self.directional_adapters.parameters()) + [self.directional_gate_logits]
        return [parameter for parameter in values if parameter.requires_grad]

    def directional_gates(self) -> torch.Tensor:
        return (
            self.MAX_GATE
            * torch.sigmoid(self.directional_gate_logits)
            * self.directional_off_diagonal
        )

    def set_attention_runtime_scale(self, scale: float) -> None:
        self.attention_runtime_scale = float(np.clip(scale, 0.0, 1.0))
        for anchor in self.anchor_models:
            anchor.set_attention_runtime_scale(self.attention_runtime_scale)

    def set_prior_dropout_rate(self, rate: float) -> None:
        for anchor in self.anchor_models:
            anchor.set_prior_dropout_rate(rate)

    def prior_strength(self) -> torch.Tensor:
        return torch.cat([anchor.prior_strength().reshape(-1) for anchor in self.anchor_models])

    def prior_reliability(self) -> torch.Tensor:
        return torch.cat([anchor.prior_reliability().reshape(-1) for anchor in self.anchor_models])

    def effective_prior_strength(self) -> torch.Tensor:
        return torch.cat([anchor.effective_prior_strength().reshape(-1) for anchor in self.anchor_models])

    def trait_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [anchor.trait_attention_weights(x)[:, 0, :] for anchor in self.anchor_models],
            dim=1,
        )

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.trait_attention_weights(x).mean(dim=1)

    def anchor_components(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        predictions = []
        representations = []
        for anchor in self.anchor_models:
            representation = anchor.base_prediction_representations(
                x,
                attention_scale=self.attention_runtime_scale,
            )[:, 0, :]
            predictions.append(anchor.heads[0](representation))
            representations.append(representation)
        return torch.cat(predictions, dim=1), representations

    def directional_corrections(
        self,
        representations: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gates = self.directional_gates()
        corrections = []
        contributions: dict[str, torch.Tensor] = {}
        source_count = max(self.trait_count - 1, 1)
        for target_index in range(self.trait_count):
            correction = representations[target_index].new_zeros(
                (representations[target_index].shape[0], 1)
            )
            for source_index in range(self.trait_count):
                if source_index == target_index:
                    continue
                key = f"{source_index}_to_{target_index}"
                contribution = gates[target_index, source_index] * self.directional_adapters[key](
                    representations[source_index]
                )
                contributions[key] = contribution
                correction = correction + contribution
            corrections.append(correction / source_count)
        return torch.cat(corrections, dim=1), contributions

    def forward_components(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        anchor_predictions, representations = self.anchor_components(x)
        corrections, contributions = self.directional_corrections(representations)
        return anchor_predictions + corrections, anchor_predictions, corrections, contributions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x)[0]

    @torch.no_grad()
    def collect_directional_diagnostics(self, x: torch.Tensor) -> dict[str, object]:
        final_predictions, anchor_predictions, corrections, contributions = self.forward_components(x)
        gates = self.directional_gates()
        summary: dict[str, object] = {}
        for target_index in range(self.trait_count):
            anchor_scale = anchor_predictions[:, target_index].std(unbiased=False).clamp_min(1e-6)
            sources = []
            for source_index in range(self.trait_count):
                if source_index == target_index:
                    continue
                key = f"{source_index}_to_{target_index}"
                values = contributions[key].squeeze(-1)
                sources.append(
                    {
                        "source_index": int(source_index),
                        "gate": float(gates[target_index, source_index].cpu()),
                        "contribution_rms": float(values.square().mean().sqrt().cpu()),
                    }
                )
            correction = corrections[:, target_index]
            summary[str(target_index)] = {
                "sources": sources,
                "anchor_prediction_std": float(anchor_scale.cpu()),
                "correction_rms": float(correction.square().mean().sqrt().cpu()),
                "correction_to_anchor_std_ratio": float(
                    correction.square().mean().sqrt().cpu() / anchor_scale.cpu()
                ),
                "mean_absolute_prediction_change": float(
                    torch.mean(
                        torch.abs(
                            final_predictions[:, target_index]
                            - anchor_predictions[:, target_index]
                        )
                    ).cpu()
                ),
            }
        return summary


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    observed = mask > 0.5
    return ((pred - target)[observed] ** 2).mean()


def masked_trait_balanced_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    squared_error = (pred - target) ** 2
    observed = mask.sum(dim=0)
    active = observed > 0
    losses = (squared_error * mask).sum(dim=0) / observed.clamp_min(1.0)
    return losses[active].mean()


def observed_correlation_matrix(y: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    trait_count = y.shape[1]
    if trait_count < 2:
        return None
    correlation = np.eye(trait_count, dtype=np.float32)
    for first in range(trait_count):
        for second in range(first + 1, trait_count):
            observed = (mask[:, first] > 0.5) & (mask[:, second] > 0.5)
            if observed.sum() < 3:
                value = 0.0
            else:
                value = float(np.corrcoef(y[observed, first], y[observed, second])[0, 1])
                if not np.isfinite(value):
                    value = 0.0
            correlation[first, second] = value
            correlation[second, first] = value
    return correlation


def _normalize_attention_mode(attention_mode: str | None) -> str:
    normalized = str(attention_mode or "prior_marker_pearson").strip().lower()
    if normalized not in {"none", "prior_marker_pearson"}:
        raise ValueError("Only 'none' and 'prior_marker_pearson' are supported.")
    return normalized


def _normalize_transfer_mode(mode: str | None, use_directional_transfer: bool = True) -> str:
    normalized = str(mode or (DIRECTIONAL_ANCHOR_MODE if use_directional_transfer else "none")).strip().lower().replace("-", "_")
    if normalized not in {"none", DIRECTIONAL_ANCHOR_MODE}:
        raise ValueError("Only 'none' and 'directional_anchor' are supported.")
    return normalized


def _build_ppmgs_model(job: TrainedJob) -> torch.nn.Module:
    params = _resolve_ppmgs_params(len(job.marker_names), job.hyperparameters)
    kwargs = {
        "marker_count": len(job.marker_names),
        "trait_count": len(job.trait_names),
        "hidden_dim": int(params["hidden_dim"]),
        "dropout": float(params["dropout"]),
        "hidden_layers": int(params["hidden_layers"]),
        "activation": str(params.get("activation", "relu")),
    }
    attention_mode = _normalize_attention_mode(job.attention_mode)
    if job.transfer_mode == DIRECTIONAL_ANCHOR_MODE:
        return DirectionalAnchorGSNet(
            **kwargs,
            prior_scores=job.prior_scores,
            use_prior_reliability_gate=attention_mode in PRIOR_RELIABILITY_ATTENTION_MODES,
        )
    if attention_mode in PRIOR_MARKER_ATTENTION_MODES:
        return PriorMarkerAttentionGSNet(
            **kwargs,
            prior_scores=job.prior_scores,
            use_prior_reliability_gate=attention_mode in PRIOR_RELIABILITY_ATTENTION_MODES,
        )
    return GenomeWideGSNet(**kwargs)


def _set_transfer_mode(model: torch.nn.Module, mode: str) -> None:
    normalized = _normalize_transfer_mode(mode, use_directional_transfer=mode != "none")
    if isinstance(model, DirectionalAnchorGSNet):
        model.configure_transfer_mode(normalized)
    elif normalized != "none":
        raise ValueError("directional_anchor requires DirectionalAnchorGSNet.")


def _directional_transfer_summary(
    model: object,
    trait_names: list[str],
    observed_correlation: np.ndarray | None = None,
) -> dict[str, object]:
    if isinstance(model, DirectionalAnchorGSNet):
        return {
            "mode": DIRECTIONAL_ANCHOR_MODE,
            "traits": list(trait_names),
            "stage_training": getattr(model, "directional_anchor_training_summary", None),
            "diagnostics": getattr(model, "directional_anchor_diagnostics", None),
            "observed_correlation": _array_to_list(observed_correlation),
        }
    return {
        "mode": "none",
        "traits": list(trait_names),
        "description": "trait-specific anchor only",
        "observed_correlation": _array_to_list(observed_correlation),
    }


def save_job(job_id: str, job: TrainedJob) -> None:
    job_dir = SAVED_MODELS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "job_id": job_id,
        "samples": job.samples,
        "marker_names": job.marker_names,
        "trait_names": job.trait_names,
        "x_mean": _array_to_list(job.x_mean),
        "x_std": _array_to_list(job.x_std),
        "y_mean": _array_to_list(job.y_mean),
        "y_std": _array_to_list(job.y_std),
        "final_loss": job.final_loss,
        "best_epoch": job.best_epoch,
        "task_type": job.task_type,
        "allow_missing_phenotype": job.allow_missing_phenotype,
        "use_directional_transfer": job.use_directional_transfer,
        "transfer_mode": job.transfer_mode,
        "attention_mode": job.attention_mode,
        "attention_architecture": getattr(job.model, "attention_architecture", None),
        "imputation_strategy": job.imputation_strategy,
        "marker_fill_values": _array_to_list(job.marker_fill_values),
        "prior_scores": _array_to_list(job.prior_scores) if job.prior_scores is not None else None,
        "prior_marker_summary": job.prior_marker_summary,
        "shap_top_markers": job.shap_top_markers,
        "shap_summary": job.shap_summary,
        "metrics": job.metrics,
        "cross_validation": job.cross_validation,
        "prediction_interval_calibration": job.prediction_interval_calibration,
        "uncertainty_metadata": job.uncertainty_metadata,
        "hyperparameters": job.hyperparameters,
        "hyperparameter_search": job.hyperparameter_search,
        "pdae_summary": job.pdae_summary,
        "phenotype_missing_summary": job.phenotype_missing_summary,
        "attention_safety": job.attention_safety,
        "directional_transfer_summary": job.directional_transfer_summary,
        "timing_summary": job.timing_summary,
    }
    (job_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    torch.save(job.model.state_dict(), job_dir / "model.pt")


def load_job(job_id: str) -> TrainedJob:
    job_dir = SAVED_MODELS_DIR / job_id
    metadata_path = job_dir / "metadata.json"
    if not metadata_path.exists():
        raise KeyError(f"Unknown job_id: {job_id}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    job = TrainedJob(
        model=None,
        samples=int(metadata["samples"]),
        marker_names=list(metadata["marker_names"]),
        trait_names=list(metadata["trait_names"]),
        x_mean=np.asarray(metadata["x_mean"], dtype=np.float32),
        x_std=np.asarray(metadata["x_std"], dtype=np.float32),
        y_mean=np.asarray(metadata["y_mean"], dtype=np.float32),
        y_std=np.asarray(metadata["y_std"], dtype=np.float32),
        final_loss=float(metadata["final_loss"]),
        best_epoch=int(metadata["best_epoch"]),
        task_type=str(metadata["task_type"]),
        allow_missing_phenotype=bool(metadata["allow_missing_phenotype"]),
        use_directional_transfer=bool(metadata.get("use_directional_transfer", True)),
        transfer_mode=_normalize_transfer_mode(
            metadata.get("transfer_mode"),
            bool(metadata.get("use_directional_transfer", True)),
        ),
        attention_mode=_normalize_attention_mode(metadata.get("attention_mode")),
        imputation_strategy=str(metadata["imputation_strategy"]),
        marker_fill_values=np.asarray(metadata["marker_fill_values"], dtype=np.float32),
        prior_scores=(
            np.asarray(metadata["prior_scores"], dtype=np.float32)
            if metadata.get("prior_scores") is not None
            else None
        ),
        prior_marker_summary=metadata.get("prior_marker_summary"),
        shap_top_markers=dict(metadata.get("shap_top_markers") or {}),
        shap_summary=metadata.get("shap_summary"),
        metrics=dict(metadata.get("metrics") or {}),
        cross_validation=metadata.get("cross_validation"),
        prediction_interval_calibration=metadata.get("prediction_interval_calibration"),
        uncertainty_metadata=metadata.get("uncertainty_metadata"),
        hyperparameters=dict(metadata.get("hyperparameters") or {}),
        hyperparameter_search=metadata.get("hyperparameter_search"),
        pdae_summary=metadata.get("pdae_summary"),
        phenotype_missing_summary=metadata.get("phenotype_missing_summary"),
        attention_safety=metadata.get("attention_safety"),
        directional_transfer_summary=metadata.get("directional_transfer_summary"),
        timing_summary=metadata.get("timing_summary"),
    )
    model = _build_ppmgs_model(job)
    state = torch.load(job_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    _set_transfer_mode(model, job.transfer_mode)
    model.to(get_torch_device())
    model.eval()
    job.model = model
    JOB_STORE[job_id] = job
    return job


def configure_training_seed(seed: int) -> int:
    """Set the process-wide seed used by CV splits and model fitting."""
    global TRAINING_BASE_SEED
    resolved = int(seed)
    if resolved < 0:
        raise ValueError('Training seed must be non-negative.')
    TRAINING_BASE_SEED = resolved
    np.random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    return resolved


def _training_seed(offset: int=0) -> int:
    return int((TRAINING_BASE_SEED + int(offset)) % (2 ** 31 - 1))


@dataclass
class TrainedJob:
    model: object
    samples: int
    marker_names: list[str]
    trait_names: list[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    final_loss: float
    best_epoch: int
    task_type: str
    allow_missing_phenotype: bool
    use_directional_transfer: bool
    transfer_mode: str
    attention_mode: str
    imputation_strategy: str
    marker_fill_values: np.ndarray
    prior_scores: np.ndarray | None = None
    prior_marker_summary: dict[str, object] | None = None
    shap_top_markers: dict[str, list[dict[str, float | str | int | None]]] = field(default_factory=dict)
    shap_summary: dict[str, object] | None = None
    metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)
    cross_validation: dict[str, object] | None = None
    prediction_interval_calibration: dict[str, object] | None = None
    uncertainty_metadata: dict[str, object] | None = None
    hyperparameters: dict[str, object] = field(default_factory=dict)
    hyperparameter_search: dict[str, object] | None = None
    pdae_summary: dict[str, object] | None = None
    phenotype_missing_summary: dict[str, object] | None = None
    attention_safety: dict[str, object] | None = None
    directional_transfer_summary: dict[str, object] | None = None
    timing_summary: dict[str, object] | None = None


def get_torch_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _array_to_list(array: np.ndarray):
    return np.asarray(array, dtype=float).tolist()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def _normalize_prior_sparsity(mode: str | None) -> str:
    key = str(mode or 'none').strip().lower().replace(' ', '_')
    return PRIOR_SPARSITY_MODES.get(key, 'none')


def _prior_sparsity_spec(mode: str) -> dict[str, object]:
    mode = _normalize_prior_sparsity(mode)
    if mode == 'top_0_5pct':
        return {'mode': mode, 'kind': 'top_fraction', 'fraction': 0.005}
    if mode == 'top_1pct':
        return {'mode': mode, 'kind': 'top_fraction', 'fraction': 0.01}
    if mode == 'top_5pct':
        return {'mode': mode, 'kind': 'top_fraction', 'fraction': 0.05}
    if mode == 'p_1e-3':
        return {'mode': mode, 'kind': 'min_score', 'min_score': 3.0, 'pvalue_threshold': 0.001}
    if mode == 'p_1e-4':
        return {'mode': mode, 'kind': 'min_score', 'min_score': 4.0, 'pvalue_threshold': 0.0001}
    return {'mode': 'none', 'kind': 'none'}


def _apply_prior_sparsity(scores: np.ndarray, mode: str | None) -> tuple[np.ndarray, dict[str, object]]:
    mode = _normalize_prior_sparsity(mode)
    values = np.asarray(scores, dtype=np.float32)
    sparse = np.zeros_like(values, dtype=np.float32)
    before = [int(np.sum(row > 0)) for row in values]
    spec = _prior_sparsity_spec(mode)
    if mode == 'none':
        sparse = values.copy()
    elif spec['kind'] == 'top_fraction':
        fraction = float(spec['fraction'])
        keep_count = max(1, int(np.ceil(values.shape[1] * fraction))) if values.shape[1] else 0
        for trait_idx, row in enumerate(values):
            positive_idx = np.flatnonzero(row > 0)
            if positive_idx.size == 0:
                continue
            selected = positive_idx[np.argsort(row[positive_idx])[::-1][:min(keep_count, positive_idx.size)]]
            sparse[trait_idx, selected] = row[selected]
    elif spec['kind'] == 'min_score':
        threshold = float(spec['min_score'])
        sparse = np.where(values >= threshold, values, 0.0).astype(np.float32)
    after = [int(np.sum(row > 0)) for row in sparse]
    summary = {'mode': mode, 'spec': spec, 'nonzero_before_by_trait': before, 'nonzero_after_by_trait': after, 'nonzero_removed_by_trait': [int(max(0, b - a)) for b, a in zip(before, after)], 'applied': mode != 'none'}
    return (sparse.astype(np.float32), summary)


def _apply_prior_sparsity_with_gwas_mask(scores: np.ndarray, mode: str | None, gwas_prior_scores: np.ndarray | None=None) -> tuple[np.ndarray, dict[str, object]]:
    mode = _normalize_prior_sparsity(mode)
    if mode in {'p_1e-3', 'p_1e-4'} and gwas_prior_scores is not None:
        values = np.asarray(scores, dtype=np.float32)
        gwas_mask = np.asarray(gwas_prior_scores, dtype=np.float32) > 0
        sparse = np.where(gwas_mask, values, 0.0).astype(np.float32)
        before = [int(np.sum(row > 0)) for row in values]
        after = [int(np.sum(row > 0)) for row in sparse]
        return (sparse, {'mode': mode, 'spec': _prior_sparsity_spec(mode), 'applied': True, 'method': 'keep_only_gwas_thresholded_markers_before_lasso_fusion', 'nonzero_before_by_trait': before, 'nonzero_after_by_trait': after, 'nonzero_removed_by_trait': [int(max(0, b - a)) for b, a in zip(before, after)]})
    return _apply_prior_sparsity(scores, mode)


def _read_csv(file_obj) -> pd.DataFrame:
    return pd.read_csv(file_obj, na_values=NA_VALUES, keep_default_na=True)


def marker_mode_impute(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marker-wise mode imputation for binary/discrete SNP genotypes."""
    filled = x.copy()
    fill_values = np.zeros(x.shape[1], dtype=np.float32)
    for col_idx in range(x.shape[1]):
        column = x[:, col_idx]
        observed = column[np.isfinite(column)]
        if observed.size == 0:
            fill_value = 0.0
        else:
            values, counts = np.unique(observed, return_counts=True)
            fill_value = float(values[np.argmax(counts)])
        fill_values[col_idx] = fill_value
        filled[~np.isfinite(filled[:, col_idx]), col_idx] = fill_value
    return (filled, fill_values)


def _phenotype_missing_summary(y_mask: np.ndarray, trait_names: list[str], dropped_all_missing_samples: int=0, retained_all_missing_samples: int=0) -> dict[str, object]:
    mask = np.asarray(y_mask, dtype=np.float32)
    sample_count = int(mask.shape[0])
    trait_count = int(mask.shape[1]) if mask.ndim == 2 else 0
    per_trait: dict[str, object] = {}
    for trait_idx, trait in enumerate(trait_names):
        observed_count = int(np.sum(mask[:, trait_idx] > 0))
        missing_count = int(sample_count - observed_count)
        per_trait[trait] = {'observed_count': observed_count, 'missing_count': missing_count, 'missing_rate': float(missing_count / max(sample_count, 1))}
    pattern_counts: dict[str, int] = {}
    observed_trait_count_distribution: dict[str, int] = {}
    for row in mask:
        pattern = ''.join(('1' if value > 0 else '0' for value in row))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        observed_count = int(np.sum(row > 0))
        key = str(observed_count)
        observed_trait_count_distribution[key] = observed_trait_count_distribution.get(key, 0) + 1
    return {'sample_count': sample_count, 'trait_count': trait_count, 'traits': per_trait, 'missing_patterns': dict(sorted(pattern_counts.items())), 'observed_trait_count_distribution': dict(sorted(observed_trait_count_distribution.items(), key=lambda item: int(item[0]))), 'dropped_all_missing_samples': int(dropped_all_missing_samples), 'retained_all_missing_samples': int(retained_all_missing_samples)}


def _read_prior_marker_file(prior_marker_file, marker_names: list[str], trait_names: list[str], prior_sparsity: str | None='none') -> tuple[np.ndarray, dict[str, object]]:
    if prior_marker_file is None:
        raise ValueError('Prior-informed marker attention requires an SNP-Marker file.')
    prior_df = _read_csv(prior_marker_file)
    if prior_df.empty:
        raise ValueError('SNP-Marker file is empty.')
    normalized_columns = {str(column).strip().lower(): column for column in prior_df.columns}
    marker_col = normalized_columns.get('marker') or normalized_columns.get('snp') or normalized_columns.get('snp_marker')
    if marker_col is None:
        marker_col = prior_df.columns[0]
    trait_col = normalized_columns.get('trait') or normalized_columns.get('phenotype') or normalized_columns.get('trait_name')
    score_col = normalized_columns.get('score') or normalized_columns.get('-log10p') or normalized_columns.get('logp') or normalized_columns.get('pvalue') or normalized_columns.get('p_value') or normalized_columns.get('p')
    trait_lookup = {str(trait).strip().lower(): idx for idx, trait in enumerate(trait_names)}
    wide_trait_cols: dict[int, str] = {}
    for trait_idx, trait in enumerate(trait_names):
        col = normalized_columns.get(str(trait).strip().lower())
        if col is not None and col != marker_col:
            wide_trait_cols[trait_idx] = col
    prior_format = 'long_trait_score' if trait_col is not None else 'wide_trait_score' if wide_trait_cols else 'generic_score'
    if prior_format == 'generic_score' and score_col is None and (len(prior_df.columns) >= 2):
        score_col = prior_df.columns[1]
    marker_to_idx = {name: idx for idx, name in enumerate(marker_names)}
    raw_scores = np.zeros((len(trait_names), len(marker_names)), dtype=np.float32)
    provided = 0
    unmatched_rows = 0
    unknown_trait_rows = 0
    matched: set[str] = set()
    matched_by_trait: list[set[str]] = [set() for _ in trait_names]
    missing: list[str] = []
    unknown_traits: list[str] = []

    def score_from_cell(value, column, default: float) -> float:
        score = pd.to_numeric(value, errors='coerce')
        if pd.isna(score) or not np.isfinite(float(score)):
            return default
        score = float(score)
        column_name = str(column).strip().lower()
        if column_name in {'pvalue', 'p_value', 'p'}:
            score = -float(np.log10(score)) if score > 0 else default
        return max(0.0, score)

    def remember_missing(marker: str) -> None:
        if len(missing) < 20:
            missing.append(marker)
    for _, row in prior_df.iterrows():
        marker = str(row[marker_col]).strip()
        if not marker or marker.lower() in {'nan', 'none'}:
            continue
        provided += 1
        idx = marker_to_idx.get(marker)
        if idx is None:
            unmatched_rows += 1
            remember_missing(marker)
            continue
        if prior_format == 'long_trait_score':
            trait_name = str(row[trait_col]).strip()
            trait_idx = trait_lookup.get(trait_name.lower())
            if trait_idx is None:
                unknown_trait_rows += 1
                if len(unknown_traits) < 20:
                    unknown_traits.append(trait_name)
                continue
            score = score_from_cell(row[score_col], score_col, default=1.0) if score_col is not None else 1.0
            raw_scores[trait_idx, idx] = max(raw_scores[trait_idx, idx], score)
            matched_by_trait[trait_idx].add(marker)
        elif prior_format == 'wide_trait_score':
            for trait_idx, col in wide_trait_cols.items():
                score = score_from_cell(row[col], col, default=0.0)
                raw_scores[trait_idx, idx] = max(raw_scores[trait_idx, idx], score)
                if score > 0:
                    matched_by_trait[trait_idx].add(marker)
        else:
            score = score_from_cell(row[score_col], score_col, default=1.0) if score_col is not None else 1.0
            raw_scores[:, idx] = np.maximum(raw_scores[:, idx], score)
            for trait_idx in range(len(trait_names)):
                matched_by_trait[trait_idx].add(marker)
        matched.add(marker)
    if not matched:
        raise ValueError('No markers in the SNP-Marker file matched genotype.csv columns.')
    sparse_raw_scores, sparsity_summary = _apply_prior_sparsity(raw_scores, prior_sparsity)
    scaled = _soft_prior_scale(sparse_raw_scores)
    positive_all = raw_scores[raw_scores > 0]
    summary = {'provided': int(provided), 'matched': int(len(matched)), 'matched_rows': int(max(provided - unmatched_rows - unknown_trait_rows, 0)), 'missing': int(unmatched_rows), 'unknown_trait_rows': int(unknown_trait_rows), 'missing_examples': missing, 'format': prior_format, 'score_column': str(score_col) if score_col is not None else None, 'trait_column': str(trait_col) if trait_col is not None else None, 'trait_score_columns': {trait_names[idx]: str(col) for idx, col in wide_trait_cols.items()}, 'unknown_trait_examples': unknown_traits, 'marker_column': str(marker_col), 'matched_by_trait': {trait: int(len(matched_by_trait[idx])) for idx, trait in enumerate(trait_names)}, 'score_transform': 'positive_scores_standardized_per_trait_to_soft_prior', 'prior_sparsity': sparsity_summary, 'nonzero_after_sparsity_by_trait': {trait: int(np.sum(sparse_raw_scores[idx] > 0)) for idx, trait in enumerate(trait_names)}, 'raw_score_min': float(positive_all.min()) if positive_all.size else None, 'raw_score_max': float(positive_all.max()) if positive_all.size else None}
    return (scaled.astype(np.float32), summary)


def _minmax_by_trait(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    normalized = np.zeros_like(scores, dtype=np.float32)
    for trait_idx in range(scores.shape[0]):
        row = scores[trait_idx]
        positive = row[row > 0]
        if positive.size == 0:
            continue
        min_value = float(positive.min())
        max_value = float(positive.max())
        if max_value <= min_value + 1e-08:
            normalized[trait_idx, row > 0] = 1.0
        else:
            normalized[trait_idx, row > 0] = (row[row > 0] - min_value) / (max_value - min_value)
    return normalized


def _soft_prior_scale(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    scaled = np.zeros_like(scores, dtype=np.float32)
    for trait_idx in range(scores.shape[0]):
        row = scores[trait_idx]
        positive = row[row > 0]
        if positive.size > 1:
            mean = float(positive.mean())
            std = float(positive.std())
            if std < 1e-08:
                scaled[trait_idx, row > 0] = 1.0
            else:
                scaled_values = (positive - mean) / std + 1.0
                scaled[trait_idx, row > 0] = np.clip(scaled_values, 0.0, 5.0)
        elif positive.size == 1:
            scaled[trait_idx, row > 0] = 1.0
    return scaled


def _auto_lasso_alpha(sample_count: int, marker_count: int) -> float:
    sample_count = max(3, int(sample_count))
    marker_count = max(2, int(marker_count))
    return float(np.clip(0.05 * np.sqrt(np.log(marker_count) / sample_count), 0.003, 0.08))


def _sha256_array(values: np.ndarray, chunk_rows: int=64) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(json.dumps(list(array.shape)).encode('ascii'))
    if array.ndim == 0:
        digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()
    for start in range(0, array.shape[0], max(1, int(chunk_rows))):
        chunk = np.ascontiguousarray(array[start:start + chunk_rows])
        digest.update(memoryview(chunk).cast('B'))
    return digest.hexdigest()


def _lasso_checkpoint_fingerprint(x: np.ndarray, y: np.ndarray, mask: np.ndarray, marker_names: list[str], trait_names: list[str], repeats: int, sample_fraction: float, alpha: float | None) -> str:
    payload = {'version': 1, 'x_sha256': _sha256_array(x), 'y_sha256': _sha256_array(y), 'mask_sha256': _sha256_array(mask), 'marker_names_sha256': hashlib.sha256('\n'.join((str(value) for value in marker_names)).encode('utf-8')).hexdigest(), 'trait_names': [str(value) for value in trait_names], 'repeats': int(repeats), 'sample_fraction': float(sample_fraction), 'alpha': float(alpha) if alpha is not None and alpha > 0 else None, 'training_seed': int(_training_seed()), 'solver': 'sklearn.linear_model.Lasso', 'max_iter': 1500, 'tol': 0.001, 'selection': 'random'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def _save_lasso_checkpoint(path: Path, *, fingerprint: str, raw_selection_counts: np.ndarray, abs_beta_sum: np.ndarray, next_repeat: int, completed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{uuid4().hex}.tmp.npz')
    np.savez_compressed(temporary, fingerprint=np.asarray(fingerprint), raw_selection_counts=np.asarray(raw_selection_counts, dtype=np.float32), abs_beta_sum=np.asarray(abs_beta_sum, dtype=np.float32), next_repeat=np.asarray(int(next_repeat), dtype=np.int64), completed=np.asarray(int(completed), dtype=np.int64))
    os.replace(temporary, path)


def _load_lasso_checkpoint(path: Path, *, fingerprint: str, marker_count: int) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            cached_fingerprint = str(np.asarray(cached['fingerprint']).item())
            raw_counts = np.asarray(cached['raw_selection_counts'], dtype=np.float32)
            abs_beta_sum = np.asarray(cached['abs_beta_sum'], dtype=np.float32)
            next_repeat = int(np.asarray(cached['next_repeat']).item())
            completed = int(np.asarray(cached['completed']).item())
        if cached_fingerprint != fingerprint:
            return None
        if raw_counts.shape != (marker_count,) or abs_beta_sum.shape != (marker_count,):
            return None
        return (raw_counts, abs_beta_sum, max(0, next_repeat), max(0, completed))
    except (OSError, ValueError, KeyError):
        return None


def _lasso_stability_scores(x: np.ndarray, y: np.ndarray, mask: np.ndarray, marker_names: list[str], trait_names: list[str], repeats: int=50, sample_fraction: float=0.8, alpha: float | None=None, checkpoint_dir: Path | None=None) -> tuple[np.ndarray, dict[str, object]]:
    repeats = int(repeats)
    sample_fraction = float(np.clip(sample_fraction, 0.5, 1.0))
    trait_count = y.shape[1]
    marker_count = x.shape[1]
    selection_counts = np.zeros((trait_count, marker_count), dtype=np.float32)
    trait_summaries: dict[str, object] = {}
    if repeats <= 0:
        for trait in trait_names:
            trait_summaries[trait] = {'runs_completed': 0, 'selected_markers': 0, 'alpha': None, 'reason': 'skipped_lasso_repeats_zero'}
        return (selection_counts, {'method': 'lasso_stability_selection', 'enabled': False, 'skipped': True, 'repeats': 0, 'sample_fraction': sample_fraction, 'score_definition': '0.7*selection_frequency + 0.3*normalized_abs_beta', 'reason': 'LASSO repeats was set to 0, so only GWAS/SNP-Marker prior scores were used.', 'traits': trait_summaries})
    rng = np.random.default_rng(_training_seed())
    checkpoint_root = Path(checkpoint_dir).resolve() if checkpoint_dir is not None else None
    checkpoint_fingerprint = None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        print(f'[LASSO] validating checkpoint cache: {checkpoint_root}', flush=True)
        checkpoint_fingerprint = _lasso_checkpoint_fingerprint(x=x, y=y, mask=mask, marker_names=marker_names, trait_names=trait_names, repeats=repeats, sample_fraction=sample_fraction, alpha=alpha)
    for trait_idx, trait in enumerate(trait_names):
        observed = np.flatnonzero(mask[:, trait_idx] > 0)
        if observed.size < 8:
            trait_summaries[trait] = {'observed_samples': int(observed.size), 'runs_completed': 0, 'selected_markers': 0, 'alpha': None, 'reason': 'too_few_observed_samples'}
            continue
        alpha_value = float(alpha) if alpha is not None and alpha > 0 else _auto_lasso_alpha(observed.size, marker_count)
        sample_size = max(5, int(round(observed.size * sample_fraction)))
        sample_size = min(sample_size, observed.size)
        completed = 0
        next_repeat = 0
        raw_selection_counts = np.zeros(marker_count, dtype=np.float32)
        trait_abs_beta_sum = np.zeros(marker_count, dtype=np.float32)
        checkpoint_path = checkpoint_root / f'trait_{trait_idx:02d}.npz' if checkpoint_root is not None else None
        if checkpoint_path is not None and checkpoint_fingerprint is not None:
            cached = _load_lasso_checkpoint(checkpoint_path, fingerprint=checkpoint_fingerprint, marker_count=marker_count)
            if cached is not None:
                raw_selection_counts, trait_abs_beta_sum, next_repeat, completed = cached
                next_repeat = min(next_repeat, repeats)
                print(f'[LASSO] resumed trait={trait} at repeat={next_repeat}/{repeats} completed_fits={completed}', flush=True)
        for repeat_idx in range(repeats):
            chosen = rng.choice(observed, size=sample_size, replace=False)
            if repeat_idx < next_repeat:
                continue
            x_sub = x[chosen]
            y_sub = y[chosen, trait_idx]
            if np.std(y_sub) < 1e-08:
                if checkpoint_path is not None and checkpoint_fingerprint is not None:
                    _save_lasso_checkpoint(checkpoint_path, fingerprint=checkpoint_fingerprint, raw_selection_counts=raw_selection_counts, abs_beta_sum=trait_abs_beta_sum, next_repeat=repeat_idx + 1, completed=completed)
                continue
            fit_started = time.perf_counter()
            model = Lasso(alpha=alpha_value, max_iter=1500, tol=0.001, selection='random', random_state=_training_seed(repeat_idx + trait_idx * 1009))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                model.fit(x_sub, y_sub)
            coef = np.asarray(model.coef_, dtype=np.float32)
            selected = np.abs(coef) > 1e-08
            raw_selection_counts[selected] += 1.0
            trait_abs_beta_sum += np.abs(coef)
            completed += 1
            if checkpoint_path is not None and checkpoint_fingerprint is not None:
                _save_lasso_checkpoint(checkpoint_path, fingerprint=checkpoint_fingerprint, raw_selection_counts=raw_selection_counts, abs_beta_sum=trait_abs_beta_sum, next_repeat=repeat_idx + 1, completed=completed)
            print(f'[LASSO] trait={trait} repeat={repeat_idx + 1}/{repeats} fit_seconds={time.perf_counter() - fit_started:.1f} selected={int(selected.sum())}', flush=True)
        frequency = raw_selection_counts / max(completed, 1)
        beta_score = trait_abs_beta_sum / max(completed, 1)
        beta_score = _minmax_by_trait(beta_score[None, :])[0]
        selection_score = np.clip(0.7 * frequency + 0.3 * beta_score, 0.0, 1.0)
        selection_counts[trait_idx] = selection_score
        top_idx = np.argsort(selection_score)[::-1][:10]
        trait_summaries[trait] = {'observed_samples': int(observed.size), 'runs_completed': int(completed), 'sample_fraction': sample_fraction, 'sample_size_per_run': int(sample_size), 'alpha': alpha_value, 'selected_markers': int((frequency > 0).sum()), 'top_markers': [{'marker': marker_names[int(idx)], 'lasso_score': float(selection_score[int(idx)]), 'selection_frequency': float(frequency[int(idx)])} for idx in top_idx if selection_score[int(idx)] > 0]}
    summary = {'method': 'lasso_stability_selection', 'repeats': repeats, 'sample_fraction': sample_fraction, 'score_definition': '0.7*selection_frequency + 0.3*normalized_abs_beta', 'checkpoint_cache': {'enabled': bool(checkpoint_root is not None), 'directory': str(checkpoint_root) if checkpoint_root is not None else None, 'fingerprint': checkpoint_fingerprint, 'granularity': 'one atomic checkpoint after each trait/repeat fit'}, 'traits': trait_summaries}
    return (np.clip(selection_counts, 0.0, 1.0).astype(np.float32), summary)


def _build_lasso_gwas_prior(x: np.ndarray, y: np.ndarray, mask: np.ndarray, marker_names: list[str], trait_names: list[str], gwas_prior_scores: np.ndarray | None, gwas_weight: float=0.5, repeats: int=50, prior_sparsity: str | None='none', lasso_checkpoint_dir: Path | None=None) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    gwas_weight = float(np.clip(gwas_weight, 0.0, 1.0))
    repeats = int(repeats)
    skip_lasso = repeats <= 0 or (gwas_prior_scores is not None and gwas_weight >= 1.0 - 1e-08)
    if skip_lasso:
        if gwas_prior_scores is None:
            raise ValueError('LASSO repeats=0 or GWAS weight=1 requires an SNP-Marker/GWAS prior file.')
        lasso_scores = np.zeros_like(gwas_prior_scores, dtype=np.float32)
        lasso_summary = {'method': 'lasso_stability_selection', 'enabled': False, 'skipped': True, 'repeats': max(0, repeats), 'reason': 'LASSO was skipped because repeats=0 or GWAS weight=1. The fused prior uses GWAS/SNP-Marker scores only.', 'traits': {trait: {'runs_completed': 0, 'selected_markers': 0, 'reason': 'skipped_for_gwas_only_prior'} for trait in trait_names}}
        effective_gwas_weight = 1.0
    else:
        lasso_scores, lasso_summary = _lasso_stability_scores(x=x, y=y, mask=mask, marker_names=marker_names, trait_names=trait_names, repeats=repeats, checkpoint_dir=lasso_checkpoint_dir)
        effective_gwas_weight = gwas_weight
    lasso_norm = _minmax_by_trait(lasso_scores)
    if gwas_prior_scores is None:
        gwas_norm = np.zeros_like(lasso_norm, dtype=np.float32)
        fused = lasso_norm
        source = 'lasso_only'
    else:
        gwas_norm = _minmax_by_trait(gwas_prior_scores)
        fused = effective_gwas_weight * gwas_norm + (1.0 - effective_gwas_weight) * lasso_norm
        source = 'gwas_only' if skip_lasso else 'gwas_lasso_fusion'
    sparse_fused, sparsity_summary = _apply_prior_sparsity_with_gwas_mask(fused, prior_sparsity, gwas_prior_scores=gwas_prior_scores)
    prior_scores = _soft_prior_scale(sparse_fused)
    prior_table = pd.DataFrame({'marker': marker_names})
    for trait_idx, trait in enumerate(trait_names):
        prior_table[trait] = sparse_fused[trait_idx]
        prior_table[f'{trait}_fused_raw'] = fused[trait_idx]
        prior_table[f'{trait}_gwas'] = gwas_norm[trait_idx]
        prior_table[f'{trait}_lasso'] = lasso_norm[trait_idx]
    top_by_trait: dict[str, list[dict[str, float | str]]] = {}
    for trait_idx, trait in enumerate(trait_names):
        top_idx = np.argsort(sparse_fused[trait_idx])[::-1][:20]
        top_by_trait[trait] = [{'marker': marker_names[int(idx)], 'prior_score': float(sparse_fused[trait_idx, int(idx)]), 'gwas_score': float(gwas_norm[trait_idx, int(idx)]), 'lasso_score': float(lasso_norm[trait_idx, int(idx)])} for idx in top_idx if sparse_fused[trait_idx, int(idx)] > 0]
    summary = {'enabled': True, 'source': source, 'formula': 'prior = gwas_weight*normalized_GWAS + (1-gwas_weight)*normalized_LASSO', 'requested_gwas_weight': gwas_weight if gwas_prior_scores is not None else 0.0, 'gwas_weight': effective_gwas_weight if gwas_prior_scores is not None else 0.0, 'lasso_weight': 1.0 - effective_gwas_weight if gwas_prior_scores is not None else 1.0, 'lasso_skipped': bool(skip_lasso), 'prior_sparsity': sparsity_summary, 'lasso': lasso_summary, 'top_fused_markers': top_by_trait}
    return (prior_scores.astype(np.float32), summary, prior_table)


def _prior_components_from_generated_table(prior_table: pd.DataFrame | None, trait_names: list[str]) -> dict[str, np.ndarray] | None:
    if prior_table is None or prior_table.empty:
        return None
    gwas_rows: list[np.ndarray] = []
    lasso_rows: list[np.ndarray] = []
    for trait in trait_names:
        gwas_col = f'{trait}_gwas'
        lasso_col = f'{trait}_lasso'
        if gwas_col not in prior_table.columns or lasso_col not in prior_table.columns:
            return None
        gwas_rows.append(pd.to_numeric(prior_table[gwas_col], errors='coerce').fillna(0).to_numpy(np.float32))
        lasso_rows.append(pd.to_numeric(prior_table[lasso_col], errors='coerce').fillna(0).to_numpy(np.float32))
    return {'gwas': np.vstack(gwas_rows).astype(np.float32), 'lasso': np.vstack(lasso_rows).astype(np.float32)}


def _prior_scores_from_components(default_prior_scores: np.ndarray | None, prior_component_scores: dict[str, np.ndarray] | None, params: dict[str, object] | None, prior_sparsity: str | None='none') -> np.ndarray | None:
    if prior_component_scores is None or params is None or 'lasso_prior_gwas_weight' not in params:
        return default_prior_scores
    try:
        weight = float(params.get('lasso_prior_gwas_weight', 0.5))
    except (TypeError, ValueError):
        return default_prior_scores
    weight = float(np.clip(weight, 0.0, 1.0))
    gwas = np.asarray(prior_component_scores.get('gwas'), dtype=np.float32)
    lasso = np.asarray(prior_component_scores.get('lasso'), dtype=np.float32)
    if gwas.shape != lasso.shape:
        return default_prior_scores
    fused = weight * gwas + (1.0 - weight) * lasso
    sparse_fused, _ = _apply_prior_sparsity_with_gwas_mask(fused, prior_sparsity, gwas_prior_scores=gwas)
    return _soft_prior_scale(sparse_fused).astype(np.float32)


def _combine_prior_sources(sources: list[tuple[str, np.ndarray, dict[str, object] | None]], trait_names: list[str]) -> tuple[np.ndarray | None, dict[str, object] | None]:
    valid_sources = [(name, np.asarray(scores, dtype=np.float32), summary) for name, scores, summary in sources if scores is not None]
    if not valid_sources:
        return (None, None)
    if len(valid_sources) == 1:
        name, scores, summary = valid_sources[0]
        result_summary = dict(summary or {})
        result_summary.setdefault('format', f'{name}_prior')
        result_summary['prior_source'] = name
        return (scores.astype(np.float32), result_summary)
    normalized = [_minmax_by_trait(scores) for _, scores, _ in valid_sources]
    merged = np.maximum.reduce(normalized)
    combined_summary = {'format': 'combined_gwas_prior_sources', 'method': 'max_of_normalized_prior_sources', 'sources': [{'name': name, 'summary': summary} for name, _, summary in valid_sources], 'matched_by_trait': {trait: int(np.sum(merged[idx] > 0)) for idx, trait in enumerate(trait_names)}}
    return (_soft_prior_scale(merged).astype(np.float32), combined_summary)


@dataclass
class _FoldwisePriorBuilder:
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    sample_ids: list[str]
    marker_names: list[str]
    trait_names: list[str]
    output_root: Path
    build_tassel_prior: bool
    build_lasso_prior: bool
    lasso_gwas_weight: float
    lasso_repeats: int
    prior_sparsity: str
    tassel_pipeline_path: str | None
    tassel_pc_count: int
    _cache: dict[tuple[object, ...], dict[str, object]] = field(default_factory=dict)

    def _cache_key(self, train_idx: np.ndarray, stage: str, repeat_number: int, fold_number: int) -> tuple[object, ...]:
        indices = np.asarray(train_idx, dtype=np.int64)
        return (str(stage), int(repeat_number), int(fold_number), tuple((int(idx) for idx in indices.tolist())))

    def build(self, train_idx: np.ndarray, *, stage: str, repeat_number: int, fold_number: int, params: dict[str, object] | None=None) -> tuple[np.ndarray | None, dict[str, object]]:
        indices = np.asarray(train_idx, dtype=np.int64)
        if indices.size < 3:
            raise ValueError('Fold-wise prior construction requires at least three training samples.')
        cache_key = self._cache_key(indices, stage, repeat_number, fold_number)
        cached = self._cache.get(cache_key)
        if cached is None:
            fold_output_dir = self.output_root / str(stage) / f'repeat_{int(repeat_number):02d}' / f'fold_{int(fold_number):02d}'
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            x_fold = np.asarray(self.x[indices], dtype=np.float32)
            y_fold = np.asarray(self.y[indices], dtype=np.float32)
            mask_fold = np.asarray(self.mask[indices], dtype=np.float32)
            x_raw_fold = x_fold * self.x_std + self.x_mean
            y_raw_fold = y_fold * self.y_std + self.y_mean
            y_raw_fold = np.where(mask_fold > 0, y_raw_fold, np.nan)
            fold_sample_ids = [self.sample_ids[int(idx)] for idx in indices.tolist()]
            x_lasso_mean = np.nanmean(x_raw_fold, axis=0)
            x_lasso_std = np.nanstd(x_raw_fold, axis=0)
            x_lasso_std = np.where(x_lasso_std > 1e-06, x_lasso_std, 1.0)
            x_lasso_fold = ((x_raw_fold - x_lasso_mean) / x_lasso_std).astype(np.float32)
            y_lasso_fold = np.zeros_like(y_raw_fold, dtype=np.float32)
            for trait_idx in range(y_raw_fold.shape[1]):
                observed = mask_fold[:, trait_idx] > 0.5
                if not np.any(observed):
                    continue
                observed_values = y_raw_fold[observed, trait_idx]
                trait_mean = float(np.mean(observed_values))
                trait_std = float(np.std(observed_values))
                if trait_std <= 1e-06:
                    trait_std = 1.0
                y_lasso_fold[observed, trait_idx] = ((observed_values - trait_mean) / trait_std).astype(np.float32)
            gwas_sources: list[tuple[str, np.ndarray, dict[str, object] | None]] = []
            if self.build_tassel_prior:
                tassel_result = build_tassel_mlm_prior(x_raw=x_raw_fold, y_raw=y_raw_fold, y_mask=mask_fold, sample_ids=fold_sample_ids, marker_names=self.marker_names, trait_names=self.trait_names, output_dir=fold_output_dir / 'tassel_mlm_gwas', tassel_pipeline_path=self.tassel_pipeline_path, pc_count=self.tassel_pc_count, prior_sparsity=self.prior_sparsity)
                gwas_sources.append(('foldwise_tassel_mlm_gwas', tassel_result.prior_scores, tassel_result.summary))
            gwas_scores, gwas_summary = _combine_prior_sources(gwas_sources, self.trait_names)
            generated_prior_table: pd.DataFrame | None = None
            component_scores: dict[str, np.ndarray] | None = None
            lasso_summary: dict[str, object] | None = None
            if self.build_lasso_prior:
                default_scores, lasso_summary, generated_prior_table = _build_lasso_gwas_prior(x=x_lasso_fold, y=y_lasso_fold, mask=mask_fold, marker_names=self.marker_names, trait_names=self.trait_names, gwas_prior_scores=gwas_scores, gwas_weight=self.lasso_gwas_weight, repeats=self.lasso_repeats, prior_sparsity=self.prior_sparsity)
                component_scores = _prior_components_from_generated_table(generated_prior_table, self.trait_names)
            else:
                default_scores = gwas_scores
            cached = {'default_scores': default_scores, 'component_scores': component_scores, 'gwas_summary': gwas_summary, 'lasso_summary': lasso_summary, 'output_dir': str(fold_output_dir), 'train_samples': int(indices.size), 'observed_samples_by_trait': {trait: int(np.sum(mask_fold[:, trait_idx] > 0.5)) for trait_idx, trait in enumerate(self.trait_names)}, 'preprocessing_scope': 'training_fold_only'}
            self._cache[cache_key] = cached
            if generated_prior_table is not None:
                generated_prior_table.to_csv(fold_output_dir / 'snp_marker_lasso_gwas_prior.csv', index=False)
        selected_scores = _prior_scores_from_components(cached['default_scores'], cached['component_scores'], params, prior_sparsity=self.prior_sparsity)
        if self.build_lasso_prior:
            selected_weight = float(params['lasso_prior_gwas_weight']) if params is not None and 'lasso_prior_gwas_weight' in params else float(self.lasso_gwas_weight)
        else:
            selected_weight = 1.0
        summary = {'enabled': True, 'scope': 'training_fold_only', 'stage': str(stage), 'repeat': int(repeat_number), 'fold': int(fold_number), 'train_samples': cached['train_samples'], 'observed_samples_by_trait': cached['observed_samples_by_trait'], 'validation_phenotypes_used': False, 'preprocessing_scope': cached['preprocessing_scope'], 'tassel_mlm_gwas': cached['gwas_summary'], 'lasso_gwas_fusion': cached['lasso_summary'], 'selected_gwas_weight': float(np.clip(selected_weight, 0.0, 1.0)), 'selected_lasso_weight': float(1.0 - np.clip(selected_weight, 0.0, 1.0)), 'output_dir': cached['output_dir']}
        return (selected_scores, summary)


def _resolve_training_options(
    task_type: str | None,
    allow_missing_phenotype: bool | None,
    attention_mode: str | None,
) -> tuple[str, bool, bool, str]:
    attention_mode = _normalize_attention_mode(attention_mode)
    use_marker_attention = attention_mode != 'none'
    task_type = task_type or 'multi_trait'
    if task_type not in TASK_TYPES:
        raise ValueError(f'Unknown task_type: {task_type}')
    if allow_missing_phenotype is None:
        allow_missing_phenotype = True
    return task_type, bool(allow_missing_phenotype), use_marker_attention, attention_mode


def prepare_training_data(genotype_file, phenotype_file, task_type: str='multi_trait', allow_missing_phenotype: bool=True, trait_name: str | None=None, trait_names: list[str] | None=None, retain_all_missing: bool=False):
    geno = _read_csv(genotype_file)
    pheno = _read_csv(phenotype_file)
    if 'sample_id' not in geno.columns or 'sample_id' not in pheno.columns:
        raise ValueError('genotype.csv and phenotype.csv must both contain sample_id.')
    merged = geno.merge(pheno, on='sample_id', how='inner', suffixes=('_marker', '_trait'))
    if merged.empty:
        raise ValueError('No shared sample_id values were found.')
    marker_names = [col for col in geno.columns if col != 'sample_id']
    all_trait_names = [col for col in pheno.columns if col != 'sample_id']
    if not marker_names or not all_trait_names:
        raise ValueError('At least one marker and one trait are required.')
    if task_type == 'single_trait':
        selected = trait_name or all_trait_names[0]
        if selected not in all_trait_names:
            raise ValueError(f"Trait '{selected}' was not found in phenotype.csv.")
        selected_trait_names = [selected]
    else:
        if trait_names:
            missing_traits = [name for name in trait_names if name not in all_trait_names]
            if missing_traits:
                available = ', '.join(all_trait_names)
                missing = ', '.join(missing_traits)
                raise ValueError(f'Trait(s) not found in phenotype.csv: {missing}. Available traits: {available}')
            selected_trait_names = list(dict.fromkeys(trait_names))
        else:
            selected_trait_names = all_trait_names
        if len(selected_trait_names) < 2:
            raise ValueError('At least two traits must be selected for multi-trait training.')
    x = merged[marker_names].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    y_raw = merged[selected_trait_names].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    y_mask = np.isfinite(y_raw).astype(np.float32)
    usable_before_filter = y_mask.sum(axis=1) > 0
    if allow_missing_phenotype:
        keep = np.ones(y_mask.shape[0], dtype=bool) if retain_all_missing else y_mask.sum(axis=1) > 0
    else:
        keep = y_mask.sum(axis=1) == y_mask.shape[1]
    sample_ids = merged.loc[keep, 'sample_id'].astype(str).tolist()
    x = x[keep]
    y_raw = y_raw[keep]
    y_mask = y_mask[keep]
    all_missing_count = int(np.sum(~usable_before_filter))
    dropped_all_missing_samples = 0 if retain_all_missing else all_missing_count
    retained_all_missing_samples = all_missing_count if retain_all_missing else 0
    phenotype_missing_summary = _phenotype_missing_summary(y_mask, selected_trait_names, dropped_all_missing_samples=dropped_all_missing_samples, retained_all_missing_samples=retained_all_missing_samples)
    if x.shape[0] < 3:
        raise ValueError('At least 3 usable samples are required after applying the selected model mode.')
    if y_mask.sum() == 0:
        raise ValueError('No observed phenotype values were found.')
    if not allow_missing_phenotype:
        y_mask = np.ones_like(y_mask, dtype=np.float32)
    x, marker_fill_values = marker_mode_impute(x)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std > 1e-06, x_std, 1.0)
    x_scaled = (x - x_mean) / x_std
    y_mean = np.nanmean(y_raw, axis=0)
    y_mean = np.where(np.isfinite(y_mean), y_mean, 0.0)
    y_std = np.nanstd(y_raw, axis=0)
    y_std = np.where(y_std > 1e-06, y_std, 1.0)
    y_filled = np.where(np.isfinite(y_raw), y_raw, y_mean)
    y_scaled = (y_filled - y_mean) / y_std
    return (x_scaled, y_scaled, y_mask, marker_names, selected_trait_names, x_mean, x_std, y_mean, y_std, marker_fill_values, phenotype_missing_summary, sample_ids)


def _prepare_truth_phenotypes(phenotype_truth_file, sample_ids: list[str], trait_names: list[str], y_mean: np.ndarray, y_std: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    truth = _read_csv(phenotype_truth_file)
    if 'sample_id' not in truth.columns:
        raise ValueError('phenotype truth CSV must contain sample_id.')
    missing_traits = [trait for trait in trait_names if trait not in truth.columns]
    if missing_traits:
        raise ValueError('Trait(s) missing from phenotype truth CSV: ' + ', '.join(missing_traits))
    if truth['sample_id'].astype(str).duplicated().any():
        raise ValueError('phenotype truth CSV contains duplicate sample_id values.')
    truth = truth.copy()
    truth['sample_id'] = truth['sample_id'].astype(str)
    available_sample_ids = set(truth['sample_id'].tolist())
    missing_samples = [sample_id for sample_id in sample_ids if sample_id not in available_sample_ids]
    if missing_samples:
        raise ValueError('phenotype truth CSV is missing sample_id values required by training data: ' + ', '.join(missing_samples[:10]))
    truth = truth.set_index('sample_id').reindex(sample_ids)
    raw = truth[trait_names].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    truth_mask = np.isfinite(raw).astype(np.float32)
    filled = np.where(np.isfinite(raw), raw, y_mean)
    scaled = ((filled - y_mean) / y_std).astype(np.float32)
    summary = {'provided': True, 'sample_count': int(len(sample_ids)), 'observed_truth_values': int(truth_mask.sum()), 'missing_truth_values': int(truth_mask.size - truth_mask.sum()), 'usage': 'evaluation_only', 'note': 'Truth phenotypes are aligned by sample_id and are never passed to the training loss.'}
    return (scaled, truth_mask, summary)


def _train_validation_split(sample_count: int, seed: int=42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)
    val_count = max(1, int(round(sample_count * 0.2)))
    val_count = min(val_count, sample_count - 2)
    return (indices[val_count:], indices[:val_count])


def _kfold_splits(sample_count: int, folds: int=5, seed: int=42) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = max(2, min(folds, sample_count))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(sample_count)
    fold_indices = np.array_split(indices, folds)
    splits = []
    for fold_idx in range(folds):
        val_idx = fold_indices[fold_idx]
        train_idx = np.concatenate([fold_indices[i] for i in range(folds) if i != fold_idx])
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


def _missing_pattern_stratified_kfold_splits(y_mask: np.ndarray, folds: int=5, seed: int=42) -> list[tuple[np.ndarray, np.ndarray]]:
    mask = np.asarray(y_mask, dtype=np.float32)
    sample_count = int(mask.shape[0])
    trait_count = int(mask.shape[1]) if mask.ndim == 2 else 0
    folds = max(2, min(int(folds), sample_count))
    if trait_count < 2:
        return _kfold_splits(sample_count, folds=folds, seed=seed)
    patterns = np.array([''.join(('1' if value > 0 else '0' for value in row)) for row in mask])
    unique_patterns, counts = np.unique(patterns, return_counts=True)
    if np.any(counts < folds):
        return _kfold_splits(sample_count, folds=folds, seed=seed)
    rng = np.random.default_rng(seed)
    fold_buckets: list[list[int]] = [[] for _ in range(folds)]
    for pattern in unique_patterns:
        indices = np.where(patterns == pattern)[0]
        rng.shuffle(indices)
        for fold_idx, chunk in enumerate(np.array_split(indices, folds)):
            fold_buckets[fold_idx].extend((int(idx) for idx in chunk.tolist()))
    all_indices = np.arange(sample_count)
    splits = []
    for fold_idx in range(folds):
        val_idx = np.asarray(fold_buckets[fold_idx], dtype=np.int64)
        if val_idx.size == 0:
            return _kfold_splits(sample_count, folds=folds, seed=seed)
        rng.shuffle(val_idx)
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
        if train_idx.size == 0:
            return _kfold_splits(sample_count, folds=folds, seed=seed)
        val_mask = mask[val_idx]
        if np.any((mask.sum(axis=0) >= folds) & (val_mask.sum(axis=0) == 0)):
            return _kfold_splits(sample_count, folds=folds, seed=seed)
        splits.append((train_idx, val_idx))
    return splits


def _validation_metrics(pred_scaled: np.ndarray, y_scaled: np.ndarray, mask: np.ndarray, trait_names: list[str], y_mean: np.ndarray, y_std: np.ndarray) -> dict[str, dict[str, float | None]]:
    pred = pred_scaled * y_std + y_mean
    target = y_scaled * y_std + y_mean
    metrics: dict[str, dict[str, float | None]] = {}
    for i, trait in enumerate(trait_names):
        observed = mask[:, i] > 0
        if observed.sum() == 0:
            metrics[trait] = {'pearson': None, 'spearman': None, 'rmse': None, 'mae': None}
            continue
        errors = pred[observed, i] - target[observed, i]
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        if observed.sum() >= 2 and np.std(pred[observed, i]) > 1e-08 and (np.std(target[observed, i]) > 1e-08):
            with np.errstate(invalid='ignore', divide='ignore'):
                value = float(np.corrcoef(pred[observed, i], target[observed, i])[0, 1])
            pearson = value if np.isfinite(value) else None
        else:
            pearson = None
        spearman = _rank_correlation(target[observed, i], pred[observed, i])
        metrics[trait] = {'pearson': pearson, 'spearman': spearman, 'rmse': rmse, 'mae': mae}
    return metrics


def _conformal_quantile(residuals: list[float], confidence: float=0.95) -> float | None:
    values = np.asarray([value for value in residuals if np.isfinite(value)], dtype=np.float32)
    if values.size == 0:
        return None
    confidence = float(np.clip(confidence, 0.5, 0.999))
    rank = int(np.ceil((values.size + 1) * confidence))
    rank = min(max(rank, 1), values.size)
    return float(np.partition(values, rank - 1)[rank - 1])


def _append_individualized_conformal_pairs(pairs_by_trait: list[list[tuple[float, float]]], pred_scaled: np.ndarray, y_scaled: np.ndarray, uncertainty_scaled: np.ndarray, mask: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> None:
    pred = pred_scaled * y_std + y_mean
    target = y_scaled * y_std + y_mean
    uncertainty = np.abs(uncertainty_scaled * y_std)
    for trait_idx in range(pred.shape[1]):
        observed = mask[:, trait_idx] > 0
        if observed.sum() == 0:
            continue
        residuals = np.abs(pred[observed, trait_idx] - target[observed, trait_idx])
        uncertainties = uncertainty[observed, trait_idx]
        valid = np.isfinite(residuals) & np.isfinite(uncertainties)
        for residual, trait_uncertainty in zip(residuals[valid].tolist(), uncertainties[valid].tolist()):
            pairs_by_trait[trait_idx].append((float(residual), float(trait_uncertainty)))


def _uncertainty_floor(values: np.ndarray) -> float | None:
    positive = values[np.isfinite(values) & (values > 1e-12)]
    if positive.size == 0:
        return None
    low_quantile = float(np.quantile(positive, 0.05))
    median_floor = float(np.median(positive)) * 0.05
    return float(max(low_quantile, median_floor, 1e-08))


def _individualized_conformal_summary_from_pairs(pairs_by_trait: list[list[tuple[float, float]]], trait_names: list[str], confidence: float=0.95, source: str='mc_dropout_scaled_out_of_fold_residuals') -> dict[str, object]:
    traits: dict[str, object] = {}
    any_enabled = False
    for trait_idx, trait in enumerate(trait_names):
        pairs = pairs_by_trait[trait_idx]
        residuals = np.asarray([pair[0] for pair in pairs if np.isfinite(pair[0])], dtype=np.float32)
        uncertainties = np.asarray([pair[1] for pair in pairs if np.isfinite(pair[1])], dtype=np.float32)
        valid = np.isfinite(residuals) & np.isfinite(uncertainties)
        residuals = residuals[valid]
        uncertainties = uncertainties[valid]
        floor = _uncertainty_floor(uncertainties)
        if residuals.size == 0 or floor is None:
            traits[trait] = {'enabled': False, 'scale_quantile': None, 'uncertainty_floor': None, 'calibration_samples': int(residuals.size), 'reason': 'no_valid_mc_dropout_uncertainty'}
            continue
        effective_uncertainty = np.maximum(uncertainties, float(floor))
        scaled_scores = residuals / effective_uncertainty
        scale_quantile = _conformal_quantile(scaled_scores.tolist(), confidence=confidence)
        if scale_quantile is None or not np.isfinite(scale_quantile):
            traits[trait] = {'enabled': False, 'scale_quantile': None, 'uncertainty_floor': float(floor), 'calibration_samples': int(residuals.size), 'reason': 'invalid_scale_quantile'}
            continue
        scale_quantile = float(scale_quantile)
        calibrated_radii = scale_quantile * effective_uncertainty
        inside = residuals <= calibrated_radii
        traits[trait] = {'enabled': True, 'scale_quantile': scale_quantile, 'uncertainty_floor': float(floor), 'calibration_samples': int(residuals.size), 'mean_abs_residual': float(np.mean(residuals)), 'median_abs_residual': float(np.median(residuals)), 'mean_mc_dropout_std': float(np.mean(uncertainties)), 'median_mc_dropout_std': float(np.median(uncertainties)), 'mean_scaled_residual': float(np.mean(scaled_scores)), 'empirical_coverage': float(np.mean(inside)) if inside.size else None, 'nominal_coverage': float(confidence), 'average_interval_width': float(np.mean(2.0 * calibrated_radii))}
        any_enabled = True
    return {'enabled': any_enabled, 'method': 'mc_dropout_scaled_conformal', 'confidence': float(confidence), 'source': source, 'interval': 'prediction +/- scale_quantile * max(mc_dropout_std, uncertainty_floor)', 'individualized': True, 'uncertainty_source': 'mc_dropout_std', 'coverage_type': 'oof_coverage' if 'out_of_fold' in source else 'calibration_coverage', 'independent_test_coverage': False, 'note': 'Individualized radii are calibrated from residuals divided by MC dropout uncertainty. Coverage is estimated from calibration residuals, not from an external test set.', 'traits': traits}


def _individualized_conformal_summary_from_predictions(pred_scaled: np.ndarray, y_scaled: np.ndarray, uncertainty_scaled: np.ndarray, mask: np.ndarray, trait_names: list[str], y_mean: np.ndarray, y_std: np.ndarray, confidence: float=0.95, source: str='holdout_validation_mc_dropout_scaled_residuals') -> dict[str, object]:
    pairs_by_trait: list[list[tuple[float, float]]] = [[] for _ in trait_names]
    _append_individualized_conformal_pairs(pairs_by_trait, pred_scaled, y_scaled, uncertainty_scaled, mask, y_mean, y_std)
    return _individualized_conformal_summary_from_pairs(pairs_by_trait, trait_names, confidence=confidence, source=source)


def _finite_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _oof_interval_rows_for_fold(repeat_number: int, fold_number: int, sample_ids: list[str], val_idx: np.ndarray, y_scaled: np.ndarray, mask: np.ndarray, trait_names: list[str], y_mean: np.ndarray, y_std: np.ndarray, interval_summary: dict[str, object], mc_mean_scaled: np.ndarray, mc_uncertainty_scaled: np.ndarray, model_train_samples: int=0, validation_samples: int=0) -> list[dict[str, object]]:
    prediction_values = mc_mean_scaled * y_std + y_mean
    target = y_scaled * y_std + y_mean
    mc_std = np.abs(mc_uncertainty_scaled * y_std)
    interval_traits = interval_summary.get('traits') or {}
    rows: list[dict[str, object]] = []
    for local_idx, sample_index in enumerate(np.asarray(val_idx, dtype=np.int64)):
        sample_id = sample_ids[int(sample_index)] if int(sample_index) < len(sample_ids) else str(int(sample_index))
        for trait_idx, trait in enumerate(trait_names):
            observed = bool(mask[local_idx, trait_idx] > 0)
            true_value = float(target[local_idx, trait_idx]) if observed else None
            prediction = float(prediction_values[local_idx, trait_idx])
            residual = abs(prediction - true_value) if true_value is not None else None
            trait_interval = interval_traits.get(trait, {}) if isinstance(interval_traits, dict) else {}
            scale_quantile = _finite_or_none(trait_interval.get('scale_quantile')) if trait_interval.get('enabled', False) else None
            uncertainty_floor = _finite_or_none(trait_interval.get('uncertainty_floor')) if trait_interval.get('enabled', False) else None
            mc_dropout_std = _finite_or_none(mc_std[local_idx, trait_idx])
            radius = None
            if scale_quantile is not None and uncertainty_floor is not None and mc_dropout_std is not None:
                radius = float(scale_quantile * max(mc_dropout_std, uncertainty_floor))
            lower = prediction - radius if radius is not None else None
            upper = prediction + radius if radius is not None else None
            covered = bool(true_value >= lower and true_value <= upper) if true_value is not None and lower is not None and upper is not None else None
            rows.append({'repeat': int(repeat_number), 'fold': int(fold_number), 'sample_index': int(sample_index), 'sample_id': sample_id, 'trait': trait, 'observed': observed, 'true_value': true_value, 'prediction': prediction, 'residual': residual, 'lower_95': lower, 'upper_95': upper, 'radius': radius, 'covered': covered, 'mc_dropout_std': mc_dropout_std, 'scale_quantile': scale_quantile, 'uncertainty_floor': uncertainty_floor, 'model_train_samples': int(model_train_samples), 'calibration_samples': int(trait_interval.get('calibration_samples', 0) or 0), 'validation_samples': int(validation_samples)})
    return rows


def _pearson_from_arrays(true_values: np.ndarray, predicted_values: np.ndarray) -> float | None:
    if true_values.size < 2:
        return None
    if np.std(true_values) <= 1e-08 or np.std(predicted_values) <= 1e-08:
        return None
    with np.errstate(invalid='ignore', divide='ignore'):
        value = float(np.corrcoef(predicted_values, true_values)[0, 1])
    return value if np.isfinite(value) else None


def _summarize_oof_interval_rows(rows: list[dict[str, object]], trait_names: list[str], prediction_key: str, lower_key: str, upper_key: str, radius_key: str, covered_key: str) -> dict[str, object]:
    traits: dict[str, object] = {}
    for trait in trait_names:
        trait_rows = [row for row in rows if row.get('trait') == trait and row.get('observed') and (_finite_or_none(row.get(prediction_key)) is not None)]
        if not trait_rows:
            traits[trait] = {'evaluation_samples': 0, 'empirical_coverage': None, 'pearson': None, 'rmse': None, 'mae': None, 'average_interval_width': None, 'mean_radius': None}
            continue
        y_true = np.asarray([float(row['true_value']) for row in trait_rows], dtype=np.float64)
        y_pred = np.asarray([float(row[prediction_key]) for row in trait_rows], dtype=np.float64)
        errors = y_pred - y_true
        interval_rows = [row for row in trait_rows if _finite_or_none(row.get(lower_key)) is not None and _finite_or_none(row.get(upper_key)) is not None and (row.get(covered_key) is not None)]
        widths = np.asarray([float(row[upper_key]) - float(row[lower_key]) for row in interval_rows], dtype=np.float64)
        radii = np.asarray([float(row[radius_key]) for row in interval_rows if _finite_or_none(row.get(radius_key)) is not None], dtype=np.float64)
        covered_values = np.asarray([bool(row[covered_key]) for row in interval_rows], dtype=bool)
        traits[trait] = {'evaluation_samples': int(len(trait_rows)), 'interval_evaluation_samples': int(len(interval_rows)), 'empirical_coverage': float(np.mean(covered_values)) if covered_values.size else None, 'nominal_coverage': 0.95, 'pearson': _pearson_from_arrays(y_true, y_pred), 'rmse': float(np.sqrt(np.mean(errors ** 2))) if errors.size else None, 'mae': float(np.mean(np.abs(errors))) if errors.size else None, 'average_interval_width': float(np.mean(widths)) if widths.size else None, 'mean_radius': float(np.mean(radii)) if radii.size else None}
    return {'enabled': bool(rows), 'traits': traits}


def _write_oof_prediction_tables(rows: list[dict[str, object]], output_dir: Path | None) -> dict[str, object]:
    if output_dir is None or not rows:
        return {'enabled': False, 'reason': 'no_oof_rows'}
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / 'oof_prediction_intervals_long.csv'
    by_sample_trait_path = output_dir / 'oof_prediction_intervals_by_sample_trait.csv'
    wide_path = output_dir / 'oof_prediction_intervals_wide.csv'
    df = pd.DataFrame(rows)
    optional_interval_columns = ['lower_95', 'upper_95', 'radius', 'covered', 'mc_dropout_std', 'scale_quantile', 'uncertainty_floor']
    empty_optional_columns = [column for column in optional_interval_columns if column in df.columns and df[column].isna().all()]
    if empty_optional_columns:
        df = df.drop(columns=empty_optional_columns)
    df.to_csv(long_path, index=False)
    observed = df[df['observed'] == True].copy()
    if observed.empty:
        by_sample_trait = pd.DataFrame()
        wide = pd.DataFrame()
    else:
        numeric_columns = ['true_value', 'prediction', 'residual', 'lower_95', 'upper_95', 'radius', 'mc_dropout_std', 'scale_quantile', 'uncertainty_floor']
        for column in numeric_columns:
            if column in observed.columns:
                observed[column] = pd.to_numeric(observed[column], errors='coerce')
        for column in ('covered',):
            if column in observed.columns:
                observed[column] = observed[column].astype(float)
        aggregation = {column: ['mean', 'std'] for column in numeric_columns if column in observed.columns}
        aggregation.update({'covered': ['mean'] if 'covered' in observed.columns else [], 'repeat': ['count']})
        aggregation = {key: value for key, value in aggregation.items() if value}
        by_sample_trait = observed.groupby(['sample_index', 'sample_id', 'trait'], dropna=False).agg(aggregation)
        by_sample_trait.columns = [f'{column}_{stat}' if stat != 'mean' else f'{column}_mean' for column, stat in by_sample_trait.columns.to_flat_index()]
        by_sample_trait = by_sample_trait.reset_index()
        by_sample_trait = by_sample_trait.rename(columns={'covered_mean': 'coverage_rate', 'repeat_count': 'oof_rows'})
        wide_parts = []
        wide_value_columns = ['true_value_mean', 'prediction_mean', 'lower_95_mean', 'upper_95_mean', 'radius_mean', 'mc_dropout_std_mean', 'coverage_rate']
        for column in wide_value_columns:
            if column not in by_sample_trait.columns:
                continue
            pivot = by_sample_trait.pivot_table(index=['sample_index', 'sample_id'], columns='trait', values=column, aggfunc='first')
            pivot.columns = [f'{trait}_{column}' for trait in pivot.columns]
            wide_parts.append(pivot)
        wide = pd.concat(wide_parts, axis=1).reset_index() if wide_parts else pd.DataFrame()
    by_sample_trait.to_csv(by_sample_trait_path, index=False)
    wide.to_csv(wide_path, index=False)
    return {'enabled': True, 'long_csv': str(long_path), 'by_sample_trait_csv': str(by_sample_trait_path), 'wide_csv': str(wide_path), 'rows': int(len(df)), 'observed_rows': int(observed.shape[0]), 'note': 'Rows are generated from the same repeated K-fold validation folds used for the main CV metrics. No inner calibration split is used.'}


def _summarize_cv_metrics(fold_metrics: list[dict[str, dict[str, float | None]]]) -> dict[str, dict[str, float | None]]:
    if not fold_metrics:
        return {}
    trait_names = fold_metrics[0].keys()
    summary: dict[str, dict[str, float | None]] = {}
    for trait in trait_names:
        summary[trait] = {}
        for metric_name in ('pearson', 'spearman', 'rmse', 'mae'):
            values = [metrics[trait][metric_name] for metrics in fold_metrics if metrics.get(trait, {}).get(metric_name) is not None]
            if values:
                arr = np.asarray(values, dtype=np.float32)
                summary[trait][f'{metric_name}_mean'] = float(arr.mean())
                summary[trait][f'{metric_name}_std'] = float(arr.std(ddof=0))
            else:
                summary[trait][f'{metric_name}_mean'] = None
                summary[trait][f'{metric_name}_std'] = None
    return summary


def _predict_in_chunks(model: torch.nn.Module, x_t: torch.Tensor, chunk_size: int=64) -> torch.Tensor:
    chunks = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, x_t.shape[0], chunk_size):
            chunks.append(model(x_t[start:start + chunk_size].to(device)).cpu())
    return torch.cat(chunks, dim=0)


def _mc_dropout_predict_scaled(model: torch.nn.Module, x: np.ndarray | torch.Tensor, passes: int=20) -> tuple[np.ndarray, np.ndarray]:
    passes = max(2, int(passes))
    x_t = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    was_training = bool(model.training)
    prior_rates: list[tuple[object, float]] = []
    for module in model.modules():
        if hasattr(module, 'prior_dropout_rate'):
            prior_rates.append((module, float(getattr(module, 'prior_dropout_rate'))))
    _set_prior_dropout_rate(model, 0.0)
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(passes):
            preds.append(_predict_in_chunks(model, x_t).cpu().numpy())
    stacked = np.stack(preds, axis=0)
    if was_training:
        model.train()
    else:
        model.eval()
    for module, rate in prior_rates:
        if hasattr(module, 'set_prior_dropout_rate'):
            module.set_prior_dropout_rate(rate)
        else:
            setattr(module, 'prior_dropout_rate', rate)
    return (stacked.mean(axis=0).astype(np.float32), stacked.std(axis=0).astype(np.float32))


def _attention_runtime_scale_for_epoch(epoch: int, epochs: int, warmup_epochs: int | None=None, ramp_epochs: int | None=None) -> float:
    if warmup_epochs is None:
        warmup_epochs = max(1, int(round(epochs * 0.2)))
    else:
        warmup_epochs = max(0, int(warmup_epochs))
    if ramp_epochs is None:
        ramp_epochs = max(1, int(round(epochs * 0.3)))
    else:
        ramp_epochs = max(1, int(ramp_epochs))
    if epoch <= warmup_epochs:
        return 0.0
    return float(np.clip((epoch - warmup_epochs) / ramp_epochs, 0.0, 1.0))


def _set_attention_runtime_scale(model: torch.nn.Module, scale: float) -> None:
    if hasattr(model, 'set_attention_runtime_scale'):
        model.set_attention_runtime_scale(scale)


def _set_prior_dropout_rate(model: torch.nn.Module, rate: float) -> None:
    if hasattr(model, 'set_prior_dropout_rate'):
        model.set_prior_dropout_rate(rate)


def _attention_regularization(model: torch.nn.Module) -> torch.Tensor | None:
    if hasattr(model, 'attention_regularization'):
        return model.attention_regularization()
    return None


def _masked_trait_pearson(pred: np.ndarray, y: np.ndarray, mask: np.ndarray, trait_idx: int) -> float | None:
    observed = mask[:, trait_idx] > 0
    if observed.sum() < 3:
        return None
    pred_values = pred[observed, trait_idx]
    true_values = y[observed, trait_idx]
    if np.std(pred_values) <= 1e-08 or np.std(true_values) <= 1e-08:
        return None
    value = float(np.corrcoef(pred_values, true_values)[0, 1])
    return value if np.isfinite(value) else None


def _mean_masked_pearson(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float | None:
    values = [value for trait_idx in range(y.shape[1]) if (value := _masked_trait_pearson(pred, y, mask, trait_idx)) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _pdae_corrupt_batch(mask: torch.Tensor, apply_probability: float) -> tuple[torch.Tensor, torch.Tensor]:
    observed = mask > 0.5
    drop_mask = torch.zeros_like(observed)
    eligible_rows = torch.nonzero(observed.sum(dim=1) >= 2, as_tuple=False).reshape(-1)
    if eligible_rows.numel() > 0 and apply_probability > 0:
        selected = eligible_rows[torch.rand(eligible_rows.numel(), device=mask.device) < float(np.clip(apply_probability, 0.0, 1.0))]
        for row_idx in selected.tolist():
            observed_traits = torch.nonzero(observed[row_idx], as_tuple=False).reshape(-1)
            chosen = observed_traits[torch.randint(observed_traits.numel(), (1,), device=mask.device)]
            drop_mask[row_idx, chosen] = True
    return ((observed & ~drop_mask).float(), drop_mask.float())


def _pdae_reconstruction_loss(pdae: PhenotypeDenoisingAutoencoder, yb: torch.Tensor, mb: torch.Tensor, mask_rate: float, observed_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_mask, drop_mask = _pdae_corrupt_batch(mb, mask_rate)
    corrupted = torch.where(input_mask > 0.5, yb, torch.zeros_like(yb))
    reconstructed = pdae(corrupted, input_mask)
    if float(drop_mask.sum().detach().cpu()) > 0:
        denoise_loss = masked_mse_loss(reconstructed, yb, drop_mask)
    else:
        denoise_loss = reconstructed.new_tensor(0.0)
    observed = mb > 0.5
    full_input = torch.where(observed, yb, torch.zeros_like(yb))
    full_reconstruction = pdae(full_input, mb)
    observed_loss = masked_mse_loss(full_reconstruction, yb, mb)
    total = denoise_loss + float(observed_weight) * observed_loss
    return (total, denoise_loss, observed_loss)


def _pdae_mask_one_trait(mask: np.ndarray, trait_idx: int) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(mask, dtype=np.float32) > 0.5
    input_mask = observed.copy()
    drop_mask = np.zeros_like(observed, dtype=np.float32)
    eligible = observed[:, trait_idx] & (observed.sum(axis=1) >= 2)
    input_mask[eligible, trait_idx] = False
    drop_mask[eligible, trait_idx] = 1.0
    return (input_mask.astype(np.float32), drop_mask)


def _pdae_mc_predict_scaled(pdae: PhenotypeDenoisingAutoencoder, y: np.ndarray, mask: np.ndarray, device: torch.device, passes: int=PDAE_MC_PASSES) -> tuple[np.ndarray, np.ndarray]:
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
    model_was_training = pdae.training
    predictions: list[np.ndarray] = []
    pdae.train()
    with torch.no_grad():
        for _ in range(max(2, int(passes))):
            prediction = pdae(torch.where(mask_tensor > 0.5, y_tensor, torch.zeros_like(y_tensor)), mask_tensor)
            predictions.append(prediction.detach().cpu().numpy())
    pdae.train(model_was_training)
    stacked = np.stack(predictions, axis=0).astype(np.float32)
    return (stacked.mean(axis=0).astype(np.float32), stacked.std(axis=0).astype(np.float32))


def _new_pdae_model(trait_count: int, params: dict[str, object], device: torch.device) -> PhenotypeDenoisingAutoencoder:
    return PhenotypeDenoisingAutoencoder(trait_count=trait_count, hidden_dim=int(params['pdae_hidden_dim']), dropout=min(0.25, max(0.05, float(params['dropout']) * 0.5))).to(device)


def _pdae_each_trait_validation_loss(pdae: PhenotypeDenoisingAutoencoder, y: np.ndarray, mask: np.ndarray, validation_idx: np.ndarray, device: torch.device) -> float:
    validation_idx = np.asarray(validation_idx, dtype=np.int64)
    if validation_idx.size == 0:
        return float('inf')
    y_validation = y[validation_idx]
    mask_validation = mask[validation_idx]
    y_tensor = torch.tensor(y_validation, dtype=torch.float32, device=device)
    losses: list[float] = []
    pdae.eval()
    with torch.no_grad():
        for trait_idx in range(y.shape[1]):
            input_mask_np, drop_mask_np = _pdae_mask_one_trait(mask_validation, trait_idx)
            if float(drop_mask_np.sum()) < 1:
                continue
            input_mask = torch.tensor(input_mask_np, dtype=torch.float32, device=device)
            drop_mask = torch.tensor(drop_mask_np, dtype=torch.float32, device=device)
            prediction = pdae(torch.where(input_mask > 0.5, y_tensor, torch.zeros_like(y_tensor)), input_mask)
            losses.append(float(masked_mse_loss(prediction, y_tensor, drop_mask).detach().cpu()))
    return float(np.mean(losses)) if losses else float('inf')


def _fit_pdae_model(y: np.ndarray, mask: np.ndarray, fit_idx: np.ndarray, validation_idx: np.ndarray | None, params: dict[str, object], device: torch.device, main_epochs: int, seed: int, fixed_epochs: int | None=None) -> tuple[PhenotypeDenoisingAutoencoder, dict[str, object]]:
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    validation_idx = np.asarray(validation_idx, dtype=np.int64) if validation_idx is not None else np.asarray([], dtype=np.int64)
    torch.manual_seed(seed)
    pdae = _new_pdae_model(y.shape[1], params, device)
    optimizer = optim.AdamW(pdae.parameters(), lr=float(np.clip(float(params['learning_rate']), 0.0001, 0.003)), weight_decay=float(params['weight_decay']))
    loader_generator = torch.Generator().manual_seed(seed)
    fit_loader = DataLoader(TensorDataset(torch.tensor(y[fit_idx], dtype=torch.float32), torch.tensor(mask[fit_idx], dtype=torch.float32)), batch_size=min(int(params['batch_size']), len(fit_idx)), shuffle=True, generator=loader_generator)
    requested_epochs = max(20, min(80, int(round(main_epochs * 0.3))))
    epochs_to_run = max(1, int(fixed_epochs)) if fixed_epochs is not None else requested_epochs
    patience = max(6, min(15, requested_epochs // 5))
    use_early_stopping = validation_idx.size > 0 and fixed_epochs is None
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float('inf')
    best_epoch = 0
    wait = 0
    final_train_loss = 0.0
    for epoch in range(1, epochs_to_run + 1):
        pdae.train()
        epoch_losses: list[float] = []
        for yb, mb in fit_loader:
            optimizer.zero_grad()
            yb = yb.to(device)
            mb = mb.to(device)
            reconstruction_loss, _, _ = _pdae_reconstruction_loss(pdae, yb, mb, mask_rate=float(params['pdae_mask_rate']), observed_weight=float(params['pdae_loss_weight']))
            reconstruction_loss.backward()
            torch.nn.utils.clip_grad_norm_(pdae.parameters(), max_norm=3.0)
            optimizer.step()
            epoch_losses.append(float(reconstruction_loss.detach().cpu()))
        final_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        if use_early_stopping:
            validation_loss = _pdae_each_trait_validation_loss(pdae, y, mask, validation_idx, device)
            if validation_loss < best_validation_loss - 1e-06:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in pdae.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        else:
            best_epoch = epoch
    if best_state is not None:
        pdae.load_state_dict(best_state)
    for parameter in pdae.parameters():
        parameter.requires_grad_(False)
    pdae.eval()
    return (pdae, {'fit_samples': int(len(fit_idx)), 'validation_complete_samples': int(len(validation_idx)), 'epochs_requested': int(epochs_to_run), 'best_epoch': int(best_epoch), 'best_validation_mse_scaled': float(best_validation_loss) if np.isfinite(best_validation_loss) else None, 'final_train_loss_scaled': float(final_train_loss)})


def _pdae_predict_each_trait_masked(pdae: PhenotypeDenoisingAutoencoder, y: np.ndarray, mask: np.ndarray, device: torch.device, passes: int=PDAE_MC_PASSES) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    prediction_mean = np.full_like(y, np.nan, dtype=np.float32)
    prediction_std = np.full_like(y, np.nan, dtype=np.float32)
    for trait_idx in range(y.shape[1]):
        input_mask, drop_mask = _pdae_mask_one_trait(mask, trait_idx)
        eligible = drop_mask[:, trait_idx] > 0.5
        if not np.any(eligible):
            continue
        mean, std = _pdae_mc_predict_scaled(pdae, y, input_mask, device, passes=passes)
        prediction_mean[eligible, trait_idx] = mean[eligible, trait_idx]
        prediction_std[eligible, trait_idx] = std[eligible, trait_idx]
    return (prediction_mean, prediction_std)


def _pdae_metric_summary(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(truth)
    prediction = prediction[valid]
    truth = truth[valid]
    if prediction.size == 0:
        return {'pearson': None, 'spearman': None, 'rmse': None, 'mae': None}
    residual = prediction - truth
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    pearson = None
    if prediction.size >= 3 and np.std(prediction) > 1e-08 and (np.std(truth) > 1e-08):
        with np.errstate(invalid='ignore', divide='ignore'):
            value = float(np.corrcoef(prediction, truth)[0, 1])
        pearson = value if np.isfinite(value) else None
    spearman = _rank_correlation(truth, prediction)
    return {'pearson': pearson, 'spearman': spearman, 'rmse': rmse, 'mae': mae}


def _pdae_positive_affine_calibration(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | bool | None]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(truth)
    prediction = prediction[valid]
    truth = truth[valid]
    if prediction.size < 3:
        return {'valid': False, 'intercept': 0.0, 'slope': 1.0, 'raw_slope': None, 'slope_was_clipped': False}
    prediction_mean = float(np.mean(prediction))
    truth_mean = float(np.mean(truth))
    centered_prediction = prediction - prediction_mean
    centered_truth = truth - truth_mean
    prediction_variance = float(np.mean(centered_prediction ** 2))
    covariance = float(np.mean(centered_prediction * centered_truth))
    raw_slope = covariance / max(prediction_variance + PDAE_AFFINE_RIDGE, 1e-08)
    if not np.isfinite(raw_slope) or raw_slope <= 0:
        return {'valid': False, 'intercept': 0.0, 'slope': 1.0, 'raw_slope': float(raw_slope) if np.isfinite(raw_slope) else None, 'slope_was_clipped': False}
    slope = float(np.clip(raw_slope, PDAE_AFFINE_SLOPE_MIN, PDAE_AFFINE_SLOPE_MAX))
    intercept = truth_mean - slope * prediction_mean
    return {'valid': True, 'intercept': float(intercept), 'slope': slope, 'raw_slope': float(raw_slope), 'slope_was_clipped': bool(abs(slope - raw_slope) > 1e-12)}


def _weighted_pseudo_label_loss(prediction: torch.Tensor, pseudo_target: torch.Tensor, pseudo_weight: torch.Tensor) -> torch.Tensor:
    denominator = pseudo_weight.sum()
    if float(denominator.detach().cpu()) < 1e-08:
        return prediction.new_tensor(0.0)
    return ((prediction - pseudo_target) ** 2 * pseudo_weight).sum() / denominator


def _select_pdae_genomic_residual_blend(genomic_prediction: np.ndarray, pdae_prediction: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    genomic_prediction = np.asarray(genomic_prediction, dtype=np.float64)
    pdae_prediction = np.asarray(pdae_prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.isfinite(genomic_prediction) & np.isfinite(pdae_prediction) & np.isfinite(truth)
    genomic_prediction = genomic_prediction[valid]
    pdae_prediction = pdae_prediction[valid]
    truth = truth[valid]
    genomic_metrics = _pdae_metric_summary(genomic_prediction, truth)
    pdae_metrics = _pdae_metric_summary(pdae_prediction, truth)
    teacher_pearson = genomic_metrics.get('pearson')
    candidates: list[dict[str, float | None]] = []
    selected_weight = 0.0
    selected_pearson = teacher_pearson
    if teacher_pearson is not None and genomic_prediction.size >= 3:
        best_score = float(teacher_pearson)
        for weight in PDAE_PEARSON_BLEND_GRID:
            blended = genomic_prediction + float(weight) * (pdae_prediction - genomic_prediction)
            metrics = _pdae_metric_summary(blended, truth)
            pearson = metrics.get('pearson')
            candidates.append({'pdae_residual_weight': float(weight), 'pearson': float(pearson) if pearson is not None else None})
            if pearson is not None and float(pearson) > best_score + 1e-12:
                best_score = float(pearson)
                selected_weight = float(weight)
                selected_pearson = float(pearson)
    pearson_gain = float(selected_pearson) - float(teacher_pearson) if selected_pearson is not None and teacher_pearson is not None else None
    enabled = bool(selected_weight > 0 and pearson_gain is not None and (pearson_gain >= PDAE_MIN_PEARSON_GAIN))
    if not enabled:
        selected_weight = 0.0
        selected_pearson = teacher_pearson
    return {'enabled': enabled, 'method': 'cross_fitted_genomic_teacher_plus_pdae_residual_grid_search', 'calibration_samples': int(genomic_prediction.size), 'genomic_teacher_metrics_scaled': genomic_metrics, 'pdae_only_metrics_scaled': pdae_metrics, 'candidate_weights': candidates, 'selected_pdae_residual_weight': float(selected_weight), 'selected_pearson': float(selected_pearson) if selected_pearson is not None else None, 'pearson_gain_over_genomic_teacher': pearson_gain, 'minimum_required_pearson_gain': float(PDAE_MIN_PEARSON_GAIN)}


def _pretrain_pdae_teacher(x: np.ndarray, y: np.ndarray, mask: np.ndarray, train_idx: np.ndarray, trait_names: list[str], params: dict[str, object], device: torch.device, main_epochs: int, prior_scores: np.ndarray | None=None, truth_y: np.ndarray | None=None, truth_mask: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    trait_count = int(y.shape[1])
    pseudo_targets = np.zeros((len(train_idx), trait_count), dtype=np.float32)
    pseudo_weights = np.zeros_like(pseudo_targets)
    observed_counts = mask[train_idx].sum(axis=1)
    missing_candidates = (mask[train_idx] < 0.5) & (observed_counts[:, None] > 0.5)
    complete_idx = train_idx[observed_counts >= trait_count]
    teacher_marker_count = min(PDAE_GENOMIC_TEACHER_MAX_MARKERS, int(x.shape[1]))
    teacher_marker_source = 'training_fold_variance'
    teacher_marker_score = np.var(x[train_idx], axis=0)
    if prior_scores is not None:
        prior_array = np.asarray(prior_scores, dtype=np.float32)
        if prior_array.ndim == 1 and prior_array.shape[0] == x.shape[1]:
            candidate_score = prior_array
        elif prior_array.ndim == 2 and prior_array.shape[1] == x.shape[1]:
            candidate_score = np.nanmax(prior_array, axis=0)
        else:
            candidate_score = None
        if candidate_score is not None and np.any(np.isfinite(candidate_score) & (candidate_score > 0)):
            teacher_marker_score = np.nan_to_num(candidate_score, nan=0.0)
            teacher_marker_source = 'maximum_trait_prior_score'
    teacher_marker_idx = np.argsort(teacher_marker_score, kind='stable')[-teacher_marker_count:]
    x_teacher = x[:, teacher_marker_idx]
    seed = 314159 + int(np.sum(train_idx) % 100003)
    min_fit_complete = max(PDAE_MIN_COMPLETE_SAMPLES, trait_count * 2)
    summary: dict[str, object] = {'enabled': False, 'requested': True, 'method': 'fold_local_cross_fitted_genomic_teacher_plus_pdae_residual', 'complete_anchor_samples': int(len(complete_idx)), 'missing_candidate_cells': int(missing_candidates.sum()), 'all_trait_missing_samples_skipped': int(np.sum(observed_counts < 0.5)), 'genomic_teacher_marker_selection': {'source': teacher_marker_source, 'selected_markers': int(teacher_marker_count), 'available_markers': int(x.shape[1])}}
    if not np.any(missing_candidates):
        summary['reason'] = 'no_missing_targets'
        return (pseudo_targets, pseudo_weights, summary)
    if len(complete_idx) < PDAE_MIN_CALIBRATION_SAMPLES_PER_TRAIT:
        summary.update({'reason': 'insufficient_complete_anchor_samples', 'required_complete_anchor_samples': int(PDAE_MIN_CALIBRATION_SAMPLES_PER_TRAIT), 'note': 'PDAE pseudo-labeling was disabled because cross-trait reconstruction cannot be calibrated reliably.'})
        return (pseudo_targets, pseudo_weights, summary)
    rng = np.random.default_rng(seed)
    shuffled_complete = complete_idx.copy()
    rng.shuffle(shuffled_complete)
    crossfit_folds = min(PDAE_CROSSFIT_FOLDS, len(shuffled_complete))
    crossfit_splits: list[np.ndarray] | None = None
    while crossfit_folds >= 2:
        candidate_splits = [np.asarray(values, dtype=np.int64) for values in np.array_split(shuffled_complete, crossfit_folds) if len(values)]
        split_is_feasible = True
        for holdout_idx in candidate_splits:
            remaining_count = len(shuffled_complete) - len(holdout_idx)
            internal_validation_count = max(trait_count * 2, int(round(remaining_count * PDAE_INTERNAL_VALIDATION_FRACTION)))
            if remaining_count - internal_validation_count < min_fit_complete:
                split_is_feasible = False
                break
        if split_is_feasible:
            crossfit_splits = candidate_splits
            break
        crossfit_folds -= 1
    if crossfit_splits is None:
        summary.update({'reason': 'insufficient_complete_anchor_samples_for_crossfit', 'required_fit_complete_samples_per_fold': int(min_fit_complete), 'requested_crossfit_folds': int(PDAE_CROSSFIT_FOLDS)})
        return (pseudo_targets, pseudo_weights, summary)
    calibration_mean_raw = np.full((len(complete_idx), trait_count), np.nan, dtype=np.float32)
    calibration_std_raw = np.full_like(calibration_mean_raw, np.nan)
    calibration_baseline = np.full_like(calibration_mean_raw, np.nan)
    genomic_calibration = np.full_like(calibration_mean_raw, np.nan)
    complete_position = {int(sample_idx): position for position, sample_idx in enumerate(complete_idx.tolist())}
    crossfit_details: list[dict[str, object]] = []
    best_epochs: list[int] = []
    crossfit_validation_losses: list[float] = []
    for fold_idx, holdout_idx in enumerate(crossfit_splits, start=1):
        holdout_set = {int(value) for value in holdout_idx.tolist()}
        remaining_complete = np.asarray([value for value in shuffled_complete.tolist() if int(value) not in holdout_set], dtype=np.int64)
        fold_rng = np.random.default_rng(seed + fold_idx * 1009)
        fold_rng.shuffle(remaining_complete)
        internal_validation_count = max(trait_count * 2, int(round(len(remaining_complete) * PDAE_INTERNAL_VALIDATION_FRACTION)))
        internal_validation_count = min(internal_validation_count, len(remaining_complete) - min_fit_complete)
        internal_validation_idx = remaining_complete[:internal_validation_count]
        excluded = holdout_set | {int(value) for value in internal_validation_idx.tolist()}
        fit_idx = np.asarray([idx for idx in train_idx.tolist() if int(idx) not in excluded and mask[int(idx)].sum() > 0], dtype=np.int64)
        pdae_fold, fold_summary = _fit_pdae_model(y=y, mask=mask, fit_idx=fit_idx, validation_idx=internal_validation_idx, params=params, device=device, main_epochs=main_epochs, seed=seed + fold_idx * 1009)
        holdout_mean, holdout_std = _pdae_predict_each_trait_masked(pdae_fold, y[holdout_idx], mask[holdout_idx], device, passes=PDAE_MC_PASSES)
        positions = np.asarray([complete_position[int(value)] for value in holdout_idx.tolist()], dtype=np.int64)
        calibration_mean_raw[positions] = holdout_mean
        calibration_std_raw[positions] = holdout_std
        reference_idx = np.asarray([idx for idx in train_idx.tolist() if int(idx) not in holdout_set], dtype=np.int64)
        genomic_teacher_fold = _fit_ridge_per_trait(x_teacher[reference_idx], y[reference_idx], mask[reference_idx], alpha=PDAE_GENOMIC_TEACHER_RIDGE_ALPHA)
        genomic_calibration[positions] = genomic_teacher_fold.predict_scaled(x_teacher[holdout_idx])
        for trait_idx in range(trait_count):
            reference_observed = mask[reference_idx, trait_idx] > 0.5
            if np.any(reference_observed):
                calibration_baseline[positions, trait_idx] = float(np.mean(y[reference_idx[reference_observed], trait_idx]))
        best_epoch = int(fold_summary['best_epoch'])
        best_epochs.append(best_epoch)
        if fold_summary.get('best_validation_mse_scaled') is not None:
            crossfit_validation_losses.append(float(fold_summary['best_validation_mse_scaled']))
        crossfit_details.append({'fold': int(fold_idx), 'holdout_complete_samples': int(len(holdout_idx)), **fold_summary})
    if not best_epochs:
        summary['reason'] = 'crossfit_training_failed'
        return (pseudo_targets, pseudo_weights, summary)
    final_teacher_epochs = max(10, int(round(float(np.median(best_epochs)))))
    final_fit_idx = np.asarray([idx for idx in train_idx.tolist() if mask[int(idx)].sum() > 0], dtype=np.int64)
    pdae, final_teacher_summary = _fit_pdae_model(y=y, mask=mask, fit_idx=final_fit_idx, validation_idx=None, params=params, device=device, main_epochs=main_epochs, seed=seed + 9001, fixed_epochs=final_teacher_epochs)
    pseudo_mean_raw, pseudo_std_raw = _pdae_mc_predict_scaled(pdae, y[train_idx], mask[train_idx], device, passes=PDAE_MC_PASSES)
    genomic_teacher = _fit_ridge_per_trait(x_teacher[train_idx], y[train_idx], mask[train_idx], alpha=PDAE_GENOMIC_TEACHER_RIDGE_ALPHA)
    genomic_candidate_prediction = genomic_teacher.predict_scaled(x_teacher[train_idx])
    trait_reliability: dict[str, object] = {}
    for trait_idx, trait in enumerate(trait_names):
        truth = y[complete_idx, trait_idx].astype(np.float32)
        raw_prediction = calibration_mean_raw[:, trait_idx]
        raw_uncertainty = calibration_std_raw[:, trait_idx]
        affine = _pdae_positive_affine_calibration(raw_prediction, truth)
        intercept = float(affine['intercept'])
        slope = float(affine['slope'])
        calibration_prediction = intercept + slope * raw_prediction
        calibration_uncertainty = abs(slope) * raw_uncertainty
        pdae_candidate_prediction = intercept + slope * pseudo_mean_raw[:, trait_idx]
        genomic_prediction = genomic_calibration[:, trait_idx]
        residual_blend = _select_pdae_genomic_residual_blend(genomic_prediction, calibration_prediction, truth)
        residual_weight = float(residual_blend['selected_pdae_residual_weight'])
        combined_calibration_prediction = genomic_prediction + residual_weight * (calibration_prediction - genomic_prediction)
        pseudo_targets[:, trait_idx] = genomic_candidate_prediction[:, trait_idx] + residual_weight * (pdae_candidate_prediction - genomic_candidate_prediction[:, trait_idx])
        candidate_uncertainty = residual_weight * abs(slope) * pseudo_std_raw[:, trait_idx]
        valid = np.isfinite(combined_calibration_prediction) & np.isfinite(calibration_uncertainty) & np.isfinite(truth)
        residuals = np.abs(combined_calibration_prediction[valid] - truth[valid])
        uncertainties = residual_weight * calibration_uncertainty[valid]
        baseline = calibration_baseline[:, trait_idx][valid]
        truth_valid = truth[valid]
        raw_metrics = _pdae_metric_summary(raw_prediction, truth)
        calibrated_metrics = _pdae_metric_summary(calibration_prediction, truth)
        calibration_mse = float(np.mean((calibration_prediction[valid] - truth_valid) ** 2)) if np.any(valid) else None
        baseline_valid = np.isfinite(baseline) & np.isfinite(truth_valid)
        baseline_mse = float(np.mean((baseline[baseline_valid] - truth_valid[baseline_valid]) ** 2)) if np.any(baseline_valid) else None
        relative_mean_skill = float(1.0 - calibration_mse / baseline_mse) if calibration_mse is not None and baseline_mse is not None and (baseline_mse > 1e-08) else None
        pearson = calibrated_metrics.get('pearson')
        rmse = calibrated_metrics.get('rmse')
        reliability_checks = {'enough_calibration_samples': bool(residuals.size >= PDAE_MIN_CALIBRATION_SAMPLES_PER_TRAIT), 'positive_affine_slope': bool(affine['valid']), 'pearson_above_threshold': bool(pearson is not None and float(pearson) >= PDAE_MIN_RECONSTRUCTION_PEARSON), 'skill_above_threshold': bool(relative_mean_skill is not None and float(relative_mean_skill) > PDAE_MIN_RELATIVE_MEAN_SKILL), 'rmse_below_threshold': bool(rmse is not None and float(rmse) < PDAE_MAX_SCALED_RMSE), 'pearson_safety_gate': bool(residual_blend['enabled'])}
        reliable = bool(all(reliability_checks.values()))
        uncertainty_floor = _uncertainty_floor(uncertainties)
        if uncertainty_floor is None:
            uncertainty_floor = max(0.001, float(np.median(residuals)) * 0.05) if residuals.size else 0.001
        scores = residuals / (uncertainties + uncertainty_floor) if residuals.size else np.asarray([], dtype=np.float32)
        conformal_scale = _conformal_quantile(scores.tolist(), confidence=PDAE_CONFIDENCE_LEVEL)
        calibration_radii = float(conformal_scale) * (uncertainties + uncertainty_floor) if conformal_scale is not None and residuals.size else np.asarray([], dtype=np.float32)
        radius_threshold = float(np.quantile(calibration_radii, PDAE_CONFIDENCE_KEEP_QUANTILE)) if calibration_radii.size else None
        candidates = missing_candidates[:, trait_idx]
        accepted = np.zeros(len(train_idx), dtype=bool)
        candidate_radii = np.full(len(train_idx), np.nan, dtype=np.float32)
        candidate_radius_threshold = None
        effective_radius_threshold = radius_threshold
        if reliable and conformal_scale is not None and (radius_threshold is not None) and (radius_threshold > 0):
            candidate_radii[candidates] = float(conformal_scale) * (candidate_uncertainty[candidates] + float(uncertainty_floor))
            finite_candidate_radii = candidate_radii[candidates & np.isfinite(candidate_radii)]
            if finite_candidate_radii.size:
                candidate_radius_threshold = float(np.quantile(finite_candidate_radii, PDAE_CONFIDENCE_KEEP_QUANTILE))
                effective_radius_threshold = min(float(radius_threshold), candidate_radius_threshold)
            accepted = candidates & np.isfinite(candidate_radii) & (candidate_radii <= float(effective_radius_threshold))
            normalized_radius = candidate_radii[accepted] / max(float(effective_radius_threshold), 1e-08)
            pseudo_weights[accepted, trait_idx] = 1.0 / (1.0 + normalized_radius ** 2)
        hidden_truth_evaluation: dict[str, object] = {'enabled': False, 'reason': 'phenotype_truth_not_provided', 'usage': 'evaluation_only_never_used_for_training_or_gating'}
        if truth_y is not None and truth_mask is not None:
            hidden_truth = truth_y[train_idx, trait_idx]
            hidden_truth_available = candidates & (truth_mask[train_idx, trait_idx] > 0.5) & np.isfinite(hidden_truth)
            accepted_truth_available = accepted & hidden_truth_available
            hidden_truth_evaluation = {'enabled': bool(np.any(hidden_truth_available)), 'usage': 'evaluation_only_never_used_for_training_or_gating', 'all_candidate_cells': int(np.sum(hidden_truth_available)), 'accepted_candidate_cells': int(np.sum(accepted_truth_available)), 'combined_pseudo_all_candidates_metrics_scaled': _pdae_metric_summary(pseudo_targets[hidden_truth_available, trait_idx], hidden_truth[hidden_truth_available]), 'combined_pseudo_accepted_metrics_scaled': _pdae_metric_summary(pseudo_targets[accepted_truth_available, trait_idx], hidden_truth[accepted_truth_available]), 'genomic_teacher_all_candidates_metrics_scaled': _pdae_metric_summary(genomic_candidate_prediction[hidden_truth_available, trait_idx], hidden_truth[hidden_truth_available]), 'pdae_only_all_candidates_metrics_scaled': _pdae_metric_summary(pdae_candidate_prediction[hidden_truth_available], hidden_truth[hidden_truth_available])}
        trait_reliability[trait] = {'reliable': reliable, 'reliability_checks': reliability_checks, 'calibration_samples': int(residuals.size), 'raw_masked_reconstruction_metrics_scaled': raw_metrics, 'masked_reconstruction_metrics_scaled': calibrated_metrics, 'relative_mean_skill': relative_mean_skill, 'mean_baseline_mse_scaled': baseline_mse, 'linear_calibration': affine, 'genomic_teacher_plus_pdae_residual': residual_blend, 'combined_calibration_metrics_scaled': _pdae_metric_summary(combined_calibration_prediction, truth), 'uncertainty_floor_scaled': float(uncertainty_floor), 'conformal_scale': float(conformal_scale) if conformal_scale is not None else None, 'radius_threshold_scaled': radius_threshold, 'candidate_radius_threshold_scaled': candidate_radius_threshold, 'effective_radius_threshold_scaled': effective_radius_threshold, 'candidate_cells': int(candidates.sum()), 'accepted_pseudo_cells': int(accepted.sum()), 'accepted_rate': float(accepted.sum() / max(int(candidates.sum()), 1)), 'hidden_truth_evaluation': hidden_truth_evaluation}
    accepted_weights = pseudo_weights[pseudo_weights > 0]
    summary.update({'enabled': bool(accepted_weights.size), 'reason': None if accepted_weights.size else 'no_pseudo_labels_passed_confidence_filter', 'pretraining': 'three_fold_cross_fitted_genomic_teacher_and_pdae_residual_calibration', 'fit_samples': int(len(final_fit_idx)), 'calibration_complete_samples': int(len(complete_idx)), 'calibration_predictions_per_complete_sample': int(trait_count), 'crossfit_folds': int(len(crossfit_splits)), 'crossfit_details': crossfit_details, 'pretrain_epochs_requested': int(max(20, min(80, int(round(main_epochs * 0.3))))), 'pretrain_best_epoch': int(final_teacher_epochs), 'pretrain_best_calibration_mse_scaled': float(np.mean(crossfit_validation_losses)) if crossfit_validation_losses else None, 'pretrain_final_loss_scaled': float(final_teacher_summary['final_train_loss_scaled']), 'final_teacher_epochs': int(final_teacher_epochs), 'mask_application_probability': float(params['pdae_mask_rate']), 'observed_reconstruction_weight': float(params['pdae_loss_weight']), 'pseudo_label_loss_weight': float(params['pdae_pseudo_weight']), 'hidden_dim': int(params['pdae_hidden_dim']), 'mc_dropout_passes': int(PDAE_MC_PASSES), 'confidence_level': float(PDAE_CONFIDENCE_LEVEL), 'confidence_keep_quantile': float(PDAE_CONFIDENCE_KEEP_QUANTILE), 'genomic_teacher': {'model': 'ridge_per_trait', 'ridge_alpha': float(PDAE_GENOMIC_TEACHER_RIDGE_ALPHA), 'maximum_markers': int(PDAE_GENOMIC_TEACHER_MAX_MARKERS), 'marker_selection': teacher_marker_source, 'predictions_for_complete_anchors': 'three_fold_cross_fitted', 'predictions_for_missing_candidates': 'full_outer_training_fold'}, 'pearson_safety_gate': {'blend_grid': [float(value) for value in PDAE_PEARSON_BLEND_GRID], 'minimum_gain': float(PDAE_MIN_PEARSON_GAIN), 'calibration_source': 'cross_fitted_complete_anchors_in_outer_training_fold', 'truth_phenotype_used': False}, 'reliability_thresholds': {'minimum_calibration_samples_per_trait': int(PDAE_MIN_CALIBRATION_SAMPLES_PER_TRAIT), 'minimum_reconstruction_pearson': float(PDAE_MIN_RECONSTRUCTION_PEARSON), 'minimum_relative_mean_skill_exclusive': float(PDAE_MIN_RELATIVE_MEAN_SKILL), 'maximum_scaled_rmse_exclusive': float(PDAE_MAX_SCALED_RMSE)}, 'linear_calibration': {'method': 'cross_fitted_positive_affine_ridge', 'ridge': float(PDAE_AFFINE_RIDGE), 'slope_min': float(PDAE_AFFINE_SLOPE_MIN), 'slope_max': float(PDAE_AFFINE_SLOPE_MAX)}, 'accepted_pseudo_cells': int(accepted_weights.size), 'accepted_pseudo_weight_mean': float(np.mean(accepted_weights)) if accepted_weights.size else None, 'accepted_pseudo_weight_min': float(np.min(accepted_weights)) if accepted_weights.size else None, 'accepted_pseudo_weight_max': float(np.max(accepted_weights)) if accepted_weights.size else None, 'traits': trait_reliability})
    return (pseudo_targets, pseudo_weights, summary)


def _resolve_ppmgs_params(marker_count: int, params: dict[str, object] | None=None, lr: float=0.001) -> dict[str, object]:
    params = params or {}
    hidden_dim = int(params.get('hidden_dim') or hidden_units_from_markers(marker_count))
    hidden_layers = int(params.get('hidden_layers') or 4)
    dropout = float(params.get('dropout', 0.3))
    activation = str(params.get('activation', 'relu')).strip().lower()
    if activation not in {'relu', 'gelu'}:
        activation = 'relu'
    learning_rate = float(params.get('learning_rate', lr))
    weight_decay = float(params.get('weight_decay', 0.0001))
    batch_size = int(params.get('batch_size') or 32)
    pdae_mask_rate = float(params.get('pdae_mask_rate', 0.3))
    pdae_loss_weight = float(params.get('pdae_loss_weight', 0.15))
    pdae_pseudo_weight = float(params.get('pdae_pseudo_weight', 0.01))
    pdae_hidden_dim = int(params.get('pdae_hidden_dim') or 16)
    lr_scheduler = str(params.get('lr_scheduler', 'plateau')).strip().lower()
    if lr_scheduler not in LR_SCHEDULER_MODES:
        lr_scheduler = 'plateau'
    lr_scheduler_factor = float(params.get('lr_scheduler_factor', 0.5))
    lr_scheduler_patience = int(params.get('lr_scheduler_patience', 10))
    min_learning_rate = float(params.get('min_learning_rate', 1e-06))
    attention_warmup_epochs_param = params.get('attention_warmup_epochs')
    attention_ramp_epochs_param = params.get('attention_ramp_epochs')
    block_size_param = params.get('block_size')
    block_size = int(block_size_param) if block_size_param is not None else None
    result = {'hidden_dim': max(16, min(2048, hidden_dim)), 'hidden_layers': max(1, min(6, hidden_layers)), 'dropout': float(np.clip(dropout, 0.0, 0.8)), 'activation': activation, 'learning_rate': max(1e-06, learning_rate), 'weight_decay': max(0.0, weight_decay), 'batch_size': max(1, min(512, batch_size)), 'pdae_mask_rate': float(np.clip(pdae_mask_rate, 0.0, 0.8)), 'pdae_loss_weight': float(np.clip(pdae_loss_weight, 0.0, 2.0)), 'pdae_pseudo_weight': float(np.clip(pdae_pseudo_weight, 0.0, 1.0)), 'pdae_hidden_dim': max(4, min(32, pdae_hidden_dim)), 'lr_scheduler': lr_scheduler, 'lr_scheduler_factor': float(np.clip(lr_scheduler_factor, 0.05, 0.95)), 'lr_scheduler_patience': max(1, min(50, lr_scheduler_patience)), 'min_learning_rate': max(1e-08, min(max(1e-08, learning_rate), min_learning_rate))}
    if attention_warmup_epochs_param is not None:
        result['attention_warmup_epochs'] = max(0, int(attention_warmup_epochs_param))
    if attention_ramp_epochs_param is not None:
        result['attention_ramp_epochs'] = max(1, int(attention_ramp_epochs_param))
    if block_size is not None:
        result['block_size'] = max(10, min(10000, block_size))
    if 'lasso_prior_gwas_weight' in params:
        result['lasso_prior_gwas_weight'] = float(np.clip(float(params.get('lasso_prior_gwas_weight', 0.5)), 0.0, 1.0))
    return result


def _ppmgs_search_space(marker_count: int, use_pdae: bool=False, tune_prior_gwas_weight: bool=False) -> dict[str, list[object]]:
    default_hidden = hidden_units_from_markers(marker_count)
    hidden_dims = [32, 64, 128, 256]
    if default_hidden <= 512:
        hidden_dims.append(default_hidden)
    if marker_count >= 5000:
        hidden_dims.append(512)
    hidden_dims = sorted(set(hidden_dims))
    search_space = {'hidden_dim': [int(value) for value in hidden_dims if 16 <= int(value) <= 2048], 'hidden_layers': [1, 2, 3, 4], 'dropout': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 'activation': ['relu', 'gelu'], 'learning_rate': [1e-05, 3e-05, 0.0001, 0.0003, 0.0005, 0.001], 'weight_decay': [0.0, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01], 'batch_size': [8, 16, 32, 64]}
    if use_pdae:
        search_space.update({'pdae_mask_rate': [0.3], 'pdae_loss_weight': [0.15], 'pdae_pseudo_weight': [0.01], 'pdae_hidden_dim': [16]})
    if tune_prior_gwas_weight:
        search_space['lasso_prior_gwas_weight'] = [0.0, 0.25, 0.5, 0.75, 1.0]
    return search_space


def _score_cv_summary(summary: dict[str, object], metric: str) -> tuple[float, float | None, str]:
    trait_summaries = list(summary.values())
    if metric in {'mse', 'rmse', 'mae'}:
        values = []
        for trait_summary in trait_summaries:
            if not isinstance(trait_summary, dict):
                continue
            if metric == 'mse':
                rmse = trait_summary.get('rmse_mean')
                if rmse is not None and np.isfinite(float(rmse)):
                    values.append(float(rmse) ** 2)
            else:
                value = trait_summary.get(f'{metric}_mean')
                if value is not None and np.isfinite(float(value)):
                    values.append(float(value))
        if not values:
            return (float('-inf'), None, f'{metric}_mean')
        mean_error = float(np.mean(values))
        return (-mean_error, mean_error, f'{metric}_mean')
    if metric == 'stable_pearson':
        values = []
        for trait_summary in trait_summaries:
            if not isinstance(trait_summary, dict):
                continue
            pearson = trait_summary.get('pearson_mean')
            pearson_sd = trait_summary.get('pearson_std')
            if pearson is None or not np.isfinite(float(pearson)):
                continue
            sd_value = float(pearson_sd) if pearson_sd is not None and np.isfinite(float(pearson_sd)) else 0.0
            values.append(0.8 * float(pearson) - 0.2 * sd_value)
        if not values:
            return (float('-inf'), None, 'stable_pearson_score')
        stable_score = float(np.mean(values))
        return (stable_score, stable_score, 'stable_pearson_score')
    values = []
    for trait_summary in trait_summaries:
        pearson = (trait_summary or {}).get('pearson_mean') if isinstance(trait_summary, dict) else None
        if pearson is not None and np.isfinite(float(pearson)):
            values.append(float(pearson))
    if not values:
        return (float('-inf'), None, 'pearson_mean')
    mean_pearson = float(np.mean(values))
    return (mean_pearson, mean_pearson, 'pearson_mean')


def _hyperopt_metric_label(metric: str) -> str:
    if metric == 'mse':
        return 'mse_min'
    if metric == 'rmse':
        return 'rmse_min'
    if metric == 'mae':
        return 'mae_min'
    if metric == 'stable_pearson':
        return 'stable_pearson_max_0.8mean_minus_0.2sd'
    return 'pearson_max'


def _hyperopt_selection_metric_name(metric: str) -> str:
    if metric == 'mse':
        return 'mse_mean'
    if metric == 'rmse':
        return 'rmse_mean'
    if metric == 'mae':
        return 'mae_mean'
    if metric == 'stable_pearson':
        return 'stable_pearson_score'
    return 'pearson_mean'


def _evaluate_ppmgs_params_cv(x: np.ndarray, y: np.ndarray, mask: np.ndarray, trait_names: list[str], y_mean: np.ndarray, y_std: np.ndarray, allow_missing_phenotype: bool, use_pdae: bool, use_directional_transfer: bool, transfer_mode: str, use_marker_attention: bool, attention_mode: str, prior_scores: np.ndarray | None, prior_component_scores: dict[str, np.ndarray] | None, prior_sparsity: str, params: dict[str, object], epochs: int, patience: int, tuning_folds: int, metric: str, trial: object | None=None, foldwise_prior_builder: _FoldwisePriorBuilder | None=None) -> tuple[dict[str, object], float, float | None, str]:
    folds = max(2, int(tuning_folds))
    fold_results = []
    fold_metrics = []
    trial_prior_scores = None if foldwise_prior_builder is not None else _prior_scores_from_components(prior_scores, prior_component_scores, params, prior_sparsity=prior_sparsity)
    splits = _missing_pattern_stratified_kfold_splits(mask, folds=folds, seed=_training_seed()) if allow_missing_phenotype and y.shape[1] >= 2 else _kfold_splits(x.shape[0], folds=folds, seed=_training_seed())
    for fold_number, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_prior_summary = None
        fold_prior_scores = trial_prior_scores
        if foldwise_prior_builder is not None:
            fold_prior_scores, fold_prior_summary = foldwise_prior_builder.build(train_idx, stage='hyperparameter_tuning', repeat_number=1, fold_number=fold_number, params=params)
        _, final_loss, best_epoch, val_pred_np = _fit_model_on_indices(x=x, y=y, mask=mask, train_idx=train_idx, val_idx=val_idx, trait_names=trait_names, allow_missing_phenotype=allow_missing_phenotype, use_pdae=use_pdae, use_directional_transfer=use_directional_transfer, transfer_mode=transfer_mode, use_marker_attention=use_marker_attention, attention_mode=attention_mode, prior_scores=fold_prior_scores, epochs=epochs, lr=float(params['learning_rate']), patience=patience, ppmgs_params=params)
        metrics = _validation_metrics(val_pred_np, y[val_idx], mask[val_idx], trait_names, y_mean, y_std)
        fold_metrics.append(metrics)
        fold_results.append({'repeat': 1, 'fold': fold_number, 'train_samples': int(len(train_idx)), 'validation_samples': int(len(val_idx)), 'best_epoch': int(best_epoch), 'final_loss': float(final_loss), 'metrics': metrics, 'foldwise_prior': fold_prior_summary})
        current_summary = _summarize_cv_metrics(fold_metrics)
        current_score, _, _ = _score_cv_summary(current_summary, metric)
        if trial is not None:
            trial.report(current_score, step=fold_number)
            if trial.should_prune():
                raise TrialPruned()
    summary = _summarize_cv_metrics(fold_metrics)
    score, metric_value, metric_name = _score_cv_summary(summary, metric)
    cv_result = {'folds': folds, 'repeats': 1, 'total_runs': len(fold_results), 'fold_results': fold_results, 'summary': summary, 'foldwise_prior': {'enabled': bool(foldwise_prior_builder is not None), 'scope': 'training_fold_only' if foldwise_prior_builder is not None else None, 'validation_phenotypes_used': False if foldwise_prior_builder is not None else None}}
    return (cv_result, score, metric_value, metric_name)


def _suggest_ppmgs_params(trial: object, marker_count: int, use_pdae: bool=False, tune_prior_gwas_weight: bool=False) -> dict[str, object]:
    search_space = _ppmgs_search_space(marker_count, use_pdae=use_pdae, tune_prior_gwas_weight=tune_prior_gwas_weight)
    return _resolve_ppmgs_params(marker_count, {name: trial.suggest_categorical(name, values) for name, values in search_space.items()})


def _optimize_ppmgs_hyperparameters_tpe(x: np.ndarray, y: np.ndarray, mask: np.ndarray, trait_names: list[str], y_mean: np.ndarray, y_std: np.ndarray, allow_missing_phenotype: bool, use_pdae: bool, use_directional_transfer: bool, transfer_mode: str, use_marker_attention: bool, attention_mode: str, prior_scores: np.ndarray | None, prior_component_scores: dict[str, np.ndarray] | None, prior_sparsity: str, tune_prior_gwas_weight: bool, epochs: int, patience: int, trials: int, tuning_folds: int, metric: str, early_stop_rounds: int | None=None, max_tuning_epochs: int | None=None, scheduler_params: dict[str, object] | None=None, foldwise_prior_builder: _FoldwisePriorBuilder | None=None) -> tuple[dict[str, object], dict[str, object]]:
    if optuna is None:
        raise ValueError('Optuna is required for TPE hyperparameter optimization.')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    search_space = _ppmgs_search_space(x.shape[1], use_pdae=use_pdae, tune_prior_gwas_weight=tune_prior_gwas_weight)
    tuning_epochs = min(int(epochs), 150)
    if max_tuning_epochs is not None and int(max_tuning_epochs) > 0:
        tuning_epochs = max(1, min(tuning_epochs, int(max_tuning_epochs)))
    tuning_patience = min(int(patience), 12)
    trial_results: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None
    best_seen_score = float('-inf')
    sampler = optuna.samplers.TPESampler(seed=_training_seed(), n_startup_trials=min(12, max(1, int(trials) // 5)))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(8, max(1, int(trials) // 5)), n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    def objective(trial):
        nonlocal best_summary, best_seen_score
        params = _suggest_ppmgs_params(trial, x.shape[1], use_pdae=use_pdae, tune_prior_gwas_weight=tune_prior_gwas_weight)
        if scheduler_params:
            params = _resolve_ppmgs_params(x.shape[1], {**params, **scheduler_params})
        try:
            cv_result, score, metric_value, metric_name = _evaluate_ppmgs_params_cv(x=x, y=y, mask=mask, trait_names=trait_names, y_mean=y_mean, y_std=y_std, allow_missing_phenotype=allow_missing_phenotype, use_pdae=use_pdae, use_directional_transfer=use_directional_transfer, transfer_mode=transfer_mode, use_marker_attention=use_marker_attention, attention_mode=attention_mode, prior_scores=prior_scores, prior_component_scores=prior_component_scores, prior_sparsity=prior_sparsity, params=params, epochs=tuning_epochs, patience=tuning_patience, tuning_folds=tuning_folds, metric=metric, trial=trial, foldwise_prior_builder=foldwise_prior_builder)
        except TrialPruned:
            trial_results.append({'trial': int(trial.number) + 1, 'params': params, 'state': 'pruned', 'selection_metric': _hyperopt_selection_metric_name(metric), 'metric_value': None, 'selection_score': None})
            raise
        trial_results.append({'trial': int(trial.number) + 1, 'params': params, 'state': 'complete', 'selection_metric': metric_name, 'metric_value': metric_value, 'selection_score': score})
        if best_summary is None or score > best_seen_score:
            best_seen_score = score
            best_summary = cv_result['summary']
        return score
    requested_trials = max(1, int(trials))
    early_stop_rounds = int(early_stop_rounds or 0)
    if early_stop_rounds <= 0:
        early_stop_rounds = max(10, min(30, requested_trials // 4 if requested_trials >= 4 else requested_trials))
    no_improvement_trials = 0
    callback_best = float('-inf')
    stopped_early = False

    def early_stop_callback(study_obj, trial_obj) -> None:
        nonlocal no_improvement_trials, callback_best, stopped_early
        if trial_obj.state != optuna.trial.TrialState.COMPLETE:
            return
        value = float(trial_obj.value) if trial_obj.value is not None else float('-inf')
        if value > callback_best + 1e-06:
            callback_best = value
            no_improvement_trials = 0
        else:
            no_improvement_trials += 1
        if early_stop_rounds > 0 and no_improvement_trials >= early_stop_rounds:
            stopped_early = True
            study_obj.stop()
    study.optimize(objective, n_trials=requested_trials, show_progress_bar=False, gc_after_trial=True, callbacks=[early_stop_callback])
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise ValueError('All hyperparameter trials were pruned. Try fewer folds or increase the number of trials.')
    best_params = _resolve_ppmgs_params(x.shape[1], {**study.best_trial.params, **(scheduler_params or {})})
    best_score = float(study.best_value)
    _, best_metric_value, metric_name = _score_cv_summary(best_summary or {}, metric)
    complete_results = [row for row in trial_results if row.get('state') == 'complete']
    top_trials = sorted(complete_results, key=lambda row: float(row['selection_score']), reverse=True)[:10]
    pruned_count = sum((1 for row in trial_results if row.get('state') == 'pruned'))
    search_result = {'enabled': True, 'method': 'tpe_bayesian_cv', 'pruner': 'median_pruner', 'trials_requested': int(trials), 'trials_completed': len(completed_trials), 'trials_pruned': pruned_count, 'early_stopping': {'enabled': True, 'early_stop_rounds': int(early_stop_rounds), 'stopped_early': bool(stopped_early), 'no_improvement_trials_at_stop': int(no_improvement_trials)}, 'tuning_folds': max(2, int(tuning_folds)), 'tuning_repeats': 1, 'tuning_epochs': tuning_epochs, 'optimization_metric': _hyperopt_metric_label(metric), 'search_space': search_space, 'best_params': best_params, 'best_score': best_metric_value if best_metric_value is not None else best_score, 'best_selection_score': best_score, 'best_metric_value': best_metric_value, 'best_tuning_summary': best_summary, 'top_trials': top_trials, 'trial_results': trial_results, 'foldwise_prior': {'enabled': bool(foldwise_prior_builder is not None), 'scope': 'tuning_training_fold_only' if foldwise_prior_builder is not None else None, 'validation_phenotypes_used': False if foldwise_prior_builder is not None else None}}
    return (best_params, search_result)


def _optimize_ppmgs_hyperparameters(x: np.ndarray, y: np.ndarray, mask: np.ndarray, trait_names: list[str], y_mean: np.ndarray, y_std: np.ndarray, allow_missing_phenotype: bool, use_pdae: bool, use_directional_transfer: bool, transfer_mode: str, use_marker_attention: bool, attention_mode: str, prior_scores: np.ndarray | None, prior_component_scores: dict[str, np.ndarray] | None, prior_sparsity: str, tune_prior_gwas_weight: bool, epochs: int, lr: float, patience: int, trials: int, tuning_folds: int, metric: str, method: str='tpe', early_stop_rounds: int | None=None, max_tuning_epochs: int | None=None, scheduler_params: dict[str, object] | None=None, foldwise_prior_builder: _FoldwisePriorBuilder | None=None) -> tuple[dict[str, object], dict[str, object]]:
    if str(method).strip().lower() != 'tpe':
        raise ValueError('The final PPMGS-Net workflow supports TPE optimization only.')
    metric = metric if metric in {'pearson', 'stable_pearson', 'mse', 'rmse', 'mae'} else 'pearson'
    return _optimize_ppmgs_hyperparameters_tpe(
        x=x,
        y=y,
        mask=mask,
        trait_names=trait_names,
        y_mean=y_mean,
        y_std=y_std,
        allow_missing_phenotype=allow_missing_phenotype,
        use_pdae=use_pdae,
        use_directional_transfer=use_directional_transfer,
        transfer_mode=transfer_mode,
        use_marker_attention=use_marker_attention,
        attention_mode=attention_mode,
        prior_scores=prior_scores,
        prior_component_scores=prior_component_scores,
        prior_sparsity=prior_sparsity,
        tune_prior_gwas_weight=tune_prior_gwas_weight,
        epochs=epochs,
        patience=patience,
        trials=trials,
        tuning_folds=tuning_folds,
        metric=metric,
        early_stop_rounds=early_stop_rounds,
        max_tuning_epochs=max_tuning_epochs,
        scheduler_params=scheduler_params,
        foldwise_prior_builder=foldwise_prior_builder,
    )


def _trait_matrix_or_none(values: np.ndarray | None, trait_count: int, marker_count: int) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == marker_count:
        return np.tile(arr[None, :], (trait_count, 1))
    if arr.ndim == 2 and arr.shape == (trait_count, marker_count):
        return arr
    if arr.ndim == 2 and arr.shape == (marker_count, trait_count):
        return arr.T
    return None


def _gradient_shap_top_markers(model: object, x: np.ndarray, marker_names: list[str], trait_names: list[str], prior_scores: np.ndarray | None=None, top_k: int=50, gradient_samples: int=8, batch_size: int=8, output_dir: Path | None=None, save_all_markers: bool=False) -> tuple[dict[str, list[dict[str, float | str | int | None]]], dict[str, object]]:
    """Compute full-sample expected-gradient SHAP-style SNP attributions.

    Every training sample is explained. The random background draws come from
    the full training set, and only the final marker list is truncated to top_k.
    """
    if not isinstance(model, torch.nn.Module):
        return ({}, {'enabled': False, 'requested': True, 'reason': 'SHAP marker explanation is currently available for PPMGS-Net torch models only.'})
    x = np.asarray(x, dtype=np.float32)
    sample_count, marker_count = x.shape
    top_k = min(max(1, int(top_k)), marker_count)
    gradient_samples = max(1, int(gradient_samples))
    batch_size = max(1, int(batch_size))
    if sample_count == 0 or marker_count == 0:
        return ({}, {'enabled': False, 'requested': True, 'reason': 'empty genotype matrix.'})
    old_training = model.training
    old_attention_scale = getattr(model, 'attention_runtime_scale', None)
    device = next(model.parameters()).device
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    prior_matrix = _trait_matrix_or_none(prior_scores, len(trait_names), marker_count)
    attention_matrix = _average_attention_weights(model, x, trait_names)
    result: dict[str, list[dict[str, float | str | int | None]]] = {}
    csv_rows: list[dict[str, float | str | int | None]] = []
    all_output_path: Path | None = None
    all_rows_written = 0
    marker_names_array = np.asarray(marker_names, dtype=object)
    model.eval()
    _set_attention_runtime_scale(model, 1.0)
    try:
        for trait_idx, trait in enumerate(trait_names):
            generator = torch.Generator(device=device)
            generator.manual_seed(_training_seed(7877 + trait_idx))
            total_abs = np.zeros(marker_count, dtype=np.float64)
            total_signed = np.zeros(marker_count, dtype=np.float64)
            positive_count = np.zeros(marker_count, dtype=np.float64)
            for start in range(0, sample_count, batch_size):
                x_batch = x_t[start:start + batch_size]
                batch_n = x_batch.shape[0]
                attribution_sum = torch.zeros_like(x_batch)
                for _ in range(gradient_samples):
                    bg_idx = torch.randint(0, sample_count, (batch_n,), device=device, generator=generator)
                    background = x_t[bg_idx]
                    alpha = torch.rand((batch_n, 1), device=device, generator=generator)
                    interpolated = (background + alpha * (x_batch - background)).detach().requires_grad_(True)
                    model.zero_grad(set_to_none=True)
                    output = model(interpolated)[:, trait_idx].sum()
                    gradient = torch.autograd.grad(output, interpolated, retain_graph=False, create_graph=False)[0]
                    attribution_sum = attribution_sum + (x_batch - background) * gradient
                attribution = (attribution_sum / float(gradient_samples)).detach().cpu().numpy()
                total_abs += np.abs(attribution).sum(axis=0)
                total_signed += attribution.sum(axis=0)
                positive_count += (attribution > 0).sum(axis=0)
            mean_abs = total_abs / float(sample_count)
            mean_signed = total_signed / float(sample_count)
            positive_fraction = positive_count / float(sample_count)
            ranked_idx = np.argsort(mean_abs)[::-1]
            top_idx = ranked_idx[:top_k]
            if save_all_markers and output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                if all_output_path is None:
                    all_output_path = output_dir / 'shap_all_markers.csv'
                    if all_output_path.exists():
                        all_output_path.unlink()
                prior_values = prior_matrix[trait_idx].astype(np.float64, copy=False) if prior_matrix is not None else np.full(marker_count, np.nan, dtype=np.float64)
                attention_values = attention_matrix[trait_idx].astype(np.float64, copy=False) if attention_matrix is not None else np.full(marker_count, np.nan, dtype=np.float64)
                all_rows = pd.DataFrame({'trait': np.repeat(trait, marker_count), 'rank': np.arange(1, marker_count + 1, dtype=np.int64), 'marker': marker_names_array[ranked_idx], 'mean_abs_shap': mean_abs[ranked_idx], 'mean_signed_shap': mean_signed[ranked_idx], 'positive_fraction': positive_fraction[ranked_idx], 'prior_score': prior_values[ranked_idx], 'attention_weight': attention_values[ranked_idx]})
                all_rows.to_csv(all_output_path, mode='a', header=all_rows_written == 0, index=False)
                all_rows_written += marker_count
            trait_rows: list[dict[str, float | str | int | None]] = []
            for rank, marker_idx in enumerate(top_idx, start=1):
                prior_score = float(prior_matrix[trait_idx, marker_idx]) if prior_matrix is not None and np.isfinite(prior_matrix[trait_idx, marker_idx]) else None
                attention_weight = float(attention_matrix[trait_idx, marker_idx]) if attention_matrix is not None and np.isfinite(attention_matrix[trait_idx, marker_idx]) else None
                row = {'trait': trait, 'rank': int(rank), 'topSNP': f'topSNP_{rank}', 'marker': marker_names[int(marker_idx)], 'topSNP_marker': f'topSNP_{rank}---{marker_names[int(marker_idx)]}', 'mean_abs_shap': float(mean_abs[int(marker_idx)]), 'mean_signed_shap': float(mean_signed[int(marker_idx)]), 'positive_fraction': float(positive_fraction[int(marker_idx)]), 'prior_score': prior_score, 'attention_weight': attention_weight}
                trait_rows.append(row)
                csv_rows.append(row)
            result[trait] = trait_rows
    finally:
        if old_attention_scale is not None:
            _set_attention_runtime_scale(model, float(old_attention_scale))
        if old_training:
            model.train()
        else:
            model.eval()
    output_file = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'shap_top{top_k}_markers.csv'
        pd.DataFrame(csv_rows).to_csv(output_path, index=False)
        output_file = str(output_path)
    summary = {'enabled': True, 'requested': True, 'method': 'expected_gradients_shap', 'description': 'Full training-sample SHAP-style expected-gradient SNP attribution with Top SNP ranking and optional all-marker export.', 'explained_samples': int(sample_count), 'sample_reduction': False, 'markers': int(marker_count), 'top_k': int(top_k), 'gradient_samples_per_batch': int(gradient_samples), 'background_source': 'random_draws_from_all_training_samples', 'ranking_score': 'mean_abs_shap', 'output_file': output_file, 'top_output_file': output_file, 'all_markers_saved': bool(save_all_markers and all_output_path is not None), 'all_output_file': str(all_output_path) if all_output_path is not None else None, 'all_marker_rows': int(all_rows_written)}
    return (result, summary)


def _learned_prior_strength(model: torch.nn.Module):
    if hasattr(model, 'prior_strength'):
        value = model.prior_strength().detach().cpu()
        if value.ndim == 0:
            return float(value)
        return [float(item) for item in value.numpy().tolist()]
    return None


def _learned_prior_strength_by_trait(model: torch.nn.Module, trait_names: list[str]) -> dict[str, float] | None:
    values = _learned_prior_strength(model)
    if values is None:
        return None
    if isinstance(values, float):
        return {trait: values for trait in trait_names}
    return {trait: float(values[idx]) for idx, trait in enumerate(trait_names[:len(values)])}


def _learned_prior_reliability_by_trait(model: torch.nn.Module, trait_names: list[str]) -> dict[str, float] | None:
    if not hasattr(model, 'prior_reliability'):
        return None
    values = model.prior_reliability().detach().cpu()
    if values.ndim == 0:
        return {trait: float(values) for trait in trait_names}
    return {trait: float(values[idx]) for idx, trait in enumerate(trait_names[:values.numel()])}


def _effective_prior_strength_by_trait(model: torch.nn.Module, trait_names: list[str]) -> dict[str, float] | None:
    if not hasattr(model, 'effective_prior_strength'):
        return None
    values = model.effective_prior_strength().detach().cpu()
    if values.ndim == 0:
        return {trait: float(values) for trait in trait_names}
    return {trait: float(values[idx]) for idx, trait in enumerate(trait_names[:values.numel()])}


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return None
    a_rank = pd.Series(a[valid]).rank(method='average').to_numpy(dtype=np.float32)
    b_rank = pd.Series(b[valid]).rank(method='average').to_numpy(dtype=np.float32)
    if np.std(a_rank) <= 1e-08 or np.std(b_rank) <= 1e-08:
        return None
    value = float(np.corrcoef(a_rank, b_rank)[0, 1])
    return value if np.isfinite(value) else None


def _average_attention_weights(model: torch.nn.Module, x: np.ndarray, trait_names: list[str], chunk_size: int=16) -> np.ndarray | None:
    if not hasattr(model, 'trait_attention_weights') and (not hasattr(model, 'attention_weights')):
        return None
    x_t = torch.tensor(x, dtype=torch.float32)
    device = next(model.parameters()).device
    if hasattr(model, 'trait_attention_weights'):
        total = torch.zeros((len(trait_names), x.shape[1]), dtype=torch.float32)
        seen = 0
        model.eval()
        with torch.no_grad():
            for start in range(0, x_t.shape[0], chunk_size):
                batch = x_t[start:start + chunk_size].to(device)
                weights = model.trait_attention_weights(batch).detach().cpu()
                total += weights.sum(dim=0)
                seen += batch.shape[0]
        return (total / max(seen, 1)).numpy()
    total = torch.zeros(x.shape[1], dtype=torch.float32)
    seen = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, x_t.shape[0], chunk_size):
            batch = x_t[start:start + chunk_size].to(device)
            weights = model.attention_weights(batch).detach().cpu()
            total += weights.sum(dim=0)
            seen += batch.shape[0]
    shared = (total / max(seen, 1)).numpy()
    return np.tile(shared[None, :], (len(trait_names), 1))


def _prior_learning_diagnostics(model: torch.nn.Module, x: np.ndarray, marker_names: list[str], trait_names: list[str], prior_scores: np.ndarray | None, attention_safety: dict[str, object] | None=None, top_k: int=50) -> dict[str, object] | None:
    if prior_scores is None:
        return None
    prior_scores = np.asarray(prior_scores, dtype=np.float32)
    if prior_scores.ndim == 1:
        prior_scores = np.tile(prior_scores[None, :], (len(trait_names), 1))
    if prior_scores.shape != (len(trait_names), len(marker_names)):
        return {'enabled': False, 'reason': f'prior_scores shape {prior_scores.shape} does not match traits x markers.'}
    attention = _average_attention_weights(model, x, trait_names)
    if attention is None or attention.shape != prior_scores.shape:
        return {'enabled': False, 'reason': 'model does not expose compatible attention weights.'}
    safety_details = {}
    if isinstance(attention_safety, dict):
        safety_details = attention_safety.get('details') or {}
    diagnostics: dict[str, object] = {'enabled': True, 'method': 'prior_attention_rank_overlap', 'top_k': int(top_k), 'traits': {}}
    k = min(int(top_k), len(marker_names))
    expected_random_overlap = k / max(len(marker_names), 1) if k > 0 else 0.0
    for trait_idx, trait in enumerate(trait_names):
        prior = prior_scores[trait_idx]
        attn = attention[trait_idx]
        positive = prior > 0
        prior_top_idx = np.argsort(prior)[::-1][:k]
        attention_top_idx = np.argsort(attn)[::-1][:k]
        prior_top_set = set((int(idx) for idx in prior_top_idx))
        attention_top_set = set((int(idx) for idx in attention_top_idx))
        overlap = len(prior_top_set & attention_top_set)
        attention_total = float(np.sum(attn)) if np.isfinite(np.sum(attn)) else 0.0
        attention_mass_top_prior = float(np.sum(attn[prior_top_idx]) / max(attention_total, 1e-08)) if k > 0 else None
        trait_safety = safety_details.get(trait, {}) if isinstance(safety_details, dict) else {}
        diagnostics['traits'][trait] = {'prior_nonzero_markers': int(np.sum(positive)), 'prior_score_max': float(np.max(prior)) if prior.size else None, 'prior_score_mean_positive': float(np.mean(prior[positive])) if np.any(positive) else None, 'spearman_prior_vs_attention_all_markers': _rank_correlation(prior, attn), 'spearman_prior_vs_attention_positive_prior': _rank_correlation(prior[positive], attn[positive]) if np.sum(positive) >= 3 else None, 'top_prior_vs_top_attention_overlap': int(overlap), 'top_prior_vs_top_attention_overlap_rate': float(overlap / max(k, 1)), 'expected_random_overlap_rate': float(expected_random_overlap), 'attention_mass_on_top_prior_markers': attention_mass_top_prior, 'weight_on_main': trait_safety.get('weight_on_main') if isinstance(trait_safety, dict) else None, 'weight_on_attention': trait_safety.get('weight_on_attention') if isinstance(trait_safety, dict) else None, 'top_prior_markers': [{'marker': marker_names[int(idx)], 'prior_score': float(prior[int(idx)])} for idx in prior_top_idx[:min(10, k)]], 'top_attention_markers': [{'marker': marker_names[int(idx)], 'attention': float(attn[int(idx)]), 'prior_score': float(prior[int(idx)])} for idx in attention_top_idx[:min(10, k)]]}
    return diagnostics


def _directional_transfer_state(model: DirectionalAnchorGSNet) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if key.startswith('directional_adapters.') or key == 'directional_gate_logits'}


def _load_directional_transfer_state(model: DirectionalAnchorGSNet, transfer_state: dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    state.update(transfer_state)
    model.load_state_dict(state)
    model.freeze_anchors()


def _directional_trait_pearsons(prediction: np.ndarray, y: np.ndarray, mask: np.ndarray, trait_names: list[str]) -> dict[str, float | None]:
    return {trait: _masked_trait_pearson(prediction, y, mask, trait_idx) for trait_idx, trait in enumerate(trait_names)}


def _fit_directional_anchor_on_indices(x: np.ndarray, y: np.ndarray, mask: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray | None, trait_names: list[str], attention_mode: str, prior_scores: np.ndarray | None, epochs: int, lr: float, patience: int, ppmgs_params: dict[str, object] | None, use_pdae: bool, truth_y: np.ndarray | None, truth_mask: np.ndarray | None) -> tuple[DirectionalAnchorGSNet, float, int, np.ndarray | None]:
    if y.shape[1] < 2:
        raise ValueError('directional_anchor requires at least two phenotype traits.')
    if attention_mode not in PRIOR_MARKER_ATTENTION_MODES:
        raise ValueError('directional_anchor currently requires a prior-marker attention mode.')
    raw_params = dict(ppmgs_params or {})
    final_fit_config = raw_params.get('directional_anchor_final_fit')
    if not isinstance(final_fit_config, dict):
        final_fit_config = {}
    fixed_final_fit = bool(final_fit_config.get('enabled'))
    fixed_anchor_epochs = final_fit_config.get('anchor_epochs_by_trait') or {}
    fixed_transfer_epochs = final_fit_config.get('transfer_epochs')
    selection_val_idx = None if fixed_final_fit else val_idx
    params = _resolve_ppmgs_params(x.shape[1], raw_params, lr=lr)
    trait_count = y.shape[1]
    device = get_torch_device()
    anchor_models: list[PriorMarkerAttentionGSNet] = []
    anchor_summaries: list[dict[str, object]] = []
    prior_values = None if prior_scores is None else np.asarray(prior_scores, dtype=np.float32)
    for trait_idx, trait in enumerate(trait_names):
        anchor_train_idx = np.asarray(train_idx, dtype=np.int64)
        anchor_train_idx = anchor_train_idx[mask[anchor_train_idx, trait_idx] > 0.5]
        if anchor_train_idx.size < 3:
            raise ValueError(f'Trait {trait} has fewer than three observed training samples for directional_anchor.')
        if selection_val_idx is None:
            anchor_val_idx = None
        else:
            anchor_val_idx = np.asarray(selection_val_idx, dtype=np.int64)
            anchor_val_idx = anchor_val_idx[mask[anchor_val_idx, trait_idx] > 0.5]
            if anchor_val_idx.size < 3:
                anchor_val_idx = None
        if prior_values is None:
            anchor_prior = None
        elif prior_values.ndim == 1:
            anchor_prior = prior_values
        else:
            anchor_prior = prior_values[trait_idx:trait_idx + 1]
        anchor_epochs = max(1, int(fixed_anchor_epochs.get(trait, epochs))) if fixed_final_fit else int(epochs)
        anchor_model, anchor_loss, anchor_best_epoch, anchor_val_prediction = _fit_model_on_indices(x=x, y=y[:, trait_idx:trait_idx + 1], mask=mask[:, trait_idx:trait_idx + 1], train_idx=anchor_train_idx, val_idx=anchor_val_idx, trait_names=[trait], allow_missing_phenotype=True, use_pdae=False, use_directional_transfer=False, transfer_mode='none', use_marker_attention=True, attention_mode=attention_mode, prior_scores=anchor_prior, epochs=anchor_epochs, lr=lr, patience=patience, ppmgs_params=params, truth_y=None, truth_mask=None)
        if not isinstance(anchor_model, PriorMarkerAttentionGSNet):
            raise TypeError('directional_anchor expected PriorMarkerAttentionGSNet single-trait anchors.')
        anchor_model.configure_transfer_mode('none')
        anchor_model.eval()
        anchor_models.append(anchor_model)
        validation_pearson = None
        if anchor_val_idx is not None and anchor_val_prediction is not None:
            validation_pearson = _masked_trait_pearson(anchor_val_prediction, y[anchor_val_idx, trait_idx:trait_idx + 1], mask[anchor_val_idx, trait_idx:trait_idx + 1], 0)
        anchor_summaries.append({'trait': trait, 'observed_train_samples': int(anchor_train_idx.size), 'observed_validation_samples': int(anchor_val_idx.size) if anchor_val_idx is not None else 0, 'best_epoch': int(anchor_best_epoch), 'final_loss': float(anchor_loss), 'validation_pearson_diagnostic': validation_pearson, 'validation_pearson_used_for_direction_selection': False, 'fixed_epoch_full_data_fit': bool(fixed_final_fit)})
    model = DirectionalAnchorGSNet(marker_count=x.shape[1], trait_count=trait_count, prior_scores=prior_values, hidden_dim=int(params['hidden_dim']), dropout=float(params['dropout']), hidden_layers=int(params['hidden_layers']), activation=str(params.get('activation', 'relu')), use_prior_reliability_gate=attention_mode in PRIOR_RELIABILITY_ATTENTION_MODES, anchor_models=anchor_models).to(device)
    model.freeze_anchors()
    model.set_attention_runtime_scale(1.0)
    model.set_prior_dropout_rate(0.0)
    pseudo_targets_np = np.zeros((len(train_idx), trait_count), dtype=np.float32)
    pseudo_weights_np = np.zeros_like(pseudo_targets_np)
    pdae_summary: dict[str, object] = {'enabled': False, 'requested': bool(use_pdae), 'reason': 'not_requested'}
    if use_pdae:
        pseudo_targets_np, pseudo_weights_np, pdae_summary = _pretrain_pdae_teacher(x=x, y=y, mask=mask, train_idx=np.asarray(train_idx, dtype=np.int64), trait_names=trait_names, params=params, device=device, main_epochs=epochs, prior_scores=prior_values, truth_y=truth_y, truth_mask=truth_mask)
    pdae_training_enabled = bool(pdae_summary.get('enabled') and np.any(pseudo_weights_np > 0))
    transfer_parameters = model.directional_parameters()
    transfer_lr = float(np.clip(float(params['learning_rate']) * DIRECTIONAL_ANCHOR_LR_MULTIPLIER, 0.0002, 0.002))
    optimizer = optim.AdamW(transfer_parameters, lr=transfer_lr, weight_decay=float(params['weight_decay']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=float(params.get('lr_scheduler_factor', 0.5)), patience=max(3, min(6, int(params.get('lr_scheduler_patience', 10)))), min_lr=float(params.get('min_learning_rate', 1e-06)))
    if fixed_final_fit and fixed_transfer_epochs is not None:
        transfer_epochs = max(0, int(fixed_transfer_epochs))
    else:
        transfer_epochs = max(20, min(DIRECTIONAL_ANCHOR_MAX_TRANSFER_EPOCHS, int(round(epochs * 0.35))))
    transfer_patience = min(DIRECTIONAL_ANCHOR_TRANSFER_PATIENCE, max(5, int(patience)))
    train_loader = DataLoader(TensorDataset(torch.tensor(x[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.float32), torch.tensor(mask[train_idx], dtype=torch.float32), torch.tensor(pseudo_targets_np, dtype=torch.float32), torch.tensor(pseudo_weights_np, dtype=torch.float32)), batch_size=min(int(params['batch_size']), len(train_idx)), shuffle=True, generator=torch.Generator().manual_seed(_training_seed(515109)))
    baseline_prediction = None
    baseline_score = float('-inf')
    baseline_trait_pearsons: dict[str, float | None] = {trait: None for trait in trait_names}
    if selection_val_idx is not None:
        model.eval()
        model.set_anchors_eval()
        with torch.no_grad():
            x_val_t = torch.tensor(x[selection_val_idx], dtype=torch.float32, device=device)
            baseline_prediction = model.anchor_components(x_val_t)[0].detach().cpu().numpy()
        baseline_trait_pearsons = _directional_trait_pearsons(baseline_prediction, y[selection_val_idx], mask[selection_val_idx], trait_names)
        baseline_values = [value for value in baseline_trait_pearsons.values() if value is not None]
        if baseline_values:
            baseline_score = float(np.mean(baseline_values))
    best_state = _directional_transfer_state(model)
    best_score = baseline_score
    best_epoch = 0
    wait = 0
    epochs_trained = 0
    final_loss = 0.0
    final_pseudo_loss: float | None = None
    final_effective_pseudo_weight = 0.0
    maximum_effective_pseudo_weight = 0.0
    pseudo_active_epochs = 0
    lr_history = [transfer_lr]
    torch.manual_seed(_training_seed(515109))
    for epoch in range(1, transfer_epochs + 1):
        epochs_trained = epoch
        model.train()
        model.set_anchors_eval()
        epoch_losses: list[float] = []
        pseudo_ramp = min(1.0, max(0, epoch - PDAE_PSEUDO_WARMUP_EPOCHS) / max(PDAE_PSEUDO_RAMP_EPOCHS, 1)) if pdae_training_enabled else 0.0
        if pseudo_ramp > 0:
            pseudo_active_epochs += 1
        epoch_pseudo_losses: list[float] = []
        epoch_effective_weights: list[float] = []
        for xb, yb, mb, pseudo_target_b, pseudo_weight_b in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            pseudo_target_b = pseudo_target_b.to(device)
            pseudo_weight_b = pseudo_weight_b.to(device)
            optimizer.zero_grad()
            prediction, anchor_prediction, _correction, _contributions = model.forward_components(xb)
            observed_loss = masked_trait_balanced_mse_loss(prediction, yb, mb)
            preservation_loss = masked_mse_loss(prediction, anchor_prediction.detach(), mb)
            gates = model.directional_gates()
            active_gates = gates[model.directional_off_diagonal > 0.5]
            gate_penalty = active_gates.mean() if active_gates.numel() else gates.new_tensor(0.0)
            loss = observed_loss + DIRECTIONAL_ANCHOR_PRESERVATION_WEIGHT * preservation_loss + DIRECTIONAL_ANCHOR_GATE_PENALTY * gate_penalty
            if pdae_training_enabled and pseudo_ramp > 0:
                pseudo_loss = _weighted_pseudo_label_loss(prediction, pseudo_target_b, pseudo_weight_b)
                observed_reference = float(observed_loss.detach().cpu())
                pseudo_value = float(pseudo_loss.detach().cpu())
                scheduled_weight = pseudo_ramp * float(params['pdae_pseudo_weight'])
                loss_ratio_cap = PDAE_MAX_PSEUDO_TO_OBSERVED_LOSS_RATIO * observed_reference / max(pseudo_value, 1e-08) if pseudo_value > 0 else 0.0
                effective_pseudo_weight = min(scheduled_weight, loss_ratio_cap)
                loss = loss + effective_pseudo_weight * pseudo_loss
                epoch_pseudo_losses.append(pseudo_value)
                epoch_effective_weights.append(float(effective_pseudo_weight))
                maximum_effective_pseudo_weight = max(maximum_effective_pseudo_weight, float(effective_pseudo_weight))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transfer_parameters, max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        final_pseudo_loss = float(np.mean(epoch_pseudo_losses)) if epoch_pseudo_losses else None
        final_effective_pseudo_weight = float(np.mean(epoch_effective_weights)) if epoch_effective_weights else 0.0
        if selection_val_idx is None:
            best_state = _directional_transfer_state(model)
            best_epoch = epoch
            continue
        model.eval()
        model.set_anchors_eval()
        val_prediction = _predict_in_chunks(model, torch.tensor(x[selection_val_idx], dtype=torch.float32)).numpy()
        current_score_value = _mean_masked_pearson(val_prediction, y[selection_val_idx], mask[selection_val_idx])
        current_score = float(current_score_value) if current_score_value is not None else float('-inf')
        scheduler.step(current_score)
        lr_history.append(float(optimizer.param_groups[0]['lr']))
        if current_score > best_score + 1e-05:
            best_score = current_score
            best_state = _directional_transfer_state(model)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= transfer_patience:
                break
    _load_directional_transfer_state(model, best_state)
    model.eval()
    model.set_anchors_eval()
    if val_idx is None:
        val_prediction_np = None
        final_trait_pearsons = {trait: None for trait in trait_names}
    else:
        val_prediction_np = _predict_in_chunks(model, torch.tensor(x[val_idx], dtype=torch.float32)).numpy()
        final_trait_pearsons = _directional_trait_pearsons(val_prediction_np, y[val_idx], mask[val_idx], trait_names)
    with torch.no_grad():
        diagnostics = model.collect_directional_diagnostics(torch.tensor(x[train_idx], dtype=torch.float32, device=device))
    setattr(model, 'directional_anchor_diagnostics', diagnostics)
    setattr(model, 'directional_anchor_training_summary', {'anchor_stage': anchor_summaries, 'transfer_stage': {'requested_epochs': int(transfer_epochs), 'epochs_trained': int(epochs_trained), 'best_epoch': int(best_epoch), 'fallback_to_unmodified_anchors': bool(best_epoch == 0), 'learning_rate': transfer_lr, 'learning_rate_history': lr_history, 'trainable_parameters': int(sum((parameter.numel() for parameter in transfer_parameters))), 'anchor_preservation_weight': DIRECTIONAL_ANCHOR_PRESERVATION_WEIGHT, 'gate_penalty': DIRECTIONAL_ANCHOR_GATE_PENALTY, 'fixed_epoch_full_data_fit': bool(fixed_final_fit), 'final_fit_config': final_fit_config if fixed_final_fit else None, 'pdae_enabled_for_transfer': bool(pdae_training_enabled), 'accepted_pseudo_cells': int(np.sum(pseudo_weights_np > 0)), 'pseudo_active_epochs': int(pseudo_active_epochs), 'final_pseudo_label_loss': final_pseudo_loss, 'final_effective_pseudo_weight': float(final_effective_pseudo_weight), 'maximum_effective_pseudo_weight': float(maximum_effective_pseudo_weight), 'baseline_validation_pearson_by_trait': baseline_trait_pearsons, 'final_validation_pearson_by_trait': final_trait_pearsons, 'baseline_mean_validation_pearson': float(baseline_score) if np.isfinite(baseline_score) else None, 'best_mean_validation_pearson': float(best_score) if np.isfinite(best_score) else None}})
    setattr(model, 'lr_scheduler_summary', {'enabled': True, 'active': selection_val_idx is not None, 'mode': 'plateau', 'monitor': 'pearson', 'initial_lr': transfer_lr, 'final_lr': float(optimizer.param_groups[0]['lr']), 'min_lr_observed': float(min(lr_history)), 'max_lr_observed': float(max(lr_history)), 'configured_min_lr': float(params.get('min_learning_rate', 1e-06)), 'factor': float(params.get('lr_scheduler_factor', 0.5)), 'patience': max(3, min(6, int(params.get('lr_scheduler_patience', 10)))), 'reductions': sum((1 for previous, current in zip(lr_history, lr_history[1:]) if current < previous - 1e-12)), 'steps': max(0, len(lr_history) - 1)})
    pdae_summary = dict(pdae_summary)
    pdae_summary['directional_transfer_usage'] = {'anchors_updated_by_pseudo_labels': False, 'transfer_adapter_updated_by_pseudo_labels': bool(pdae_training_enabled), 'accepted_pseudo_cells': int(np.sum(pseudo_weights_np > 0)), 'pseudo_active_epochs': int(pseudo_active_epochs), 'maximum_effective_pseudo_weight': float(maximum_effective_pseudo_weight), 'pseudo_loss_ratio_cap': float(PDAE_MAX_PSEUDO_TO_OBSERVED_LOSS_RATIO)}
    setattr(model, 'pdae_summary', pdae_summary)
    setattr(model, 'attention_safety', {
        'enabled': True,
        'method': 'learned_attention_fusion_alpha',
        'selection_metric': 'pearson_early_stopping',
    })
    return (model, final_loss, best_epoch, val_prediction_np)


def _fit_model_on_indices(x: np.ndarray, y: np.ndarray, mask: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray | None, trait_names: list[str] | None, allow_missing_phenotype: bool, use_pdae: bool, use_directional_transfer: bool, transfer_mode: str, use_marker_attention: bool, epochs: int, lr: float, patience: int, ppmgs_params: dict[str, object] | None=None, attention_mode: str='none', prior_scores: np.ndarray | None=None, truth_y: np.ndarray | None=None, truth_mask: np.ndarray | None=None):
    if transfer_mode == DIRECTIONAL_ANCHOR_MODE:
        return _fit_directional_anchor_on_indices(x=x, y=y, mask=mask, train_idx=train_idx, val_idx=val_idx, trait_names=trait_names or [f'trait_{idx + 1}' for idx in range(y.shape[1])], attention_mode=attention_mode, prior_scores=prior_scores, epochs=epochs, lr=lr, patience=patience, ppmgs_params=ppmgs_params, use_pdae=use_pdae, truth_y=truth_y, truth_mask=truth_mask)
    torch.manual_seed(_training_seed())
    device = get_torch_device()
    params = _resolve_ppmgs_params(x.shape[1], ppmgs_params, lr=lr)
    attention_mode = _normalize_attention_mode(attention_mode)
    use_marker_attention = attention_mode != 'none'
    model_kwargs = {'marker_count': x.shape[1], 'trait_count': y.shape[1], 'hidden_dim': int(params['hidden_dim']), 'dropout': float(params['dropout']), 'hidden_layers': int(params['hidden_layers']), 'activation': str(params.get('activation', 'relu'))}
    if attention_mode in PRIOR_MARKER_ATTENTION_MODES:
        model = PriorMarkerAttentionGSNet(**model_kwargs, prior_scores=prior_scores, use_prior_reliability_gate=attention_mode in PRIOR_RELIABILITY_ATTENTION_MODES)
    elif allow_missing_phenotype:
        model = GenomeWideGSNet(**model_kwargs)
    else:
        model = GenomeWideGSNet(**model_kwargs)
    model.to(device)
    _set_transfer_mode(model, transfer_mode)
    early_stop_metric = 'pearson'
    pdae_requested = bool(use_pdae)
    pdae_summary: dict[str, object] = {'enabled': False, 'requested': pdae_requested, 'reason': None}
    pseudo_targets_np = np.zeros((len(train_idx), y.shape[1]), dtype=np.float32)
    pseudo_weights_np = np.zeros_like(pseudo_targets_np)
    if use_pdae and (not allow_missing_phenotype):
        pdae_summary['reason'] = 'requires_allow_missing_phenotype'
    elif use_pdae and y.shape[1] < 2:
        pdae_summary['reason'] = 'requires_at_least_two_traits'
    elif use_pdae:
        pseudo_targets_np, pseudo_weights_np, pdae_summary = _pretrain_pdae_teacher(x=x, y=y, mask=mask, train_idx=train_idx, trait_names=trait_names or [f'trait_{idx + 1}' for idx in range(y.shape[1])], params=params, device=device, main_epochs=epochs, prior_scores=prior_scores, truth_y=truth_y, truth_mask=truth_mask)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    base_parameters = trainable_parameters
    learning_rate = float(params['learning_rate'])
    optimizer_groups: list[dict[str, object]] = [{'params': base_parameters, 'lr': learning_rate}]
    optimizer = optim.AdamW(optimizer_groups, weight_decay=float(params['weight_decay']))
    scheduler_mode = str(params.get('lr_scheduler', 'plateau')).strip().lower()
    if scheduler_mode not in LR_SCHEDULER_MODES:
        scheduler_mode = 'plateau'
    scheduler_factor = float(params.get('lr_scheduler_factor', 0.5))
    scheduler_patience = int(params.get('lr_scheduler_patience', 10))
    min_learning_rate = float(params.get('min_learning_rate', 1e-06))
    scheduler = None
    if scheduler_mode == 'plateau' and val_idx is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if early_stop_metric == 'pearson' else 'min', factor=scheduler_factor, patience=scheduler_patience, min_lr=min_learning_rate)
    elif scheduler_mode == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)), eta_min=min_learning_rate)
    lr_history = [float(optimizer.param_groups[0]['lr'])]
    x_train = torch.tensor(x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    mask_train = torch.tensor(mask[train_idx], dtype=torch.float32)
    pseudo_targets_train = torch.tensor(pseudo_targets_np, dtype=torch.float32)
    pseudo_weights_train = torch.tensor(pseudo_weights_np, dtype=torch.float32)
    main_loader_generator = torch.Generator().manual_seed(_training_seed(424200))
    train_loader = DataLoader(TensorDataset(x_train, y_train, mask_train, pseudo_targets_train, pseudo_weights_train), batch_size=min(int(params['batch_size']), x_train.shape[0]), shuffle=True, generator=main_loader_generator)
    best_state = None
    best_val_score = float('-inf') if early_stop_metric == 'pearson' else float('inf')
    best_epoch = 0
    wait = 0
    final_loss = 0.0
    final_pdae_pseudo_loss: float | None = None
    final_observed_loss: float | None = None
    final_weighted_pseudo_contribution: float | None = None
    final_effective_pseudo_weight: float | None = None
    max_effective_pseudo_weight = 0.0
    max_pseudo_to_observed_loss_ratio = 0.0
    best_epoch_effective_pseudo_weight = 0.0
    best_epoch_pseudo_to_observed_loss_ratio = 0.0
    observed_count_values = mask[train_idx].sum(axis=1).astype(int)
    observed_count_unique, observed_count_freq = np.unique(observed_count_values, return_counts=True)
    pdae_observed_trait_count_distribution: dict[str, int] = {str(int(count)): int(freq) for count, freq in zip(observed_count_unique.tolist(), observed_count_freq.tolist())}
    pdae_training_enabled = bool(pdae_summary.get('enabled'))
    pdae_warmup_epochs = min(PDAE_PSEUDO_WARMUP_EPOCHS, max(0, int(epochs) - 1))
    pdae_ramp_epochs = min(PDAE_PSEUDO_RAMP_EPOCHS, max(1, int(epochs) - pdae_warmup_epochs))
    pdae_full_weight_epoch = pdae_warmup_epochs + pdae_ramp_epochs
    pdae_checkpoint_start_epoch = 1
    pdae_min_training_epochs = min(int(epochs), max(PDAE_MIN_MAIN_TRAINING_EPOCHS, pdae_checkpoint_start_epoch)) if pdae_training_enabled else 1
    epochs_trained = 0
    pseudo_active_epochs = 0
    max_pseudo_ramp = 0.0
    attention_safety: dict[str, object] | None = None
    torch.manual_seed(_training_seed(424200))
    for epoch in range(1, epochs + 1):
        epochs_trained = epoch
        if pdae_training_enabled and epoch > pdae_warmup_epochs:
            pseudo_ramp = min(1.0, (epoch - pdae_warmup_epochs) / max(pdae_ramp_epochs, 1))
        else:
            pseudo_ramp = 0.0
        if pseudo_ramp > 0:
            pseudo_active_epochs += 1
            max_pseudo_ramp = max(max_pseudo_ramp, float(pseudo_ramp))
        attention_scale = _attention_runtime_scale_for_epoch(epoch, epochs, warmup_epochs=int(params['attention_warmup_epochs']) if 'attention_warmup_epochs' in params else None, ramp_epochs=int(params['attention_ramp_epochs']) if 'attention_ramp_epochs' in params else None) if attention_mode in ATTENTION_AUXILIARY_MODES else 1.0
        _set_attention_runtime_scale(model, attention_scale)
        _set_prior_dropout_rate(model, 0.15 if attention_mode in PRIOR_REQUIRED_ATTENTION_MODES and attention_scale > 0 else 0.0)
        model.train()
        epoch_loss = 0.0
        epoch_pdae_pseudo_loss = 0.0
        epoch_observed_loss = 0.0
        epoch_weighted_pseudo_contribution = 0.0
        epoch_effective_pseudo_weight = 0.0
        batch_count = 0
        for xb, yb, mb, pseudo_target_b, pseudo_weight_b in train_loader:
            optimizer.zero_grad()
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            pseudo_target_b = pseudo_target_b.to(device)
            pseudo_weight_b = pseudo_weight_b.to(device)
            pred = model(xb)
            observed_loss = masked_trait_balanced_mse_loss(pred, yb, mb)
            pdae_cap_reference_loss = masked_mse_loss(pred, yb, mb) if pdae_training_enabled else observed_loss
            loss = observed_loss
            attention_reg = _attention_regularization(model)
            if attention_reg is not None and attention_scale > 0:
                loss = loss + 0.01 * attention_scale * attention_reg
            if pdae_training_enabled:
                pdae_pseudo_loss = _weighted_pseudo_label_loss(pred, pseudo_target_b, pseudo_weight_b)
                pdae_cap_reference_value = float(pdae_cap_reference_loss.detach().cpu())
                pseudo_loss_value = float(pdae_pseudo_loss.detach().cpu())
                scheduled_weight = pseudo_ramp * float(params['pdae_pseudo_weight'])
                loss_ratio_capped_weight = PDAE_MAX_PSEUDO_TO_OBSERVED_LOSS_RATIO * pdae_cap_reference_value / max(pseudo_loss_value, 1e-08) if pseudo_loss_value > 0 else 0.0
                effective_pseudo_weight = min(scheduled_weight, loss_ratio_capped_weight)
                weighted_pseudo_contribution = effective_pseudo_weight * pdae_pseudo_loss
                loss = loss + weighted_pseudo_contribution
                epoch_pdae_pseudo_loss += float(pdae_pseudo_loss.detach().cpu())
                epoch_weighted_pseudo_contribution += float(weighted_pseudo_contribution.detach().cpu())
                epoch_effective_pseudo_weight += float(effective_pseudo_weight)
                max_effective_pseudo_weight = max(max_effective_pseudo_weight, float(effective_pseudo_weight))
                if pdae_cap_reference_value > 0:
                    max_pseudo_to_observed_loss_ratio = max(max_pseudo_to_observed_loss_ratio, float(effective_pseudo_weight * pseudo_loss_value / pdae_cap_reference_value))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=3.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_observed_loss += float(observed_loss.detach().cpu())
            batch_count += 1
        final_loss = epoch_loss / max(batch_count, 1)
        final_observed_loss = epoch_observed_loss / max(batch_count, 1)
        if pdae_training_enabled:
            final_pdae_pseudo_loss = epoch_pdae_pseudo_loss / max(batch_count, 1)
            final_weighted_pseudo_contribution = epoch_weighted_pseudo_contribution / max(batch_count, 1)
            final_effective_pseudo_weight = epoch_effective_pseudo_weight / max(batch_count, 1)
        epoch_pseudo_to_observed_loss_ratio = float(final_weighted_pseudo_contribution / final_observed_loss) if final_weighted_pseudo_contribution is not None and final_observed_loss is not None and (final_observed_loss > 0) else 0.0
        if val_idx is None:
            if scheduler is not None and scheduler_mode == 'cosine':
                scheduler.step()
                lr_history.append(float(optimizer.param_groups[0]['lr']))
            best_epoch = epoch
            continue
        _set_prior_dropout_rate(model, 0.0)
        model.eval()
        x_val = torch.tensor(x[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32)
        mask_val = torch.tensor(mask[val_idx], dtype=torch.float32)
        val_pred = _predict_in_chunks(model, x_val)
        val_loss = float(masked_mse_loss(val_pred, y_val, mask_val).detach().cpu())
        if early_stop_metric == 'pearson':
            val_score = _mean_masked_pearson(val_pred.detach().cpu().numpy(), y_val.detach().cpu().numpy(), mask_val.detach().cpu().numpy())
            current_score = float(val_score) if val_score is not None else -val_loss
            improved = current_score > best_val_score + 1e-05
        else:
            current_score = val_loss
            improved = current_score < best_val_score - 1e-05
        if scheduler is not None:
            if scheduler_mode == 'plateau':
                scheduler.step(current_score)
            else:
                scheduler.step()
            lr_history.append(float(optimizer.param_groups[0]['lr']))
        checkpoint_eligible = epoch >= pdae_checkpoint_start_epoch
        if checkpoint_eligible:
            if improved:
                best_val_score = current_score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_epoch_effective_pseudo_weight = float(final_effective_pseudo_weight or 0.0)
                best_epoch_pseudo_to_observed_loss_ratio = float(epoch_pseudo_to_observed_loss_ratio)
                wait = 0
            else:
                wait += 1
                if epoch >= pdae_min_training_epochs and wait >= patience:
                    break
        else:
            wait = 0
    if best_state is not None:
        model.load_state_dict(best_state)
    final_lr = float(optimizer.param_groups[0]['lr'])
    lr_reductions = sum((1 for previous, current in zip(lr_history, lr_history[1:]) if current < previous - 1e-12))
    setattr(model, 'lr_scheduler_summary', {'enabled': scheduler_mode != 'none', 'active': scheduler is not None, 'mode': scheduler_mode, 'monitor': early_stop_metric, 'initial_lr': float(params['learning_rate']), 'final_lr': final_lr, 'min_lr_observed': float(min(lr_history)) if lr_history else final_lr, 'max_lr_observed': float(max(lr_history)) if lr_history else final_lr, 'configured_min_lr': min_learning_rate, 'factor': scheduler_factor if scheduler_mode == 'plateau' else None, 'patience': scheduler_patience if scheduler_mode == 'plateau' else None, 'reductions': int(lr_reductions), 'steps': max(0, len(lr_history) - 1)})
    _set_attention_runtime_scale(model, 1.0)
    _set_prior_dropout_rate(model, 0.0)
    model.eval()
    if val_idx is None:
        val_pred_np = None
    else:
        if attention_mode in ATTENTION_AUXILIARY_MODES:
            attention_safety = {
                'enabled': True,
                'method': 'learned_attention_fusion_alpha',
                'selection_metric': 'pearson_early_stopping',
            }
        val_pred_np = _predict_in_chunks(model, torch.tensor(x[val_idx], dtype=torch.float32)).cpu().numpy()
    if pdae_requested:
        if pdae_training_enabled and best_epoch > pdae_warmup_epochs:
            best_pseudo_ramp = min(1.0, (best_epoch - pdae_warmup_epochs) / max(pdae_ramp_epochs, 1))
        else:
            best_pseudo_ramp = 0.0
        pdae_summary.update({'main_model_optimizer_includes_pdae': False, 'teacher_frozen_during_main_training': True, 'pseudo_schedule_applied': pdae_training_enabled, 'pseudo_warmup_epochs': int(pdae_warmup_epochs), 'pseudo_ramp_epochs': int(pdae_ramp_epochs), 'pseudo_full_weight_epoch': int(pdae_full_weight_epoch), 'minimum_main_training_epochs': int(pdae_min_training_epochs), 'checkpoint_eligible_epoch': int(pdae_checkpoint_start_epoch) if pdae_training_enabled else None, 'epochs_trained': int(epochs_trained), 'pseudo_active_epochs': int(pseudo_active_epochs), 'maximum_pseudo_ramp': float(max_pseudo_ramp), 'best_epoch_after_pseudo_started': bool(pdae_training_enabled and best_epoch > pdae_warmup_epochs), 'best_epoch_at_full_pseudo_weight': bool(pdae_training_enabled and best_epoch >= pdae_full_weight_epoch), 'best_scheduled_pseudo_weight': float(best_pseudo_ramp * float(params['pdae_pseudo_weight'])), 'best_effective_pseudo_weight': float(best_epoch_effective_pseudo_weight), 'best_pseudo_to_observed_loss_ratio': float(best_epoch_pseudo_to_observed_loss_ratio), 'pseudo_loss_ratio_cap': float(PDAE_MAX_PSEUDO_TO_OBSERVED_LOSS_RATIO), 'maximum_effective_pseudo_weight': float(max_effective_pseudo_weight), 'maximum_pseudo_to_observed_loss_ratio': float(max_pseudo_to_observed_loss_ratio), 'final_observed_loss': final_observed_loss, 'final_pseudo_label_loss': final_pdae_pseudo_loss, 'final_weighted_pseudo_contribution': final_weighted_pseudo_contribution, 'final_effective_pseudo_weight': final_effective_pseudo_weight, 'observed_trait_count_distribution': dict(sorted(pdae_observed_trait_count_distribution.items(), key=lambda item: int(item[0])))})
    setattr(model, 'pdae_summary', pdae_summary)
    setattr(model, 'attention_safety', attention_safety)
    return (model, final_loss, best_epoch, val_pred_np)


def _cross_validate(x: np.ndarray, y: np.ndarray, mask: np.ndarray, truth_y: np.ndarray | None, truth_mask: np.ndarray | None, trait_names: list[str], y_mean: np.ndarray, y_std: np.ndarray, allow_missing_phenotype: bool, use_pdae: bool, use_directional_transfer: bool, transfer_mode: str, use_marker_attention: bool, epochs: int, lr: float, patience: int, folds: int, repeats: int=10, ppmgs_params: dict[str, object] | None=None, attention_mode: str='none', prior_scores: np.ndarray | None=None, save_oof_intervals: bool=False, sample_ids: list[str] | None=None, output_dir: Path | None=None, foldwise_prior_builder: _FoldwisePriorBuilder | None=None) -> dict[str, object]:
    repeats = max(1, int(repeats))
    folds = max(2, int(folds))
    fold_results = []
    fold_metrics = []
    hidden_truth_fold_metrics: list[dict[str, dict[str, float | None]]] = []
    all_truth_fold_metrics: list[dict[str, dict[str, float | None]]] = []
    individualized_pairs: list[list[tuple[float, float]]] = [[] for _ in trait_names]
    sample_ids = sample_ids or [str(idx) for idx in range(x.shape[0])]
    oof_chunks: list[dict[str, object]] = []
    for repeat_number in range(1, repeats + 1):
        fold_seed = _training_seed(repeat_number - 1)
        splits = _missing_pattern_stratified_kfold_splits(mask, folds=folds, seed=fold_seed) if allow_missing_phenotype and y.shape[1] >= 2 else _kfold_splits(x.shape[0], folds=folds, seed=fold_seed)
        for fold_number, (train_idx, val_idx) in enumerate(splits, start=1):
            fold_prior_summary = None
            fold_prior_scores = prior_scores
            if foldwise_prior_builder is not None:
                fold_prior_scores, fold_prior_summary = foldwise_prior_builder.build(train_idx, stage='formal_cross_validation', repeat_number=repeat_number, fold_number=fold_number, params=ppmgs_params)
            fold_model, final_loss, best_epoch, val_pred_np = _fit_model_on_indices(x=x, y=y, mask=mask, train_idx=train_idx, val_idx=val_idx, trait_names=trait_names, allow_missing_phenotype=allow_missing_phenotype, use_pdae=use_pdae, use_directional_transfer=use_directional_transfer, transfer_mode=transfer_mode, use_marker_attention=use_marker_attention, attention_mode=attention_mode, prior_scores=fold_prior_scores, epochs=epochs, lr=lr, patience=patience, ppmgs_params=ppmgs_params, truth_y=truth_y, truth_mask=truth_mask)
            metrics = _validation_metrics(val_pred_np, y[val_idx], mask[val_idx], trait_names, y_mean, y_std)
            hidden_truth_metrics = None
            all_truth_metrics = None
            if truth_y is not None and truth_mask is not None:
                hidden_truth_mask = ((truth_mask[val_idx] > 0.5) & (mask[val_idx] < 0.5)).astype(np.float32)
                hidden_truth_metrics = _validation_metrics(val_pred_np, truth_y[val_idx], hidden_truth_mask, trait_names, y_mean, y_std)
                all_truth_metrics = _validation_metrics(val_pred_np, truth_y[val_idx], truth_mask[val_idx], trait_names, y_mean, y_std)
                hidden_truth_fold_metrics.append(hidden_truth_metrics)
                all_truth_fold_metrics.append(all_truth_metrics)
            val_mc_mean_np, val_uncertainty_np = _mc_dropout_predict_scaled(fold_model, x[val_idx], passes=INDIVIDUALIZED_CONFORMAL_MC_PASSES)
            _append_individualized_conformal_pairs(individualized_pairs, val_mc_mean_np, y[val_idx], val_uncertainty_np, mask[val_idx], y_mean, y_std)
            if save_oof_intervals:
                oof_chunks.append({'repeat': repeat_number, 'fold': fold_number, 'train_idx': train_idx, 'val_idx': val_idx, 'val_pred_np': val_pred_np, 'val_mc_mean_np': val_mc_mean_np, 'val_uncertainty_np': val_uncertainty_np})
            fold_metrics.append(metrics)
            fold_pdae_summary = getattr(fold_model, 'pdae_summary', None)
            fold_model_summary = fold_model.model_summary() if hasattr(fold_model, 'model_summary') else None
            fold_directional_transfer_summary = _directional_transfer_summary(fold_model, trait_names, observed_correlation_matrix(y[train_idx], mask[train_idx]) if len(trait_names) >= 2 else None)
            fold_results.append({'repeat': repeat_number, 'fold': fold_number, 'train_samples': int(len(train_idx)), 'validation_samples': int(len(val_idx)), 'best_epoch': int(best_epoch), 'final_loss': float(final_loss), 'metrics': metrics, 'hidden_truth_metrics': hidden_truth_metrics, 'all_truth_metrics': all_truth_metrics, 'pdae_summary': fold_pdae_summary, 'model_summary': fold_model_summary, 'directional_transfer_summary': fold_directional_transfer_summary, 'foldwise_prior': fold_prior_summary})
    interval_calibration = _individualized_conformal_summary_from_pairs(individualized_pairs, trait_names, confidence=0.95, source='repeated_kfold_out_of_fold_mc_dropout_scaled_residuals')
    oof_prediction_intervals: dict[str, object] = {'enabled': False, 'reason': 'save_oof_intervals_not_requested'}
    oof_interval_summary: dict[str, object] = {'enabled': False, 'reason': 'not_requested'}
    if save_oof_intervals:
        oof_rows: list[dict[str, object]] = []
        for chunk in oof_chunks:
            val_idx = np.asarray(chunk['val_idx'], dtype=np.int64)
            train_idx = np.asarray(chunk['train_idx'], dtype=np.int64)
            oof_rows.extend(_oof_interval_rows_for_fold(repeat_number=int(chunk['repeat']), fold_number=int(chunk['fold']), sample_ids=sample_ids, val_idx=val_idx, y_scaled=y[val_idx], mask=mask[val_idx], trait_names=trait_names, y_mean=y_mean, y_std=y_std, interval_summary=interval_calibration, mc_mean_scaled=np.asarray(chunk['val_mc_mean_np'], dtype=np.float32), mc_uncertainty_scaled=np.asarray(chunk['val_uncertainty_np'], dtype=np.float32), model_train_samples=len(train_idx), validation_samples=len(val_idx)))
        oof_interval_summary = _summarize_oof_interval_rows(oof_rows, trait_names, prediction_key='prediction', lower_key='lower_95', upper_key='upper_95', radius_key='radius', covered_key='covered')
        oof_prediction_files = _write_oof_prediction_tables(oof_rows, output_dir)
        oof_prediction_intervals = {'enabled': bool(oof_rows), 'method': 'repeated_kfold_oof_mc_dropout_scaled_conformal_intervals', 'folds': folds, 'repeats': repeats, 'interval_summary': oof_interval_summary, 'files': oof_prediction_files, 'row_definition': 'one row per repeat/fold/validation sample/trait', 'note': 'This export preserves the original repeated K-fold training split. It does not use an inner calibration split.'}
    return {'folds': folds, 'repeats': repeats, 'total_runs': len(fold_results), 'split_strategy': 'missing_pattern_stratified_when_possible' if allow_missing_phenotype and y.shape[1] >= 2 else 'random_kfold', 'fold_results': fold_results, 'foldwise_prior': {'enabled': bool(foldwise_prior_builder is not None), 'scope': 'outer_training_fold_only' if foldwise_prior_builder is not None else None, 'validation_phenotypes_used': False if foldwise_prior_builder is not None else None, 'tassel_mlm_rebuilt_per_fold': bool(foldwise_prior_builder is not None and foldwise_prior_builder.build_tassel_prior), 'lasso_rebuilt_per_fold': bool(foldwise_prior_builder is not None and foldwise_prior_builder.build_lasso_prior)}, 'summary': _summarize_cv_metrics(fold_metrics), 'hidden_truth_summary': {'enabled': True, 'definition': 'truth cells present in phenotype-truth CSV but hidden in the training phenotype CSV', 'summary': _summarize_cv_metrics(hidden_truth_fold_metrics)} if truth_y is not None and truth_mask is not None else {'enabled': False, 'reason': 'phenotype_truth_not_provided'}, 'all_truth_summary': {'enabled': True, 'definition': 'all finite cells in phenotype-truth CSV', 'summary': _summarize_cv_metrics(all_truth_fold_metrics)} if truth_y is not None and truth_mask is not None else {'enabled': False, 'reason': 'phenotype_truth_not_provided'}, 'prediction_interval_calibration': interval_calibration, 'oof_interval_summary': oof_interval_summary, 'oof_prediction_intervals': oof_prediction_intervals}


def _directional_anchor_final_fit_config(cross_validation: dict[str, object] | None, trait_names: list[str], use_pdae: bool) -> dict[str, object]:
    fold_results = cross_validation.get('fold_results', []) if isinstance(cross_validation, dict) else []
    anchor_epochs: dict[str, list[int]] = {trait: [] for trait in trait_names}
    transfer_epochs: list[int] = []
    total_folds = 0
    active_folds = 0
    for fold_result in fold_results:
        interaction = fold_result.get('directional_transfer_summary') or {}
        if interaction.get('mode') != DIRECTIONAL_ANCHOR_MODE:
            continue
        stage = interaction.get('stage_training') or {}
        for anchor in stage.get('anchor_stage', []):
            trait = str(anchor.get('trait', ''))
            epoch = int(anchor.get('best_epoch', 0) or 0)
            if trait in anchor_epochs and epoch > 0:
                anchor_epochs[trait].append(epoch)
        transfer = stage.get('transfer_stage') or {}
        best_epoch = int(transfer.get('best_epoch', 0) or 0)
        total_folds += 1
        if best_epoch > 0:
            active_folds += 1
            transfer_epochs.append(best_epoch)
    anchor_epochs_by_trait = {trait: max(1, int(round(float(np.median(values))))) for trait, values in anchor_epochs.items() if values}
    active_fraction = float(active_folds / max(total_folds, 1))
    transfer_is_supported = bool(transfer_epochs and active_fraction >= DIRECTIONAL_ANCHOR_MIN_ACTIVE_FOLD_FRACTION)
    selected_transfer_epochs = max(1, int(round(float(np.median(transfer_epochs))))) if transfer_is_supported else 0
    if use_pdae and selected_transfer_epochs > 0:
        selected_transfer_epochs = max(selected_transfer_epochs, PDAE_PSEUDO_WARMUP_EPOCHS + PDAE_PSEUDO_RAMP_EPOCHS)
    return {'enabled': bool(anchor_epochs_by_trait), 'method': 'median_cv_epoch_full_data_refit', 'anchor_epochs_by_trait': anchor_epochs_by_trait, 'transfer_epochs': int(selected_transfer_epochs), 'active_transfer_folds': int(active_folds), 'total_folds': int(total_folds), 'active_transfer_fraction': active_fraction, 'minimum_active_fold_fraction': DIRECTIONAL_ANCHOR_MIN_ACTIVE_FOLD_FRACTION, 'pdae_enabled_for_transfer': bool(use_pdae), 'transfer_epoch_distribution': {'median_active': float(np.median(transfer_epochs)) if transfer_epochs else None, 'minimum_active': int(min(transfer_epochs)) if transfer_epochs else None, 'maximum_active': int(max(transfer_epochs)) if transfer_epochs else None}, 'note': 'Epochs are selected from repeated-CV fold summaries and then used for fixed-epoch refitting on all available training samples.'}


def train_model(genotype_file, phenotype_file, prior_marker_file=None, epochs: int=220, lr: float=0.001, task_type: str | None=None, allow_missing_phenotype: bool | None=None, attention_mode: str | None=None, trait_name: str | None=None, trait_names: list[str] | None=None, patience: int=18, use_cross_validation: bool=False, cv_folds: int=5, cv_repeats: int=10, save_oof_intervals: bool=False, use_pdae: bool=False, use_directional_transfer: bool | None=None, transfer_mode: str | None=None, pdae_mask_rate: float=0.3, pdae_loss_weight: float=0.15, pdae_pseudo_weight: float=0.01, build_lasso_prior: bool=False, lasso_prior_gwas_weight: float=0.5, lasso_prior_repeats: int=50, lasso_prior_cache_dir: str | Path | None=None, prior_sparsity: str='top_1pct', build_tassel_prior: bool=False, tassel_pipeline_path: str | None=None, tassel_pc_count: int=3, optimize_hyperparameters: bool=False, hyperparameter_trials: int=100, hyperparameter_folds: int=3, hyperparameter_metric: str='pearson', hyperparameter_method: str='tpe', hyperparameter_early_stop_rounds: int | None=20, hyperparameter_max_epochs: int | None=None, lr_scheduler: str='plateau', lr_scheduler_factor: float=0.5, lr_scheduler_patience: int=10, min_learning_rate: float=1e-06, ppmgs_params_override: dict[str, object] | None=None, compute_shap: bool=False, shap_top_k: int=50, save_full_shap: bool=False, phenotype_truth_file=None) -> tuple[str, TrainedJob, int]:
    compute_shap = bool(compute_shap or save_full_shap)
    prior_sparsity = _normalize_prior_sparsity(prior_sparsity)
    lr_scheduler = str(lr_scheduler or 'plateau').strip().lower()
    if lr_scheduler not in LR_SCHEDULER_MODES:
        lr_scheduler = 'plateau'
    scheduler_params = {'lr_scheduler': lr_scheduler, 'lr_scheduler_factor': lr_scheduler_factor, 'lr_scheduler_patience': lr_scheduler_patience, 'min_learning_rate': min_learning_rate}
    timing_start = time.perf_counter()
    timing_sections: dict[str, float] = {}

    def _mark_timing(name: str, started_at: float) -> float:
        elapsed = round(float(time.perf_counter() - started_at), 4)
        timing_sections[f'{name}_seconds'] = elapsed
        return elapsed
    task_type, allow_missing_phenotype, use_marker_attention, attention_mode = _resolve_training_options(
        task_type=task_type,
        allow_missing_phenotype=allow_missing_phenotype,
        attention_mode=attention_mode,
    )
    if use_directional_transfer is None:
        use_directional_transfer = bool(
            task_type == 'multi_trait' and attention_mode in PRIOR_MARKER_ATTENTION_MODES
        )
    transfer_mode = _normalize_transfer_mode(transfer_mode, use_directional_transfer)
    use_directional_transfer = transfer_mode != 'none'
    if task_type == 'single_trait' and use_directional_transfer:
        raise ValueError('Directional transfer is available only for multi-trait training.')
    if use_directional_transfer and attention_mode not in PRIOR_MARKER_ATTENTION_MODES:
        raise ValueError('Directional transfer requires trait-specific prior-aware anchors.')
    if use_pdae and task_type != 'multi_trait':
        raise ValueError('PDAE supervision is available only for multi-trait training.')
    pdae_requested = bool(use_pdae)
    use_pdae = bool(use_pdae and allow_missing_phenotype)
    data_preparation_start = time.perf_counter()
    x, y, mask, marker_names, trait_names, x_mean, x_std, y_mean, y_std, marker_fill_values, phenotype_missing_summary, sample_ids = prepare_training_data(genotype_file, phenotype_file, task_type=task_type, allow_missing_phenotype=allow_missing_phenotype, trait_name=trait_name, trait_names=trait_names, retain_all_missing=bool(phenotype_truth_file is not None and allow_missing_phenotype))
    truth_y: np.ndarray | None = None
    truth_mask: np.ndarray | None = None
    truth_summary: dict[str, object] = {'provided': False}
    if phenotype_truth_file is not None:
        truth_y, truth_mask, truth_summary = _prepare_truth_phenotypes(phenotype_truth_file, sample_ids, trait_names, y_mean, y_std)
    phenotype_missing_summary = dict(phenotype_missing_summary)
    phenotype_missing_summary['truth_evaluation'] = truth_summary
    phenotype_missing_summary['hidden_truth_values'] = int(np.sum((truth_mask > 0.5) & (mask < 0.5))) if truth_mask is not None else None
    has_missing_training_targets = bool(np.any(mask < 0.5))
    use_pdae = bool(use_pdae and has_missing_training_targets)
    _mark_timing('data_preparation', data_preparation_start)
    job_id = uuid4().hex[:12]
    job_dir = SAVED_MODELS_DIR / job_id
    prior_scores = None
    prior_marker_summary = None
    generated_prior_table: pd.DataFrame | None = None
    prior_component_scores: dict[str, np.ndarray] | None = None
    has_gwas_prior_scores = False
    prior_compatible_attention = attention_mode in PRIOR_COMPATIBLE_ATTENTION_MODES
    if build_tassel_prior and (not prior_compatible_attention):
        raise ValueError('TASSEL-MLM GWAS prior requires a prior-compatible PPMGS-Net attention branch.')
    prior_total_start = time.perf_counter()
    if prior_compatible_attention:
        if attention_mode in PRIOR_REQUIRED_ATTENTION_MODES and prior_marker_file is None and (not build_lasso_prior) and (not build_tassel_prior):
            raise ValueError('Prior-informed PPMGS-Net branches require an SNP-Marker file, LASSO prior construction, or TASSEL-MLM GWAS prior construction.')
        gwas_sources: list[tuple[str, np.ndarray, dict[str, object] | None]] = []
        if prior_marker_file is not None and (not build_tassel_prior):
            uploaded_prior_start = time.perf_counter()
            uploaded_scores, uploaded_summary = _read_prior_marker_file(prior_marker_file, marker_names, trait_names, prior_sparsity=prior_sparsity)
            _mark_timing('uploaded_prior_parse', uploaded_prior_start)
            gwas_sources.append(('uploaded_snp_marker', uploaded_scores, uploaded_summary))
        if build_tassel_prior:
            tassel_prior_start = time.perf_counter()
            x_raw_for_gwas = x * x_std + x_mean
            y_raw_for_gwas = y * y_std + y_mean
            y_raw_for_gwas = np.where(mask > 0, y_raw_for_gwas, np.nan)
            tassel_result = build_tassel_mlm_prior(x_raw=x_raw_for_gwas, y_raw=y_raw_for_gwas, y_mask=mask, sample_ids=sample_ids, marker_names=marker_names, trait_names=trait_names, output_dir=job_dir / 'tassel_mlm_gwas', tassel_pipeline_path=tassel_pipeline_path, pc_count=tassel_pc_count, prior_sparsity=prior_sparsity)
            _mark_timing('tassel_prior', tassel_prior_start)
            gwas_sources.append(('tassel_mlm_gwas', tassel_result.prior_scores, tassel_result.summary))
        prior_combine_start = time.perf_counter()
        gwas_prior_scores, gwas_summary = _combine_prior_sources(gwas_sources, trait_names)
        has_gwas_prior_scores = bool(gwas_prior_scores is not None and np.any(np.asarray(gwas_prior_scores) > 0))
        _mark_timing('prior_source_combination', prior_combine_start)
        if build_lasso_prior:
            lasso_prior_start = time.perf_counter()
            prior_scores, lasso_prior_summary, generated_prior_table = _build_lasso_gwas_prior(x=x, y=y, mask=mask, marker_names=marker_names, trait_names=trait_names, gwas_prior_scores=gwas_prior_scores, gwas_weight=lasso_prior_gwas_weight, repeats=lasso_prior_repeats, prior_sparsity=prior_sparsity, lasso_checkpoint_dir=Path(lasso_prior_cache_dir).expanduser().resolve() if lasso_prior_cache_dir is not None else None)
            prior_component_scores = _prior_components_from_generated_table(generated_prior_table, trait_names)
            _mark_timing('lasso_prior', lasso_prior_start)
            prior_marker_summary = {
                'format': 'generated_lasso_gwas_prior',
                'input_gwas_prior': gwas_summary,
                'generated_prior': lasso_prior_summary,
                'used_by_attention_mode': attention_mode,
                'attention_fusion': 'learned_attention_fusion_alpha',
            }
        else:
            prior_scores = gwas_prior_scores
            prior_marker_summary = gwas_summary
            if prior_marker_summary is not None:
                prior_marker_summary = dict(prior_marker_summary)
                prior_marker_summary['used_by_attention_mode'] = attention_mode
                prior_marker_summary['attention_fusion'] = 'learned_attention_fusion_alpha'
        _mark_timing('prior_total', prior_total_start)
    else:
        timing_sections['prior_total_seconds'] = 0.0
    foldwise_prior_builder: _FoldwisePriorBuilder | None = None
    if use_cross_validation and prior_compatible_attention and build_tassel_prior:
        foldwise_prior_builder = _FoldwisePriorBuilder(x=x, y=y, mask=mask, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, sample_ids=sample_ids, marker_names=marker_names, trait_names=trait_names, output_root=job_dir / 'foldwise_priors', build_tassel_prior=True, build_lasso_prior=bool(build_lasso_prior), lasso_gwas_weight=lasso_prior_gwas_weight, lasso_repeats=lasso_prior_repeats, prior_sparsity=prior_sparsity, tassel_pipeline_path=tassel_pipeline_path, tassel_pc_count=tassel_pc_count)
        prior_marker_summary = dict(prior_marker_summary or {})
        prior_marker_summary['cross_validation_prior_protocol'] = {'enabled': True, 'scope': 'training_fold_only', 'tassel_mlm_gwas_rebuilt_per_fold': True, 'lasso_stability_rebuilt_per_fold': bool(build_lasso_prior), 'validation_phenotypes_used': False, 'uploaded_prior_used_for_evaluation': False, 'final_model_prior': 'rebuilt_on_all_available_training_samples_after_evaluation'}
        if prior_marker_file is not None:
            prior_marker_summary['uploaded_prior_file'] = {'provided': True, 'used': False, 'reason': 'TASSEL prior construction was enabled; fold-wise TASSEL GWAS takes precedence.'}
    ppmgs_params: dict[str, object] | None = None
    hyperparameter_search = None
    if optimize_hyperparameters:
        hyperparameter_start = time.perf_counter()
        ppmgs_params, hyperparameter_search = _optimize_ppmgs_hyperparameters(x=x, y=y, mask=mask, trait_names=trait_names, y_mean=y_mean, y_std=y_std, allow_missing_phenotype=allow_missing_phenotype, use_pdae=use_pdae, use_directional_transfer=use_directional_transfer, transfer_mode=transfer_mode, use_marker_attention=use_marker_attention, attention_mode=attention_mode, prior_scores=prior_scores, prior_component_scores=prior_component_scores, prior_sparsity=prior_sparsity, tune_prior_gwas_weight=bool(build_lasso_prior and (foldwise_prior_builder is not None or (prior_component_scores is not None and has_gwas_prior_scores))), epochs=epochs, lr=lr, patience=patience, trials=hyperparameter_trials, tuning_folds=hyperparameter_folds, metric=hyperparameter_metric, method=hyperparameter_method, early_stop_rounds=hyperparameter_early_stop_rounds, max_tuning_epochs=hyperparameter_max_epochs, scheduler_params=scheduler_params, foldwise_prior_builder=foldwise_prior_builder)
        if isinstance(hyperparameter_search, dict):
            hyperparameter_search = dict(hyperparameter_search)
            hyperparameter_search['lr_scheduler'] = {'mode': ppmgs_params.get('lr_scheduler'), 'factor': ppmgs_params.get('lr_scheduler_factor'), 'patience': ppmgs_params.get('lr_scheduler_patience'), 'min_learning_rate': ppmgs_params.get('min_learning_rate')}
        _mark_timing('hyperparameter_search', hyperparameter_start)
    elif ppmgs_params_override is not None:
        ppmgs_params = _resolve_ppmgs_params(x.shape[1], {**ppmgs_params_override, **scheduler_params}, lr=lr)
        hyperparameter_search = {'enabled': False, 'method': 'fixed_ppmgs_params', 'source': 'ppmgs_params_override', 'best_params': ppmgs_params, 'note': 'PPMGS-Net hyperparameters were loaded from a previous run and reused without a new search.'}
    else:
        ppmgs_params = _resolve_ppmgs_params(x.shape[1], {'learning_rate': lr, 'pdae_mask_rate': pdae_mask_rate, 'pdae_loss_weight': pdae_loss_weight, 'pdae_pseudo_weight': pdae_pseudo_weight, **scheduler_params, **({'lasso_prior_gwas_weight': lasso_prior_gwas_weight} if build_lasso_prior else {})}, lr=lr)
    if prior_component_scores is not None and ppmgs_params is not None:
        prior_scores = _prior_scores_from_components(prior_scores, prior_component_scores, ppmgs_params, prior_sparsity=prior_sparsity)
        selected_gwas_weight = ppmgs_params.get('lasso_prior_gwas_weight')
        if selected_gwas_weight is not None and prior_marker_summary is not None:
            prior_marker_summary = dict(prior_marker_summary)
            prior_marker_summary['hyperopt_selected_gwas_weight'] = float(selected_gwas_weight)
            prior_marker_summary['hyperopt_selected_lasso_weight'] = float(1.0 - float(selected_gwas_weight))
            prior_marker_summary['prior_scores_recomputed_from_selected_weight'] = True
    if 'hyperparameter_search_seconds' not in timing_sections:
        timing_sections['hyperparameter_search_seconds'] = 0.0
    cross_validation = None
    if use_cross_validation:
        cross_validation_start = time.perf_counter()
        cross_validation = _cross_validate(x=x, y=y, mask=mask, truth_y=truth_y, truth_mask=truth_mask, trait_names=trait_names, y_mean=y_mean, y_std=y_std, allow_missing_phenotype=allow_missing_phenotype, use_pdae=use_pdae, use_directional_transfer=use_directional_transfer, transfer_mode=transfer_mode, use_marker_attention=use_marker_attention, attention_mode=attention_mode, epochs=epochs, lr=lr, patience=patience, folds=cv_folds, repeats=cv_repeats, ppmgs_params=ppmgs_params, prior_scores=prior_scores, save_oof_intervals=bool(save_oof_intervals), sample_ids=sample_ids, output_dir=job_dir if save_oof_intervals else None, foldwise_prior_builder=foldwise_prior_builder)
        _mark_timing('cross_validation', cross_validation_start)
        if transfer_mode == DIRECTIONAL_ANCHOR_MODE:
            final_fit_config = _directional_anchor_final_fit_config(cross_validation, trait_names, use_pdae=use_pdae)
            ppmgs_params = dict(ppmgs_params or {})
            ppmgs_params['directional_anchor_final_fit'] = final_fit_config
            cross_validation['directional_anchor_final_fit'] = final_fit_config
    else:
        timing_sections['cross_validation_seconds'] = 0.0
    if use_cross_validation:
        final_train_idx = np.arange(x.shape[0])
        val_idx_for_metrics = np.arange(x.shape[0])
    else:
        final_train_idx, val_idx_for_metrics = _train_validation_split(x.shape[0])
    final_training_start = time.perf_counter()
    model, final_loss, best_epoch, val_pred_np = _fit_model_on_indices(x=x, y=y, mask=mask, train_idx=final_train_idx, val_idx=val_idx_for_metrics, trait_names=trait_names, allow_missing_phenotype=allow_missing_phenotype, use_pdae=use_pdae, use_directional_transfer=use_directional_transfer, transfer_mode=transfer_mode, use_marker_attention=use_marker_attention, attention_mode=attention_mode, epochs=epochs, lr=lr, patience=patience, ppmgs_params=ppmgs_params, prior_scores=prior_scores, truth_y=truth_y, truth_mask=truth_mask)
    _mark_timing('final_training', final_training_start)
    post_training_start = time.perf_counter()
    metrics = _validation_metrics(val_pred_np, y[val_idx_for_metrics], mask[val_idx_for_metrics], trait_names, y_mean, y_std)
    ppmgs_params = dict(ppmgs_params or {})
    lr_scheduler_summary = getattr(model, 'lr_scheduler_summary', None)
    if isinstance(lr_scheduler_summary, dict):
        ppmgs_params['lr_scheduler_summary'] = lr_scheduler_summary
    if cross_validation is not None and isinstance(cross_validation.get('prediction_interval_calibration'), dict):
        prediction_interval_calibration = cross_validation['prediction_interval_calibration']
    else:
        val_mc_mean_np, val_uncertainty_np = _mc_dropout_predict_scaled(model, x[val_idx_for_metrics], passes=INDIVIDUALIZED_CONFORMAL_MC_PASSES)
        prediction_interval_calibration = _individualized_conformal_summary_from_predictions(val_mc_mean_np, y[val_idx_for_metrics], val_uncertainty_np, mask[val_idx_for_metrics], trait_names, y_mean, y_std, confidence=0.95, source='holdout_validation_mc_dropout_scaled_residuals')
    uncertainty_metadata = {'uncertainty_source': 'mc_dropout_std', 'interval_method': 'mc_dropout_scaled_conformal', 'interval_note': 'PPMGS-Net uses MC dropout standard deviation to scale conformal radii for individualized 95% prediction intervals.'}
    pdae_summary = getattr(model, 'pdae_summary', None)
    if pdae_requested and (not use_pdae):
        pdae_summary = {'enabled': False, 'requested': True, 'reason': 'no_missing_targets' if allow_missing_phenotype else 'requires_allow_missing_phenotype', 'method': 'fold_local_pretrained_frozen_pdae_teacher', 'note': 'PDAE is bypassed when there are no missing phenotype cells, so it cannot perturb genotype-model optimization.'}
    attention_safety = getattr(model, 'attention_safety', None)
    directional_transfer_summary = _directional_transfer_summary(model, trait_names, observed_correlation_matrix(y[final_train_idx], mask[final_train_idx]) if len(trait_names) >= 2 else None)
    if prior_marker_summary is not None:
        prior_marker_summary = dict(prior_marker_summary)
        prior_marker_summary['prior_strength_learned'] = _learned_prior_strength(model)
        prior_marker_summary['prior_strength_by_trait'] = _learned_prior_strength_by_trait(model, trait_names)
        prior_marker_summary['prior_reliability_by_trait'] = _learned_prior_reliability_by_trait(model, trait_names)
        prior_marker_summary['effective_prior_strength_by_trait'] = _effective_prior_strength_by_trait(model, trait_names)
        prior_marker_summary['prior_learning_diagnostics'] = _prior_learning_diagnostics(model, x[final_train_idx], marker_names, trait_names, prior_scores, attention_safety=attention_safety)
    _mark_timing('post_training_evaluation', post_training_start)
    shap_top_markers: dict[str, list[dict[str, float | str | int | None]]] = {}
    shap_summary: dict[str, object] | None = {'enabled': False, 'requested': False}
    if compute_shap:
        shap_start = time.perf_counter()
        shap_top_markers, shap_summary = _gradient_shap_top_markers(model, x, marker_names, trait_names, prior_scores=prior_scores, top_k=shap_top_k, output_dir=job_dir, save_all_markers=save_full_shap)
        shap_seconds = _mark_timing('shap', shap_start)
        if shap_summary is not None:
            shap_summary = dict(shap_summary)
            shap_summary['elapsed_seconds'] = shap_seconds
            shap_summary['elapsed_minutes'] = round(shap_seconds / 60.0, 4)
    else:
        timing_sections['shap_seconds'] = 0.0
    if generated_prior_table is not None and prior_marker_summary is not None:
        generated_prior_write_start = time.perf_counter()
        job_dir.mkdir(parents=True, exist_ok=True)
        generated_prior_path = job_dir / 'snp_marker_lasso_gwas_prior.csv'
        generated_prior_table.to_csv(generated_prior_path, index=False)
        prior_marker_summary = dict(prior_marker_summary)
        prior_marker_summary['generated_prior_file'] = str(generated_prior_path)
        _mark_timing('generated_prior_write', generated_prior_write_start)
    else:
        timing_sections['generated_prior_write_seconds'] = 0.0
    for timing_key in ['data_preparation_seconds', 'uploaded_prior_parse_seconds', 'tassel_prior_seconds', 'prior_source_combination_seconds', 'lasso_prior_seconds', 'prior_total_seconds', 'hyperparameter_search_seconds', 'cross_validation_seconds', 'final_training_seconds', 'post_training_evaluation_seconds', 'shap_seconds', 'generated_prior_write_seconds']:
        timing_sections.setdefault(timing_key, 0.0)
    total_train_model_seconds = round(float(time.perf_counter() - timing_start), 4)
    timing_summary = {'enabled': True, 'unit': 'seconds', 'total_train_model_seconds': total_train_model_seconds, 'total_train_model_minutes': round(total_train_model_seconds / 60.0, 4), **timing_sections, 'requested': {'tassel_prior': bool(build_tassel_prior), 'lasso_prior': bool(build_lasso_prior), 'uploaded_prior_marker': bool(prior_marker_file is not None), 'hyperparameter_search': bool(optimize_hyperparameters), 'cross_validation': bool(use_cross_validation), 'oof_prediction_intervals': bool(save_oof_intervals and use_cross_validation), 'shap': bool(compute_shap), 'pdae': bool(pdae_requested), 'pdae_effective': bool(use_pdae), 'phenotype_truth_evaluation': bool(phenotype_truth_file is not None), 'directional_transfer': bool(use_directional_transfer), 'transfer_mode': transfer_mode, 'prior_sparsity': prior_sparsity, 'attention_fusion': 'learned_attention_fusion_alpha', 'ppmgs_early_stopping_metric': 'pearson', 'hyperparameter_early_stop_rounds': int(hyperparameter_early_stop_rounds or 0) if optimize_hyperparameters else 0, 'hyperparameter_max_epochs': int(hyperparameter_max_epochs or 0) if optimize_hyperparameters else 0, 'lr_scheduler': lr_scheduler, 'lr_scheduler_factor': float(scheduler_params['lr_scheduler_factor']), 'lr_scheduler_patience': int(scheduler_params['lr_scheduler_patience']), 'min_learning_rate': float(scheduler_params['min_learning_rate'])}, 'cross_validation_runs': int(cv_folds * cv_repeats) if use_cross_validation else 0, 'hyperparameter_trials_requested': int(hyperparameter_trials) if optimize_hyperparameters else 0, 'hyperparameter_tuning_folds': int(hyperparameter_folds) if optimize_hyperparameters else 0, 'note': 'Timings are measured inside train_model with time.perf_counter. CLI/API elapsed_seconds may be slightly larger because it also includes request handling and result serialization.'}
    job = TrainedJob(model=model, samples=x.shape[0], marker_names=marker_names, trait_names=trait_names, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, final_loss=final_loss, best_epoch=best_epoch, task_type=task_type, allow_missing_phenotype=allow_missing_phenotype, use_directional_transfer=use_directional_transfer, transfer_mode=transfer_mode, attention_mode=attention_mode, imputation_strategy='marker_mode', marker_fill_values=marker_fill_values, prior_scores=prior_scores, prior_marker_summary=prior_marker_summary, shap_top_markers=shap_top_markers, shap_summary=shap_summary, metrics=metrics, cross_validation=cross_validation, prediction_interval_calibration=prediction_interval_calibration, uncertainty_metadata=uncertainty_metadata, hyperparameters=ppmgs_params, hyperparameter_search=hyperparameter_search, pdae_summary=pdae_summary, phenotype_missing_summary=phenotype_missing_summary, attention_safety=attention_safety, directional_transfer_summary=directional_transfer_summary, timing_summary=timing_summary)
    JOB_STORE[job_id] = job
    save_job(job_id, job)
    return (job_id, job, int(mask.sum()))


def predict(job_id: str, genotype_file, mc_passes: int=20) -> list[dict]:
    if job_id not in JOB_STORE:
        load_job(job_id)
    job = JOB_STORE[job_id]
    geno = _read_csv(genotype_file)
    if 'sample_id' not in geno.columns:
        raise ValueError('Prediction genotype CSV must contain sample_id.')
    missing_markers = [name for name in job.marker_names if name not in geno.columns]
    if missing_markers:
        raise ValueError(f'Prediction file is missing markers: {missing_markers[:8]}')
    sample_ids = geno['sample_id'].astype(str).tolist()
    x = geno[job.marker_names].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    x = np.where(np.isfinite(x), x, job.marker_fill_values)
    x = (x - job.x_mean) / job.x_std
    x_t = torch.tensor(x, dtype=torch.float32)
    mean_scaled, std_scaled = _mc_dropout_predict_scaled(job.model, x_t, passes=mc_passes)
    mean = mean_scaled * job.y_std + job.y_mean
    std = np.abs(std_scaled * job.y_std)
    interval_traits = {}
    if isinstance(job.prediction_interval_calibration, dict):
        interval_traits = job.prediction_interval_calibration.get('traits') or {}
    scale_quantiles = np.full(len(job.trait_names), np.nan, dtype=np.float32)
    uncertainty_floors = np.full(len(job.trait_names), np.nan, dtype=np.float32)
    for trait_idx, trait in enumerate(job.trait_names):
        trait_interval = interval_traits.get(trait, {}) if isinstance(interval_traits, dict) else {}
        scale_quantile = trait_interval.get('scale_quantile') if isinstance(trait_interval, dict) and trait_interval.get('enabled', False) else None
        uncertainty_floor = trait_interval.get('uncertainty_floor') if isinstance(trait_interval, dict) and trait_interval.get('enabled', False) else None
        if scale_quantile is not None and uncertainty_floor is not None:
            if np.isfinite(scale_quantile) and np.isfinite(uncertainty_floor):
                scale_quantiles[trait_idx] = float(scale_quantile)
                uncertainty_floors[trait_idx] = float(uncertainty_floor)
    certainty_levels: list[dict[str, str]] = []
    q25 = np.nanpercentile(std, 25, axis=0)
    q75 = np.nanpercentile(std, 75, axis=0)
    for row_idx in range(len(sample_ids)):
        row_levels = {}
        for trait_idx, trait in enumerate(job.trait_names):
            value = float(std[row_idx, trait_idx])
            if not np.isfinite(value):
                level = 'not_available'
            elif value <= float(q25[trait_idx]):
                level = 'high'
            elif value >= float(q75[trait_idx]):
                level = 'low'
            else:
                level = 'medium'
            row_levels[trait] = level
        certainty_levels.append(row_levels)
    trait_mean = np.nanmean(mean, axis=0)
    trait_std = np.nanstd(mean, axis=0)
    trait_std = np.where(trait_std > 1e-08, trait_std, 1.0)
    standardized_mean = (mean - trait_mean) / trait_std
    radii = scale_quantiles[None, :] * np.maximum(std, uncertainty_floors[None, :])
    radii[:, ~np.isfinite(scale_quantiles)] = np.nan
    conservative_matrix = mean - radii
    standardized_conservative = (conservative_matrix - trait_mean) / trait_std
    mean_counts = np.sum(np.isfinite(standardized_mean), axis=1)
    conservative_counts = np.sum(np.isfinite(standardized_conservative), axis=1)
    selection_index_mean = np.divide(np.nansum(standardized_mean, axis=1), np.maximum(mean_counts, 1))
    selection_index_conservative = np.divide(np.nansum(standardized_conservative, axis=1), np.maximum(conservative_counts, 1))
    selection_index_mean = np.where(mean_counts > 0, selection_index_mean, np.nan)
    selection_index_conservative = np.where(conservative_counts > 0, selection_index_conservative, np.nan)
    selection_index_mean = np.where(np.isfinite(selection_index_mean), selection_index_mean, np.nan)
    selection_index_conservative = np.where(np.isfinite(selection_index_conservative), selection_index_conservative, np.nan)
    rows = []
    for row_idx, sample_id in enumerate(sample_ids):
        prediction_interval_95 = {}
        conservative_score = {}
        interval_width = {}
        for trait_idx, trait in enumerate(job.trait_names):
            radius = float(radii[row_idx, trait_idx])
            if not np.isfinite(radius):
                prediction_interval_95[trait] = {'lower': None, 'upper': None, 'radius': None, 'interval_width': None, 'conservative_score': None, 'mc_dropout_std': float(std[row_idx, trait_idx]) if np.isfinite(std[row_idx, trait_idx]) else None, 'scale_quantile': None, 'uncertainty_floor': None}
                conservative_score[trait] = None
                interval_width[trait] = None
                continue
            predicted_value = float(mean[row_idx, trait_idx])
            lower = predicted_value - radius
            upper = predicted_value + radius
            width = upper - lower
            prediction_interval_95[trait] = {'lower': lower, 'upper': upper, 'radius': radius, 'interval_width': width, 'conservative_score': lower, 'mc_dropout_std': float(std[row_idx, trait_idx]), 'scale_quantile': float(scale_quantiles[trait_idx]), 'uncertainty_floor': float(uncertainty_floors[trait_idx])}
            conservative_score[trait] = lower
            interval_width[trait] = width
        rows.append({'sample_id': sample_id, 'predictions': {trait: float(mean[row_idx, trait_idx]) for trait_idx, trait in enumerate(job.trait_names)}, 'uncertainty': {trait: float(std[row_idx, trait_idx]) for trait_idx, trait in enumerate(job.trait_names)}, 'certainty_level': certainty_levels[row_idx], 'prediction_interval_95': prediction_interval_95, 'interval_width': interval_width, 'conservative_score': conservative_score, 'selection_index_mean': float(selection_index_mean[row_idx]) if np.isfinite(selection_index_mean[row_idx]) else None, 'selection_index_conservative': float(selection_index_conservative[row_idx]) if np.isfinite(selection_index_conservative[row_idx]) else None})
    return rows
