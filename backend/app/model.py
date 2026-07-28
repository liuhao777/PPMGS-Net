from __future__ import annotations

import numpy as np
import torch
from torch import nn

try:  # Optional: used on Linux/GPU servers when mamba-ssm is installed.
    from mamba_ssm import Mamba as _NativeMamba
except Exception:  # pragma: no cover - optional dependency.
    _NativeMamba = None


def hidden_units_from_markers(marker_count: int) -> int:
    """Use an MTMEGPS-like width while keeping tiny demos and large SNP sets stable."""
    return max(64, min(1024, int(round(marker_count * 0.7))))


def activation_layer(name: str = "relu") -> nn.Module:
    name = str(name or "relu").strip().lower()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()


class DenseDropoutBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3, activation: str = "relu") -> None:
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
    hidden_layers = max(1, int(hidden_layers))
    layers: list[nn.Module] = [DenseDropoutBlock(input_dim, hidden_dim, dropout, activation=activation)]
    for _ in range(hidden_layers - 1):
        layers.append(DenseDropoutBlock(hidden_dim, hidden_dim, dropout, activation=activation))
    return nn.Sequential(*layers)


class TraitInteractionBlock(nn.Module):
    """Trait interaction with legacy attention and bounded residual borrowing modes."""

    VALID_MODES = {"none", "legacy", "residual_global", "residual_dynamic"}
    RESIDUAL_MAX_GATE = 0.30
    RESIDUAL_INITIAL_GATE = 0.05
    RESIDUAL_DYNAMIC_DELTA = 0.50

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.3,
        heads: int = 4,
        trait_count: int | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            heads = 1
        self.hidden_dim = int(hidden_dim)
        self.trait_count = int(trait_count or 1)
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
        if self.trait_count <= 1:
            self.trait_borrow_logits = None
        else:
            self.trait_borrow_logits = nn.Parameter(torch.full((self.trait_count, self.trait_count), -4.0))

        residual_rank = min(16, max(8, hidden_dim // 64))
        self.residual_rank = residual_rank
        self.residual_adapters = nn.ModuleDict()
        for target_idx in range(self.trait_count):
            for source_idx in range(self.trait_count):
                if source_idx == target_idx:
                    continue
                adapter = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, residual_rank, bias=False),
                    activation_layer(activation),
                    nn.Linear(residual_rank, hidden_dim, bias=False),
                    nn.Dropout(dropout),
                )
                nn.init.normal_(adapter[3].weight, mean=0.0, std=1e-3)
                self.residual_adapters[f"{source_idx}_to_{target_idx}"] = adapter

        initial_probability = self.RESIDUAL_INITIAL_GATE / self.RESIDUAL_MAX_GATE
        initial_logit = float(np.log(initial_probability / (1.0 - initial_probability)))
        self.residual_global_logits = nn.Parameter(
            torch.full((self.trait_count, self.trait_count), initial_logit)
        )
        self.residual_context = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, residual_rank),
            activation_layer(activation),
        )
        self.residual_dynamic_weights = nn.Parameter(
            torch.zeros(self.trait_count, self.trait_count, residual_rank)
        )
        self.residual_dropout = nn.Dropout(dropout)
        self.trait_gate_mode = "legacy"
        self.trait_borrowing_enabled = True
        self.set_trait_gate_mode("legacy")

    def set_trait_borrowing_enabled(self, enabled: bool) -> None:
        self.set_trait_gate_mode("legacy" if enabled else "none")

    def set_trait_gate_mode(self, mode: str) -> None:
        mode = str(mode or "legacy").strip().lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown trait gate mode: {mode}")
        self.trait_gate_mode = mode
        self.trait_borrowing_enabled = mode != "none"
        if self.trait_borrow_logits is not None:
            self.trait_borrow_logits.requires_grad_(mode == "legacy")
        residual_enabled = mode in {"residual_global", "residual_dynamic"}
        for parameter in self.residual_adapters.parameters():
            parameter.requires_grad_(residual_enabled)
        self.residual_global_logits.requires_grad_(residual_enabled)
        dynamic_enabled = mode == "residual_dynamic"
        for parameter in self.residual_context.parameters():
            parameter.requires_grad_(dynamic_enabled)
        self.residual_dynamic_weights.requires_grad_(dynamic_enabled)

    def trait_borrow_gates(self) -> torch.Tensor | None:
        if self.trait_borrow_logits is None:
            return None
        if self.trait_gate_mode != "legacy":
            return torch.eye(
                self.trait_borrow_logits.shape[0],
                device=self.trait_borrow_logits.device,
                dtype=self.trait_borrow_logits.dtype,
            )
        gates = torch.sigmoid(self.trait_borrow_logits)
        eye = torch.eye(gates.shape[0], device=gates.device, dtype=gates.dtype)
        return gates * (1.0 - eye) + eye

    def residual_global_gates(self) -> torch.Tensor:
        gates = self.RESIDUAL_MAX_GATE * torch.sigmoid(self.residual_global_logits)
        eye = torch.eye(gates.shape[0], device=gates.device, dtype=gates.dtype)
        return gates * (1.0 - eye)

    def residual_gate_weights(self, trait_reps: torch.Tensor) -> torch.Tensor:
        logits = self.residual_global_logits[None, :, :].expand(trait_reps.shape[0], -1, -1)
        if self.trait_gate_mode == "residual_dynamic":
            context = self.residual_context(trait_reps.mean(dim=1))
            dynamic_logits = torch.einsum("br,tsr->bts", context, self.residual_dynamic_weights)
            logits = logits + self.RESIDUAL_DYNAMIC_DELTA * torch.tanh(dynamic_logits)
        gates = self.RESIDUAL_MAX_GATE * torch.sigmoid(logits)
        eye = torch.eye(gates.shape[1], device=gates.device, dtype=gates.dtype)
        return gates * (1.0 - eye[None, :, :])

    def trait_attention_bias(self, trait_count: int, device: torch.device) -> torch.Tensor | None:
        gates = self.trait_borrow_gates()
        if gates is None or gates.shape[0] != trait_count:
            return None
        return torch.log(gates.to(device).clamp_min(1e-6))

    def _attention_forward(
        self,
        trait_reps: torch.Tensor,
        attention_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            trait_reps,
            trait_reps,
            trait_reps,
            attn_mask=attention_bias,
            need_weights=False,
        )
        refined = self.norm1(trait_reps + self.dropout(attended))
        updated = self.feed_forward(refined)
        return self.norm2(refined + self.dropout(updated))

    def _self_only_forward(self, trait_reps: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(
            trait_reps.shape[1],
            device=trait_reps.device,
            dtype=trait_reps.dtype,
        )
        attention_bias = torch.log(identity.clamp_min(1e-6))
        return self._attention_forward(trait_reps, attention_bias)

    def _residual_forward(self, base_reps: torch.Tensor) -> torch.Tensor:
        gates = self.residual_gate_weights(base_reps)
        updates = []
        source_count = max(base_reps.shape[1] - 1, 1)
        for target_idx in range(base_reps.shape[1]):
            update = torch.zeros_like(base_reps[:, target_idx, :])
            for source_idx in range(base_reps.shape[1]):
                if source_idx == target_idx:
                    continue
                adapter = self.residual_adapters[f"{source_idx}_to_{target_idx}"]
                message = adapter(base_reps[:, source_idx, :])
                update = update + gates[:, target_idx, source_idx, None] * message
            updates.append(update / source_count)
        residual = torch.stack(updates, dim=1)
        return base_reps + self.residual_dropout(residual)

    def forward(self, trait_reps: torch.Tensor) -> torch.Tensor:
        if trait_reps.shape[1] <= 1:
            return trait_reps
        if self.trait_gate_mode == "legacy":
            attention_bias = self.trait_attention_bias(trait_reps.shape[1], trait_reps.device)
            return self._attention_forward(trait_reps, attention_bias)

        base_reps = self._self_only_forward(trait_reps)
        if self.trait_gate_mode == "none":
            return base_reps
        return self._residual_forward(base_reps)


class SourcePrivateTraitTransferBlock(nn.Module):
    """Transfer only source-trait deviations from the shared prior representation."""

    VALID_MODES = {"source_private_global_v2", "source_private_dynamic_v2"}
    MAX_GATE = 0.30
    INITIAL_GATE = 0.02
    DYNAMIC_DELTA = 0.75

    def __init__(
        self,
        hidden_dim: int,
        trait_count: int,
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.trait_count = int(trait_count)
        self.rank = max(2, int(rank))
        self.trait_gate_mode = "none"

        self.adapters = nn.ModuleDict()
        self.query_projections = nn.ModuleDict()
        self.key_projections = nn.ModuleDict()
        for target_idx in range(self.trait_count):
            self.query_projections[str(target_idx)] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.rank, bias=False),
            )
            for source_idx in range(self.trait_count):
                if source_idx == target_idx:
                    continue
                key = f"{source_idx}_to_{target_idx}"
                self.adapters[key] = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, self.rank, bias=False),
                    nn.GELU(),
                    nn.Linear(self.rank, hidden_dim, bias=False),
                    nn.Tanh(),
                )
                self.key_projections[key] = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, self.rank, bias=False),
                )

        initial_probability = self.INITIAL_GATE / self.MAX_GATE
        initial_logit = float(np.log(initial_probability / (1.0 - initial_probability)))
        self.global_logits = nn.Parameter(
            torch.full((self.trait_count, self.trait_count), initial_logit)
        )
        self.set_trait_gate_mode("none")

    def set_trait_gate_mode(self, mode: str) -> None:
        mode = str(mode or "none").strip().lower()
        self.trait_gate_mode = mode if mode in self.VALID_MODES else "none"
        enabled = self.trait_gate_mode in self.VALID_MODES
        dynamic = self.trait_gate_mode == "source_private_dynamic_v2"
        for parameter in self.adapters.parameters():
            parameter.requires_grad_(enabled)
        self.global_logits.requires_grad_(enabled)
        for parameter in self.query_projections.parameters():
            parameter.requires_grad_(dynamic)
        for parameter in self.key_projections.parameters():
            parameter.requires_grad_(dynamic)

    def global_gates(self) -> torch.Tensor:
        gates = self.MAX_GATE * torch.sigmoid(self.global_logits)
        eye = torch.eye(gates.shape[0], device=gates.device, dtype=gates.dtype)
        return gates * (1.0 - eye)

    def gate_weights(
        self,
        base_reps: torch.Tensor,
        private_reps: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.global_logits[None, :, :].expand(base_reps.shape[0], -1, -1)
        if self.trait_gate_mode == "source_private_dynamic_v2":
            private_deviation = private_reps - private_reps.mean(dim=1, keepdim=True)
            compatibility = torch.zeros_like(logits)
            for target_idx in range(self.trait_count):
                query = self.query_projections[str(target_idx)](base_reps[:, target_idx, :])
                for source_idx in range(self.trait_count):
                    if source_idx == target_idx:
                        continue
                    key_name = f"{source_idx}_to_{target_idx}"
                    key = self.key_projections[key_name](private_deviation[:, source_idx, :])
                    compatibility[:, target_idx, source_idx] = torch.nn.functional.cosine_similarity(
                        query,
                        key,
                        dim=-1,
                        eps=1e-6,
                    )
            logits = logits + self.DYNAMIC_DELTA * compatibility
        gates = self.MAX_GATE * torch.sigmoid(logits)
        eye = torch.eye(gates.shape[1], device=gates.device, dtype=gates.dtype)
        return gates * (1.0 - eye[None, :, :])

    def forward(
        self,
        base_reps: torch.Tensor,
        private_reps: torch.Tensor,
    ) -> torch.Tensor:
        if self.trait_count <= 1 or self.trait_gate_mode not in self.VALID_MODES:
            return base_reps
        private_deviation = private_reps - private_reps.mean(dim=1, keepdim=True)
        gates = self.gate_weights(base_reps, private_reps)
        updates = []
        source_count = max(self.trait_count - 1, 1)
        for target_idx in range(self.trait_count):
            update = torch.zeros_like(base_reps[:, target_idx, :])
            for source_idx in range(self.trait_count):
                if source_idx == target_idx:
                    continue
                key = f"{source_idx}_to_{target_idx}"
                message = self.adapters[key](private_deviation[:, source_idx, :])
                update = update + gates[:, target_idx, source_idx, None] * message
            updates.append(update / source_count)
        return base_reps + torch.stack(updates, dim=1)

    @torch.no_grad()
    def diagnostics(
        self,
        base_reps: torch.Tensor,
        private_reps: torch.Tensor,
    ) -> dict[str, object]:
        gates = self.gate_weights(base_reps, private_reps)
        transferred = self.forward(base_reps, private_reps) - base_reps
        ratio = transferred.norm(dim=-1) / base_reps.norm(dim=-1).clamp_min(1e-6)
        by_target: dict[str, object] = {}
        for target_idx in range(self.trait_count):
            source_rows = []
            for source_idx in range(self.trait_count):
                if source_idx == target_idx:
                    continue
                values = gates[:, target_idx, source_idx]
                source_rows.append(
                    {
                        "source_index": int(source_idx),
                        "mean": float(values.mean().cpu()),
                        "std": float(values.std(unbiased=False).cpu()),
                        "p05": float(torch.quantile(values, 0.05).cpu()),
                        "p50": float(torch.quantile(values, 0.50).cpu()),
                        "p95": float(torch.quantile(values, 0.95).cpu()),
                    }
                )
            ratio_values = ratio[:, target_idx]
            by_target[str(target_idx)] = {
                "sources": source_rows,
                "effective_residual_ratio": {
                    "mean": float(ratio_values.mean().cpu()),
                    "std": float(ratio_values.std(unbiased=False).cpu()),
                    "p05": float(torch.quantile(ratio_values, 0.05).cpu()),
                    "p50": float(torch.quantile(ratio_values, 0.50).cpu()),
                    "p95": float(torch.quantile(ratio_values, 0.95).cpu()),
                },
            }
        return by_target


class PLELiteTraitMixer(nn.Module):
    """Lightweight shared/private expert mixer for small multi-trait datasets."""

    MODE = "ple_lite_pcgrad"
    MAX_PRIVATE_GATE = 0.50
    INITIAL_PRIVATE_GATE = 0.10

    def __init__(self, hidden_dim: int, trait_count: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.trait_count = int(trait_count)
        self.rank = min(32, max(8, self.hidden_dim // 16))
        self.enabled = False

        self.private_adapters = nn.ModuleList()
        self.mixing_gates = nn.ModuleList()
        initial_probability = self.INITIAL_PRIVATE_GATE / self.MAX_PRIVATE_GATE
        initial_logit = float(np.log(initial_probability / (1.0 - initial_probability)))
        for _trait_idx in range(self.trait_count):
            adapter = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.rank, bias=False),
                nn.GELU(),
                nn.Linear(self.rank, self.hidden_dim, bias=False),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
            nn.init.normal_(adapter[3].weight, mean=0.0, std=1e-2)
            self.private_adapters.append(adapter)

            gate = nn.Sequential(
                nn.LayerNorm(self.hidden_dim * 2),
                nn.Linear(self.hidden_dim * 2, self.rank),
                nn.GELU(),
                nn.Linear(self.rank, 1),
            )
            nn.init.zeros_(gate[3].weight)
            nn.init.constant_(gate[3].bias, initial_logit)
            self.mixing_gates.append(gate)

        self.set_enabled(False)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        for parameter in self.parameters():
            parameter.requires_grad_(self.enabled)

    def _components(
        self,
        base_reps: torch.Tensor,
        private_reps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_reps = []
        gate_values = []
        residuals = []
        for trait_idx in range(self.trait_count):
            base_rep = base_reps[:, trait_idx, :]
            private_delta = self.private_adapters[trait_idx](private_reps[:, trait_idx, :])
            gate_input = torch.cat([base_rep, private_reps[:, trait_idx, :]], dim=-1)
            gate = self.MAX_PRIVATE_GATE * torch.sigmoid(self.mixing_gates[trait_idx](gate_input))
            residual = gate * private_delta
            mixed_reps.append(base_rep + residual)
            gate_values.append(gate.squeeze(-1))
            residuals.append(residual)
        return (
            torch.stack(mixed_reps, dim=1),
            torch.stack(gate_values, dim=1),
            torch.stack(residuals, dim=1),
        )

    def forward(self, base_reps: torch.Tensor, private_reps: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.trait_count <= 1:
            return base_reps
        mixed, _gates, _residuals = self._components(base_reps, private_reps)
        return mixed

    @torch.no_grad()
    def diagnostics(self, base_reps: torch.Tensor, private_reps: torch.Tensor) -> dict[str, object]:
        _mixed, gates, residuals = self._components(base_reps, private_reps)
        residual_ratio = residuals.norm(dim=-1) / base_reps.norm(dim=-1).clamp_min(1e-6)
        by_trait: dict[str, object] = {}
        for trait_idx in range(self.trait_count):
            trait_gates = gates[:, trait_idx]
            trait_ratios = residual_ratio[:, trait_idx]
            by_trait[str(trait_idx)] = {
                "private_gate": {
                    "mean": float(trait_gates.mean().cpu()),
                    "std": float(trait_gates.std(unbiased=False).cpu()),
                    "p05": float(torch.quantile(trait_gates, 0.05).cpu()),
                    "p50": float(torch.quantile(trait_gates, 0.50).cpu()),
                    "p95": float(torch.quantile(trait_gates, 0.95).cpu()),
                },
                "effective_private_residual_ratio": {
                    "mean": float(trait_ratios.mean().cpu()),
                    "std": float(trait_ratios.std(unbiased=False).cpu()),
                    "p05": float(torch.quantile(trait_ratios, 0.05).cpu()),
                    "p50": float(torch.quantile(trait_ratios, 0.50).cpu()),
                    "p95": float(torch.quantile(trait_ratios, 0.95).cpu()),
                },
            }
        return by_trait


class CGCLiteGlobalTraitMixer(nn.Module):
    """Low-rank shared/private experts with trait-level global gates."""

    MODE = "cgc_lite_global"
    RANK = 8
    MAX_RESIDUAL_SCALE = 0.15
    INITIAL_RESIDUAL_SCALE = 0.03
    INITIAL_SHARED_WEIGHT = 0.65

    def __init__(self, hidden_dim: int, trait_count: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.trait_count = int(trait_count)
        self.rank = min(self.RANK, self.hidden_dim)
        self.enabled = False

        self.shared_expert = self._build_expert(dropout)
        self.private_experts = nn.ModuleList(
            [self._build_expert(dropout) for _ in range(self.trait_count)]
        )

        initial_private_weight = 1.0 - self.INITIAL_SHARED_WEIGHT
        initial_gate_logits = torch.log(
            torch.tensor(
                [self.INITIAL_SHARED_WEIGHT, initial_private_weight],
                dtype=torch.float32,
            )
        )
        self.gate_logits = nn.Parameter(
            initial_gate_logits[None, :].repeat(self.trait_count, 1)
        )
        initial_scale_probability = self.INITIAL_RESIDUAL_SCALE / self.MAX_RESIDUAL_SCALE
        initial_scale_logit = float(
            np.log(initial_scale_probability / (1.0 - initial_scale_probability))
        )
        self.residual_scale_logits = nn.Parameter(
            torch.full((self.trait_count,), initial_scale_logit)
        )
        self.set_enabled(False)

    def _build_expert(self, dropout: float) -> nn.Sequential:
        expert = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.rank, bias=False),
            nn.GELU(),
            nn.Linear(self.rank, self.hidden_dim, bias=False),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        nn.init.normal_(expert[3].weight, mean=0.0, std=2e-2)
        return expert

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        for parameter in self.parameters():
            parameter.requires_grad_(self.enabled)

    def global_gate_weights(self) -> torch.Tensor:
        return torch.softmax(self.gate_logits, dim=-1)

    def residual_scales(self) -> torch.Tensor:
        return self.MAX_RESIDUAL_SCALE * torch.sigmoid(self.residual_scale_logits)

    @staticmethod
    def _match_rms(expert_output: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        expert_rms = expert_output.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        reference_rms = reference.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return expert_output * (reference_rms / expert_rms)

    def _components(
        self,
        base_reps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_output = self.shared_expert(base_reps.mean(dim=1))
        gates = self.global_gate_weights()
        scales = self.residual_scales()
        mixed_reps = []
        shared_contributions = []
        private_contributions = []
        residuals = []
        for trait_idx in range(self.trait_count):
            base_rep = base_reps[:, trait_idx, :]
            shared_delta = self._match_rms(shared_output, base_rep)
            private_output = self.private_experts[trait_idx](base_rep)
            private_delta = self._match_rms(private_output, base_rep)
            shared_contribution = scales[trait_idx] * gates[trait_idx, 0] * shared_delta
            private_contribution = scales[trait_idx] * gates[trait_idx, 1] * private_delta
            residual = shared_contribution + private_contribution
            mixed_reps.append(base_rep + residual)
            shared_contributions.append(shared_contribution)
            private_contributions.append(private_contribution)
            residuals.append(residual)
        return (
            torch.stack(mixed_reps, dim=1),
            gates,
            torch.stack(shared_contributions, dim=1),
            torch.stack(private_contributions, dim=1),
            torch.stack(residuals, dim=1),
        )

    def forward(self, base_reps: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.trait_count <= 1:
            return base_reps
        mixed, _gates, _shared, _private, _residuals = self._components(base_reps)
        return mixed

    @torch.no_grad()
    def diagnostics(self, base_reps: torch.Tensor) -> dict[str, object]:
        _mixed, gates, shared, private, residuals = self._components(base_reps)
        base_norm = base_reps.norm(dim=-1).clamp_min(1e-6)
        residual_ratio = residuals.norm(dim=-1) / base_norm
        shared_ratio = shared.norm(dim=-1) / base_norm
        private_ratio = private.norm(dim=-1) / base_norm
        scales = self.residual_scales()
        by_trait: dict[str, object] = {}
        for trait_idx in range(self.trait_count):
            by_trait[str(trait_idx)] = {
                "global_gate_weights": {
                    "shared": float(gates[trait_idx, 0].cpu()),
                    "private": float(gates[trait_idx, 1].cpu()),
                },
                "residual_scale": float(scales[trait_idx].cpu()),
                "effective_residual_ratio": {
                    "mean": float(residual_ratio[:, trait_idx].mean().cpu()),
                    "std": float(residual_ratio[:, trait_idx].std(unbiased=False).cpu()),
                },
                "shared_contribution_ratio": {
                    "mean": float(shared_ratio[:, trait_idx].mean().cpu()),
                    "std": float(shared_ratio[:, trait_idx].std(unbiased=False).cpu()),
                },
                "private_contribution_ratio": {
                    "mean": float(private_ratio[:, trait_idx].mean().cpu()),
                    "std": float(private_ratio[:, trait_idx].std(unbiased=False).cpu()),
                },
                "expert_output_norms": {
                    "shared_mean": float(shared[:, trait_idx].norm(dim=-1).mean().cpu()),
                    "private_mean": float(private[:, trait_idx].norm(dim=-1).mean().cpu()),
                    "base_mean": float(base_reps[:, trait_idx].norm(dim=-1).mean().cpu()),
                },
            }
        return by_trait


class PhenotypeDenoisingAutoencoder(nn.Module):
    """Denoise partially observed phenotype vectors.

    The input is the standardized phenotype values concatenated with a binary
    mask that tells the network which trait values are available. It is used
    only during training to learn trait-trait structure; prediction still
    depends on genotype only.
    """

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


class MultiTraitGSNet(nn.Module):
    """Trait-aware multi-task GS model.

    The genotype encoder learns a shared representation. Each trait has an
    embedding that modulates this representation before producing a prediction.
    """

    def __init__(
        self,
        marker_count: int,
        trait_count: int,
        hidden_dim: int | None = None,
        trait_dim: int = 32,
        dropout: float = 0.3,
        hidden_layers: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or hidden_units_from_markers(marker_count)
        self.marker_norm = nn.LayerNorm(marker_count)
        self.encoder = dense_dropout_stack(marker_count, hidden_dim, dropout, hidden_layers, activation=activation)
        self.trait_embedding = nn.Embedding(trait_count, trait_dim)
        self.trait_gate = nn.Sequential(
            nn.Linear(hidden_dim + trait_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
        )
        self.trait_interaction = TraitInteractionBlock(
            hidden_dim,
            dropout=dropout,
            trait_count=trait_count,
            activation=activation,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def trait_representations(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.marker_norm(x))
        trait_ids = torch.arange(self.trait_embedding.num_embeddings, device=x.device)
        trait_z = self.trait_embedding(trait_ids)

        batch_size = x.shape[0]
        z_expanded = z[:, None, :].expand(batch_size, trait_z.shape[0], z.shape[1])
        trait_expanded = trait_z[None, :, :].expand(batch_size, trait_z.shape[0], trait_z.shape[1])
        return self.trait_gate(torch.cat([z_expanded, trait_expanded], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.trait_interaction(self.trait_representations(x))
        return self.head(fused).squeeze(-1)


class MultiHeadGSNet(nn.Module):
    """Shared encoder with one independent regression head per trait."""

    def __init__(
        self,
        marker_count: int,
        trait_count: int,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
        hidden_layers: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or hidden_units_from_markers(marker_count)
        self.marker_norm = nn.LayerNorm(marker_count)
        self.encoder = dense_dropout_stack(marker_count, hidden_dim, dropout, hidden_layers, activation=activation)
        trait_dim = min(32, hidden_dim)
        self.trait_embedding = nn.Embedding(trait_count, trait_dim)
        self.trait_gate = nn.Sequential(
            nn.Linear(hidden_dim + trait_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
        )
        self.trait_interaction = TraitInteractionBlock(
            hidden_dim,
            dropout=dropout,
            trait_count=trait_count,
            activation=activation,
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(trait_count)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.marker_norm(x))
        trait_ids = torch.arange(self.trait_embedding.num_embeddings, device=x.device)
        trait_z = self.trait_embedding(trait_ids)
        z_expanded = z[:, None, :].expand(x.shape[0], trait_z.shape[0], z.shape[1])
        trait_expanded = trait_z[None, :, :].expand(x.shape[0], trait_z.shape[0], trait_z.shape[1])
        trait_reps = self.trait_interaction(self.trait_gate(torch.cat([z_expanded, trait_expanded], dim=-1)))
        return torch.cat([head(trait_reps[:, idx, :]) for idx, head in enumerate(self.heads)], dim=1)


class LegacySNPTokenAttentionGSNet(nn.Module):
    """Previous marker-gated MLP kept for loading older saved attention jobs."""

    attention_architecture = "legacy_marker_gate_v1"

    def __init__(
        self,
        marker_count: int,
        trait_count: int,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
        hidden_layers: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.marker_count = marker_count
        hidden_dim = hidden_dim or hidden_units_from_markers(marker_count)
        self.marker_norm = nn.LayerNorm(marker_count)
        self.marker_gate_logits = nn.Parameter(torch.zeros(marker_count))
        self.input_dropout = nn.Dropout(min(0.2, dropout))
        self.encoder = dense_dropout_stack(marker_count, hidden_dim, dropout, hidden_layers, activation=activation)
        trait_dim = min(32, hidden_dim)
        self.trait_embedding = nn.Embedding(trait_count, trait_dim)
        self.trait_gate = nn.Sequential(
            nn.Linear(hidden_dim + trait_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
        )
        self.trait_interaction = TraitInteractionBlock(
            hidden_dim,
            dropout=dropout,
            trait_count=trait_count,
            activation=activation,
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(trait_count)])

    def marker_gates(self) -> torch.Tensor:
        return 1.0 + 0.5 * torch.tanh(self.marker_gate_logits)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        gate_deviation = torch.abs(self.marker_gates() - 1.0)
        if torch.sum(gate_deviation) <= 1e-8:
            weights = torch.full_like(gate_deviation, 1.0 / max(self.marker_count, 1))
        else:
            weights = gate_deviation / torch.sum(gate_deviation)
        return weights[None, :].expand(x.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.marker_norm(x)
        x = self.input_dropout(x * self.marker_gates())
        z = self.encoder(x)
        trait_ids = torch.arange(self.trait_embedding.num_embeddings, device=x.device)
        trait_z = self.trait_embedding(trait_ids)
        z_expanded = z[:, None, :].expand(x.shape[0], trait_z.shape[0], z.shape[1])
        trait_expanded = trait_z[None, :, :].expand(x.shape[0], trait_z.shape[0], trait_z.shape[1])
        trait_reps = self.trait_interaction(self.trait_gate(torch.cat([z_expanded, trait_expanded], dim=-1)))
        return torch.cat([head(trait_reps[:, idx, :]) for idx, head in enumerate(self.heads)], dim=1)


class SNPTokenAttentionGSNet(nn.Module):
    """Residual hybrid marker-attention model.

    The first attention version pooled all SNP tokens into one compact vector.
    That can discard the many small additive effects that genomic selection often
    depends on. The next marker-gated version was safer, but attention still
    controlled the only input path. This version keeps an ordinary MLP branch as
    the main predictor and adds a lightweight gated auxiliary branch through a
    small residual coefficient. If attention is useful the coefficient can grow;
    if it is noisy the model remains close to the plain MLP.
    """

    attention_architecture = "residual_hybrid_marker_attention_v1"

    def __init__(
        self,
        marker_count: int,
        trait_count: int,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
        hidden_layers: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.marker_count = marker_count
        hidden_dim = hidden_dim or hidden_units_from_markers(marker_count)
        attention_dim = max(32, min(128, hidden_dim // 4))

        self.marker_norm = nn.LayerNorm(marker_count)
        self.marker_gate_logits = nn.Parameter(torch.zeros(marker_count))
        self.attention_input_dropout = nn.Dropout(min(0.2, dropout))
        self.main_encoder = dense_dropout_stack(marker_count, hidden_dim, dropout, hidden_layers, activation=activation)
        self.attention_encoder = nn.Sequential(
            DenseDropoutBlock(marker_count, attention_dim, dropout, activation=activation),
            DenseDropoutBlock(attention_dim, attention_dim, dropout, activation=activation),
            nn.Linear(attention_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
        )
        self.attention_fusion_logit = nn.Parameter(torch.tensor(-4.0))
        self.attention_runtime_scale = 1.0
        self.eval_blend_weights: torch.Tensor | None = None
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        trait_dim = min(32, hidden_dim)
        self.trait_embedding = nn.Embedding(trait_count, trait_dim)
        self.trait_gate = nn.Sequential(
            nn.Linear(hidden_dim + trait_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
        )
        self.trait_interaction = TraitInteractionBlock(
            hidden_dim,
            dropout=dropout,
            trait_count=trait_count,
            activation=activation,
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(trait_count)])

    def marker_gates(self) -> torch.Tensor:
        # Starts exactly as a plain MLP input and can learn marker amplification
        # or suppression without removing the original SNP signal.
        return 1.0 + 0.5 * torch.tanh(self.marker_gate_logits)

    def attention_fusion_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.attention_fusion_logit)

    def set_attention_runtime_scale(self, scale: float) -> None:
        self.attention_runtime_scale = float(np.clip(scale, 0.0, 1.0))

    def set_eval_blend_weights(self, weights) -> None:
        self.eval_blend_weights = torch.as_tensor(weights, dtype=torch.float32)

    def clear_eval_blend_weights(self) -> None:
        self.eval_blend_weights = None

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        gate_deviation = torch.abs(self.marker_gates() - 1.0)
        if torch.sum(gate_deviation) <= 1e-8:
            weights = torch.full_like(gate_deviation, 1.0 / max(self.marker_count, 1))
        else:
            weights = gate_deviation / torch.sum(gate_deviation)
        return weights[None, :].expand(x.shape[0], -1)

    def attention_regularization(self) -> torch.Tensor:
        gate_penalty = torch.mean((self.marker_gates() - 1.0) ** 2)
        fusion_penalty = self.attention_fusion_alpha() ** 2
        return gate_penalty + 0.05 * fusion_penalty

    def encode(self, x: torch.Tensor, attention_scale: float | None = None) -> torch.Tensor:
        if attention_scale is None:
            attention_scale = self.attention_runtime_scale
        x = self.marker_norm(x)
        main_z = self.main_encoder(x)
        gated_x = self.attention_input_dropout(x * self.marker_gates())
        attention_z = self.attention_encoder(gated_x)
        return self.fusion_norm(main_z + float(attention_scale) * self.attention_fusion_alpha() * attention_z)

    def _forward_with_attention_scale(self, x: torch.Tensor, attention_scale: float) -> torch.Tensor:
        z = self.encode(x, attention_scale=attention_scale)
        trait_ids = torch.arange(self.trait_embedding.num_embeddings, device=x.device)
        trait_z = self.trait_embedding(trait_ids)
        z_expanded = z[:, None, :].expand(x.shape[0], trait_z.shape[0], z.shape[1])
        trait_expanded = trait_z[None, :, :].expand(x.shape[0], trait_z.shape[0], trait_z.shape[1])
        trait_reps = self.trait_interaction(self.trait_gate(torch.cat([z_expanded, trait_expanded], dim=-1)))
        return torch.cat([head(trait_reps[:, idx, :]) for idx, head in enumerate(self.heads)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.eval_blend_weights is not None:
            main_pred = self._forward_with_attention_scale(x, attention_scale=0.0)
            attention_pred = self._forward_with_attention_scale(x, attention_scale=1.0)
            weight_on_main = self.eval_blend_weights.to(x.device).view(1, -1)
            return weight_on_main * main_pred + (1.0 - weight_on_main) * attention_pred
        return self._forward_with_attention_scale(x, attention_scale=self.attention_runtime_scale)


class PriorMarkerAttentionGSNet(SNPTokenAttentionGSNet):
    """Trait-specific marker attention guided by user-supplied SNP priors."""

    attention_architecture = "trait_specific_prior_informed_marker_attention_v2"

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
        resolved_hidden_dim = hidden_dim or hidden_units_from_markers(marker_count)
        super().__init__(
            marker_count=marker_count,
            trait_count=trait_count,
            hidden_dim=resolved_hidden_dim,
            dropout=dropout,
            hidden_layers=hidden_layers,
            activation=activation,
        )
        self.trait_count = trait_count
        if prior_scores is None:
            prior_scores_t = torch.zeros(trait_count, marker_count, dtype=torch.float32)
        else:
            prior_scores_t = torch.as_tensor(prior_scores, dtype=torch.float32)
            if prior_scores_t.ndim == 1:
                if prior_scores_t.numel() != marker_count:
                    raise ValueError(
                        f"prior_scores length {prior_scores_t.numel()} does not match marker_count {marker_count}."
                    )
                prior_scores_t = prior_scores_t[None, :].expand(trait_count, marker_count).clone()
            elif prior_scores_t.ndim == 2:
                if prior_scores_t.shape == (marker_count, trait_count):
                    prior_scores_t = prior_scores_t.T.contiguous()
                if prior_scores_t.shape != (trait_count, marker_count):
                    raise ValueError(
                        "prior_scores shape must be [marker_count], [trait_count, marker_count], "
                        f"or [marker_count, trait_count]; got {tuple(prior_scores_t.shape)}."
                    )
            else:
                raise ValueError(f"prior_scores must be 1D or 2D; got {prior_scores_t.ndim}D.")
        self.register_buffer("prior_scores", prior_scores_t)
        self.trait_marker_gate_logits = nn.Parameter(torch.zeros(trait_count, marker_count))
        self.prior_strength_raw = nn.Parameter(torch.zeros(trait_count))
        self.use_prior_reliability_gate = bool(use_prior_reliability_gate)
        if self.use_prior_reliability_gate:
            self.prior_reliability_logit = nn.Parameter(torch.full((trait_count,), -1.0))
        else:
            self.register_parameter("prior_reliability_logit", None)
        self.prior_dropout_rate = 0.0
        self.source_private_transfer = SourcePrivateTraitTransferBlock(
            hidden_dim=resolved_hidden_dim,
            trait_count=trait_count,
            rank=8,
        )
        self.ple_lite_mixer = PLELiteTraitMixer(
            hidden_dim=resolved_hidden_dim,
            trait_count=trait_count,
            dropout=dropout,
        )
        self.cgc_lite_mixer = CGCLiteGlobalTraitMixer(
            hidden_dim=resolved_hidden_dim,
            trait_count=trait_count,
            dropout=dropout,
        )
        self.trait_gate_mode = "legacy"

    def prior_strength(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.prior_strength_raw)

    def prior_reliability(self) -> torch.Tensor:
        if self.prior_reliability_logit is None:
            return torch.ones_like(self.prior_strength_raw)
        return torch.sigmoid(self.prior_reliability_logit)

    def effective_prior_strength(self) -> torch.Tensor:
        return self.prior_strength() * self.prior_reliability()

    def set_prior_dropout_rate(self, rate: float) -> None:
        self.prior_dropout_rate = float(np.clip(rate, 0.0, 0.8))

    def trait_marker_gates(self) -> torch.Tensor:
        trait_delta = 0.25 * torch.tanh(self.trait_marker_gate_logits)
        prior_scores = self.prior_scores
        if self.training and self.prior_dropout_rate > 0:
            keep_prob = 1.0 - self.prior_dropout_rate
            dropout_mask = (torch.rand_like(prior_scores) < keep_prob).float() / max(keep_prob, 1e-6)
            prior_scores = prior_scores * dropout_mask
        logits = self.marker_gate_logits[None, :] + trait_delta + self.effective_prior_strength()[:, None] * prior_scores
        return 1.0 + 0.5 * torch.tanh(logits)

    def marker_gates(self) -> torch.Tensor:
        return self.trait_marker_gates().mean(dim=0)

    def trait_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        gate_deviation = torch.abs(self.trait_marker_gates() - 1.0)
        denom = gate_deviation.sum(dim=1, keepdim=True)
        uniform = torch.full_like(gate_deviation, 1.0 / max(self.marker_count, 1))
        weights = torch.where(denom <= 1e-8, uniform, gate_deviation / denom.clamp_min(1e-8))
        return weights[None, :, :].expand(x.shape[0], -1, -1)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.trait_attention_weights(x).mean(dim=1)

    def attention_regularization(self) -> torch.Tensor:
        gates = self.trait_marker_gates()
        prior_weight = 1.0 / (1.0 + torch.clamp(self.prior_scores, min=0.0))
        gate_penalty = torch.mean(((gates - 1.0) ** 2) * prior_weight)
        prior_strength_penalty = torch.mean(self.prior_strength() ** 2)
        reliability_penalty = torch.mean(self.prior_reliability() ** 2) if self.prior_reliability_logit is not None else 0.0
        fusion_penalty = self.attention_fusion_alpha() ** 2
        return gate_penalty + 0.01 * prior_strength_penalty + 0.01 * reliability_penalty + 0.05 * fusion_penalty

    def trait_encoding_components(
        self,
        x: torch.Tensor,
        attention_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if attention_scale is None:
            attention_scale = self.attention_runtime_scale
        x = self.marker_norm(x)
        main_z = self.main_encoder(x)
        alpha = float(attention_scale) * self.attention_fusion_alpha()
        private_reps = []
        fused_reps = []
        gates = self.trait_marker_gates()
        for trait_idx in range(self.trait_count):
            gated_x = self.attention_input_dropout(x * gates[trait_idx])
            attention_z = self.attention_encoder(gated_x)
            private_reps.append(attention_z)
            fused_reps.append(self.fusion_norm(main_z + alpha * attention_z))
        return main_z, torch.stack(private_reps, dim=1), torch.stack(fused_reps, dim=1)

    def trait_encoded_representations(self, x: torch.Tensor, attention_scale: float | None = None) -> torch.Tensor:
        return self.trait_encoding_components(x, attention_scale=attention_scale)[2]

    def configure_trait_gate_mode(self, mode: str) -> None:
        mode = str(mode or "legacy").strip().lower()
        self.trait_gate_mode = mode
        if mode == PLELiteTraitMixer.MODE:
            self.trait_interaction.set_trait_gate_mode("none")
            self.source_private_transfer.set_trait_gate_mode("none")
            self.ple_lite_mixer.set_enabled(True)
            self.cgc_lite_mixer.set_enabled(False)
        elif mode == CGCLiteGlobalTraitMixer.MODE:
            self.trait_interaction.set_trait_gate_mode("none")
            self.source_private_transfer.set_trait_gate_mode("none")
            self.ple_lite_mixer.set_enabled(False)
            self.cgc_lite_mixer.set_enabled(True)
        elif mode in SourcePrivateTraitTransferBlock.VALID_MODES:
            self.trait_interaction.set_trait_gate_mode("none")
            self.source_private_transfer.set_trait_gate_mode(mode)
            self.ple_lite_mixer.set_enabled(False)
            self.cgc_lite_mixer.set_enabled(False)
        else:
            self.trait_interaction.set_trait_gate_mode(mode)
            self.source_private_transfer.set_trait_gate_mode("none")
            self.ple_lite_mixer.set_enabled(False)
            self.cgc_lite_mixer.set_enabled(False)

    def cgc_lite_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.cgc_lite_mixer.parameters() if parameter.requires_grad]

    def pcgrad_shared_parameters(self) -> list[nn.Parameter]:
        """Parameters whose gradients are shared by all trait objectives."""
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        modules = (
            self.marker_norm,
            self.main_encoder,
            self.attention_encoder,
            self.fusion_norm,
            self.trait_gate,
            self.trait_interaction,
        )
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        shared_parameters = (self.marker_gate_logits, self.attention_fusion_logit)
        for parameter in shared_parameters:
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
        return parameters

    def source_private_components(
        self,
        x: torch.Tensor,
        attention_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_scale is None:
            attention_scale = self.attention_runtime_scale
        _main_z, private_reps, trait_specific_z = self.trait_encoding_components(
            x,
            attention_scale=attention_scale,
        )
        trait_ids = torch.arange(self.trait_embedding.num_embeddings, device=x.device)
        trait_z = self.trait_embedding(trait_ids)
        trait_expanded = trait_z[None, :, :].expand(x.shape[0], trait_z.shape[0], trait_z.shape[1])
        conditioned = self.trait_gate(torch.cat([trait_specific_z, trait_expanded], dim=-1))
        base_reps = self.trait_interaction(conditioned)
        return base_reps, float(attention_scale) * private_reps

    def predict_from_source_private_components(
        self,
        base_reps: torch.Tensor,
        private_reps: torch.Tensor,
    ) -> torch.Tensor:
        transferred = self.source_private_transfer(base_reps, private_reps)
        return torch.cat([head(transferred[:, idx, :]) for idx, head in enumerate(self.heads)], dim=1)

    def ple_lite_components(
        self,
        x: torch.Tensor,
        attention_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _main_z, private_reps, fused_reps = self.trait_encoding_components(
            x,
            attention_scale=attention_scale,
        )
        return fused_reps, float(attention_scale) * private_reps

    @torch.no_grad()
    def collect_ple_lite_diagnostics(self, x: torch.Tensor) -> dict[str, object]:
        base_reps, private_reps = self.ple_lite_components(x, attention_scale=1.0)
        return self.ple_lite_mixer.diagnostics(base_reps, private_reps)

    def cgc_lite_base_representations(
        self,
        x: torch.Tensor,
        attention_scale: float,
    ) -> torch.Tensor:
        return self.base_prediction_representations(x, attention_scale=attention_scale)

    def base_prediction_representations(
        self,
        x: torch.Tensor,
        attention_scale: float | None = None,
    ) -> torch.Tensor:
        if attention_scale is None:
            attention_scale = self.attention_runtime_scale
        trait_specific_z = self.trait_encoded_representations(
            x,
            attention_scale=attention_scale,
        )
        trait_ids = torch.arange(self.trait_embedding.num_embeddings, device=x.device)
        trait_z = self.trait_embedding(trait_ids)
        trait_expanded = trait_z[None, :, :].expand(x.shape[0], trait_z.shape[0], trait_z.shape[1])
        conditioned = self.trait_gate(torch.cat([trait_specific_z, trait_expanded], dim=-1))
        return self.trait_interaction(conditioned)

    @torch.no_grad()
    def collect_cgc_lite_diagnostics(self, x: torch.Tensor) -> dict[str, object]:
        base_reps = self.cgc_lite_base_representations(x, attention_scale=1.0)
        return self.cgc_lite_mixer.diagnostics(base_reps)

    def _forward_with_attention_scale(self, x: torch.Tensor, attention_scale: float) -> torch.Tensor:
        if self.trait_gate_mode == CGCLiteGlobalTraitMixer.MODE:
            base_reps = self.cgc_lite_base_representations(x, attention_scale=attention_scale)
            mixed_reps = self.cgc_lite_mixer(base_reps)
            return torch.cat([head(mixed_reps[:, idx, :]) for idx, head in enumerate(self.heads)], dim=1)
        if self.trait_gate_mode == PLELiteTraitMixer.MODE:
            base_reps, private_reps = self.ple_lite_components(x, attention_scale=attention_scale)
            mixed_reps = self.ple_lite_mixer(base_reps, private_reps)
            trait_ids = torch.arange(self.trait_embedding.num_embeddings, device=x.device)
            trait_z = self.trait_embedding(trait_ids)
            trait_expanded = trait_z[None, :, :].expand(x.shape[0], trait_z.shape[0], trait_z.shape[1])
            trait_reps = self.trait_interaction(self.trait_gate(torch.cat([mixed_reps, trait_expanded], dim=-1)))
            return torch.cat([head(trait_reps[:, idx, :]) for idx, head in enumerate(self.heads)], dim=1)
        if self.trait_gate_mode in SourcePrivateTraitTransferBlock.VALID_MODES:
            base_reps, private_reps = self.source_private_components(x, attention_scale=attention_scale)
            return self.predict_from_source_private_components(base_reps, private_reps)
        trait_specific_z = self.trait_encoded_representations(x, attention_scale=attention_scale)
        trait_ids = torch.arange(self.trait_embedding.num_embeddings, device=x.device)
        trait_z = self.trait_embedding(trait_ids)
        trait_expanded = trait_z[None, :, :].expand(x.shape[0], trait_z.shape[0], trait_z.shape[1])
        trait_reps = self.trait_interaction(self.trait_gate(torch.cat([trait_specific_z, trait_expanded], dim=-1)))
        return torch.cat([head(trait_reps[:, idx, :]) for idx, head in enumerate(self.heads)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.eval_blend_weights is not None:
            main_pred = self._forward_with_attention_scale(x, attention_scale=0.0)
            attention_pred = self._forward_with_attention_scale(x, attention_scale=1.0)
            weight_on_main = self.eval_blend_weights.to(x.device).view(1, -1)
            return weight_on_main * main_pred + (1.0 - weight_on_main) * attention_pred
        return self._forward_with_attention_scale(x, attention_scale=self.attention_runtime_scale)


class DirectionalAnchorGSNet(nn.Module):
    """Frozen single-trait anchors with asymmetric low-rank residual transfer."""

    MODE = "directional_anchor"
    RANK = 8
    MAX_GATE = 0.25
    INITIAL_GATE = 0.10
    attention_architecture = "directional_single_trait_anchor_residual_transfer_v1"

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
        self.trait_gate_mode = self.MODE
        self.attention_runtime_scale = 1.0
        self.eval_blend_weights: torch.Tensor | None = None

        prior_scores_t = self._normalize_prior_scores(prior_scores)
        self.register_buffer("prior_scores", prior_scores_t)
        if anchor_models is None:
            anchor_models = []
            for trait_idx in range(self.trait_count):
                anchor = PriorMarkerAttentionGSNet(
                    marker_count=self.marker_count,
                    trait_count=1,
                    prior_scores=prior_scores_t[trait_idx : trait_idx + 1],
                    hidden_dim=self.hidden_dim,
                    dropout=dropout,
                    hidden_layers=hidden_layers,
                    activation=activation,
                    use_prior_reliability_gate=use_prior_reliability_gate,
                )
                anchor.configure_trait_gate_mode("none")
                anchor_models.append(anchor)
        if len(anchor_models) != self.trait_count:
            raise ValueError(
                f"Expected {self.trait_count} anchor models; received {len(anchor_models)}."
            )
        self.anchor_models = nn.ModuleList(anchor_models)

        self.directional_adapters = nn.ModuleDict()
        for source_idx in range(self.trait_count):
            for target_idx in range(self.trait_count):
                if source_idx == target_idx:
                    continue
                adapter = nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.rank, bias=False),
                    nn.GELU(),
                    nn.Linear(self.rank, 1, bias=False),
                    nn.Tanh(),
                )
                nn.init.zeros_(adapter[3].weight)
                self.directional_adapters[f"{source_idx}_to_{target_idx}"] = adapter

        initial_probability = self.INITIAL_GATE / self.MAX_GATE
        initial_logit = float(np.log(initial_probability / (1.0 - initial_probability)))
        gate_logits = torch.full((self.trait_count, self.trait_count), initial_logit)
        gate_logits.fill_diagonal_(-20.0)
        self.directional_gate_logits = nn.Parameter(gate_logits)
        off_diagonal = 1.0 - torch.eye(self.trait_count, dtype=torch.float32)
        self.register_buffer("directional_off_diagonal", off_diagonal)
        self.freeze_anchors()

    def _normalize_prior_scores(
        self,
        prior_scores: np.ndarray | torch.Tensor | None,
    ) -> torch.Tensor:
        if prior_scores is None:
            return torch.zeros(self.trait_count, self.marker_count, dtype=torch.float32)
        values = torch.as_tensor(prior_scores, dtype=torch.float32)
        if values.ndim == 1:
            if values.numel() != self.marker_count:
                raise ValueError("prior_scores marker count does not match marker_count.")
            return values[None, :].expand(self.trait_count, -1).clone()
        if values.shape == (self.marker_count, self.trait_count):
            values = values.T.contiguous()
        if values.shape != (self.trait_count, self.marker_count):
            raise ValueError(
                "prior_scores must have shape [trait_count, marker_count] for directional_anchor."
            )
        return values.clone()

    def replace_anchor_models(self, anchor_models: list[PriorMarkerAttentionGSNet]) -> None:
        if len(anchor_models) != self.trait_count:
            raise ValueError(
                f"Expected {self.trait_count} anchor models; received {len(anchor_models)}."
            )
        self.anchor_models = nn.ModuleList(anchor_models)
        self.freeze_anchors()

    def freeze_anchors(self) -> None:
        for anchor in self.anchor_models:
            anchor.configure_trait_gate_mode("none")
            for parameter in anchor.parameters():
                parameter.requires_grad_(False)
            anchor.eval()

    def set_anchors_eval(self) -> None:
        for anchor in self.anchor_models:
            anchor.eval()

    def configure_trait_gate_mode(self, mode: str) -> None:
        normalized = str(mode or self.MODE).strip().lower().replace("-", "_")
        if normalized != self.MODE:
            raise ValueError(f"DirectionalAnchorGSNet only supports {self.MODE}; got {mode}.")
        self.trait_gate_mode = self.MODE
        self.freeze_anchors()

    def directional_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.directional_adapters.parameters()) + [self.directional_gate_logits]
        return [parameter for parameter in parameters if parameter.requires_grad]

    def directional_gates(self) -> torch.Tensor:
        gates = self.MAX_GATE * torch.sigmoid(self.directional_gate_logits)
        return gates * self.directional_off_diagonal

    def set_attention_runtime_scale(self, scale: float) -> None:
        self.attention_runtime_scale = float(np.clip(scale, 0.0, 1.0))
        for anchor in self.anchor_models:
            anchor.set_attention_runtime_scale(self.attention_runtime_scale)

    def set_prior_dropout_rate(self, rate: float) -> None:
        for anchor in self.anchor_models:
            anchor.set_prior_dropout_rate(rate)

    def set_eval_blend_weights(self, weights) -> None:
        values = torch.as_tensor(weights, dtype=torch.float32).flatten()
        if values.numel() != self.trait_count:
            raise ValueError("One evaluation blend weight is required per trait.")
        self.eval_blend_weights = values
        for trait_idx, anchor in enumerate(self.anchor_models):
            anchor.set_eval_blend_weights([float(values[trait_idx])])

    def clear_eval_blend_weights(self) -> None:
        self.eval_blend_weights = None
        for anchor in self.anchor_models:
            anchor.clear_eval_blend_weights()

    def prior_strength(self) -> torch.Tensor:
        return torch.cat([anchor.prior_strength().reshape(-1) for anchor in self.anchor_models])

    def prior_reliability(self) -> torch.Tensor:
        return torch.cat([anchor.prior_reliability().reshape(-1) for anchor in self.anchor_models])

    def effective_prior_strength(self) -> torch.Tensor:
        return torch.cat(
            [anchor.effective_prior_strength().reshape(-1) for anchor in self.anchor_models]
        )

    def trait_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        rows = [anchor.trait_attention_weights(x)[:, 0, :] for anchor in self.anchor_models]
        return torch.stack(rows, dim=1)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.trait_attention_weights(x).mean(dim=1)

    def anchor_components(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
        for target_idx in range(self.trait_count):
            correction = representations[target_idx].new_zeros(
                (representations[target_idx].shape[0], 1)
            )
            for source_idx in range(self.trait_count):
                if source_idx == target_idx:
                    continue
                key = f"{source_idx}_to_{target_idx}"
                contribution = gates[target_idx, source_idx] * self.directional_adapters[key](
                    representations[source_idx]
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
        by_target: dict[str, object] = {}
        for target_idx in range(self.trait_count):
            anchor_scale = anchor_predictions[:, target_idx].std(unbiased=False).clamp_min(1e-6)
            correction_values = corrections[:, target_idx]
            sources = []
            for source_idx in range(self.trait_count):
                if source_idx == target_idx:
                    continue
                key = f"{source_idx}_to_{target_idx}"
                source_values = contributions[key].squeeze(-1)
                sources.append(
                    {
                        "source_index": int(source_idx),
                        "gate": float(gates[target_idx, source_idx].cpu()),
                        "contribution_rms": float(source_values.square().mean().sqrt().cpu()),
                    }
                )
            by_target[str(target_idx)] = {
                "sources": sources,
                "anchor_prediction_std": float(anchor_scale.cpu()),
                "correction_rms": float(correction_values.square().mean().sqrt().cpu()),
                "correction_to_anchor_std_ratio": float(
                    correction_values.square().mean().sqrt().cpu() / anchor_scale.cpu()
                ),
                "mean_absolute_prediction_change": float(
                    torch.mean(torch.abs(final_predictions[:, target_idx] - anchor_predictions[:, target_idx])).cpu()
                ),
            }
        return by_target


class BlockSNPTransformerGSNet(nn.Module):
    """Block-wise SNP Transformer for marker interaction modeling.

    Full SNP-level self-attention is too expensive for high-dimensional GS
    input, so consecutive markers are first compressed into block tokens. The
    Transformer then models interactions among those SNP blocks.
    """

    attention_architecture = "prior_aware_blockwise_snp_transformer_v2"

    def __init__(
        self,
        marker_count: int,
        trait_count: int,
        prior_scores: np.ndarray | torch.Tensor | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
        hidden_layers: int = 2,
        block_size: int | None = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.marker_count = marker_count
        if block_size is None:
            if marker_count >= 1_000_000:
                block_size = 5000
            elif marker_count >= 300_000:
                block_size = 2500
            elif marker_count >= 100_000:
                block_size = 1000
            elif marker_count >= 50_000:
                block_size = 500
            else:
                block_size = 100
        self.block_size = max(10, int(block_size))
        self.num_blocks = max(1, int(np.ceil(marker_count / self.block_size)))
        self.padded_marker_count = self.num_blocks * self.block_size
        hidden_dim = hidden_dim or min(256, hidden_units_from_markers(marker_count))
        if marker_count >= 1_000_000:
            hidden_dim = min(int(hidden_dim), 128)
        elif marker_count >= 300_000:
            hidden_dim = min(int(hidden_dim), 192)
        else:
            hidden_dim = min(int(hidden_dim), 512)
        heads = 4 if hidden_dim % 4 == 0 else 2 if hidden_dim % 2 == 0 else 1
        transformer_layers = max(1, min(3, int(hidden_layers)))

        self.marker_norm = nn.Identity() if marker_count >= 100_000 else nn.LayerNorm(marker_count)
        prior_scores_t = self._prepare_prior_scores(prior_scores, trait_count, marker_count)
        self.register_buffer("prior_scores", prior_scores_t)
        padded_prior = prior_scores_t
        if self.padded_marker_count > marker_count:
            padded_prior = torch.nn.functional.pad(padded_prior, (0, self.padded_marker_count - marker_count))
        prior_blocks = padded_prior.reshape(trait_count, self.num_blocks, self.block_size)
        block_prior = 0.7 * prior_blocks.max(dim=-1).values + 0.3 * prior_blocks.mean(dim=-1)
        self.register_buffer("block_prior_scores", block_prior)
        self.register_buffer("marker_prior_mean", prior_scores_t.mean(dim=0))
        self.prior_strength_raw = nn.Parameter(torch.full((trait_count,), -1.0))
        self.input_prior_strength_raw = nn.Parameter(torch.tensor(-1.5))
        self.block_projection = nn.Sequential(
            nn.Linear(self.block_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_blocks, hidden_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=max(64, hidden_dim * 2),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.output_norm = nn.LayerNorm(hidden_dim)

        trait_dim = min(32, hidden_dim)
        self.trait_embedding = nn.Embedding(trait_count, trait_dim)
        self.trait_gate = nn.Sequential(
            nn.Linear(hidden_dim + trait_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
        )
        self.trait_interaction = TraitInteractionBlock(
            hidden_dim,
            dropout=dropout,
            trait_count=trait_count,
            activation=activation,
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(trait_count)])

    @staticmethod
    def _prepare_prior_scores(
        prior_scores: np.ndarray | torch.Tensor | None,
        trait_count: int,
        marker_count: int,
    ) -> torch.Tensor:
        if prior_scores is None:
            return torch.zeros(trait_count, marker_count, dtype=torch.float32)
        scores = torch.as_tensor(prior_scores, dtype=torch.float32)
        if scores.ndim == 1:
            if scores.numel() != marker_count:
                raise ValueError(f"prior_scores length {scores.numel()} does not match marker_count {marker_count}.")
            scores = scores[None, :].expand(trait_count, marker_count).clone()
        elif scores.ndim == 2:
            if scores.shape == (marker_count, trait_count):
                scores = scores.T.contiguous()
            if scores.shape != (trait_count, marker_count):
                raise ValueError(
                    "prior_scores shape must be [marker_count], [trait_count, marker_count], "
                    f"or [marker_count, trait_count]; got {tuple(scores.shape)}."
                )
        else:
            raise ValueError(f"prior_scores must be 1D or 2D; got {scores.ndim}D.")
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        max_per_trait = scores.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        return (scores / max_per_trait).clamp(0.0, 1.0)

    def prior_strength(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.prior_strength_raw)

    def input_prior_strength(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.input_prior_strength_raw)

    def _block_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.marker_norm(x)
        if torch.any(self.marker_prior_mean > 0):
            prior = self.marker_prior_mean.to(x.device).view(1, -1)
            x = x * (1.0 + self.input_prior_strength() * prior)
        if self.padded_marker_count > self.marker_count:
            pad_width = self.padded_marker_count - self.marker_count
            x = torch.nn.functional.pad(x, (0, pad_width))
        blocks = x.reshape(x.shape[0], self.num_blocks, self.block_size)
        tokens = self.block_projection(blocks) + self.position_embedding
        return self.transformer(tokens)

    def trait_block_attention(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._block_tokens(x)
        base_logits = self.pool_score(tokens).squeeze(-1)
        prior_bias = self.prior_strength().to(x.device).view(1, -1, 1) * self.block_prior_scores.to(x.device).view(
            1, self.trait_embedding.num_embeddings, self.num_blocks
        )
        logits = base_logits[:, None, :] + prior_bias
        return torch.softmax(logits, dim=-1)

    def block_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.trait_block_attention(x).mean(dim=1)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        block_weights = self.block_attention(x)
        marker_weights = block_weights.repeat_interleave(self.block_size, dim=1)
        marker_weights = marker_weights[:, : self.marker_count]
        return marker_weights / marker_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def trait_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        block_weights = self.trait_block_attention(x)
        marker_weights = block_weights.repeat_interleave(self.block_size, dim=2)
        marker_weights = marker_weights[:, :, : self.marker_count]
        return marker_weights / marker_weights.sum(dim=2, keepdim=True).clamp_min(1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._block_tokens(x)
        block_weights = self.trait_block_attention_from_tokens(tokens)
        pooled = torch.sum(tokens[:, None, :, :] * block_weights[..., None], dim=2)
        return self.output_norm(pooled)

    def trait_block_attention_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        base_logits = self.pool_score(tokens).squeeze(-1)
        prior_bias = self.prior_strength().to(tokens.device).view(1, -1, 1) * self.block_prior_scores.to(tokens.device).view(
            1, self.trait_embedding.num_embeddings, self.num_blocks
        )
        return torch.softmax(base_logits[:, None, :] + prior_bias, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        trait_ids = torch.arange(self.trait_embedding.num_embeddings, device=x.device)
        trait_z = self.trait_embedding(trait_ids)
        trait_expanded = trait_z[None, :, :].expand(x.shape[0], trait_z.shape[0], trait_z.shape[1])
        trait_reps = self.trait_interaction(self.trait_gate(torch.cat([z, trait_expanded], dim=-1)))
        return torch.cat([head(trait_reps[:, idx, :]) for idx, head in enumerate(self.heads)], dim=1)


class MambaLiteBlock(nn.Module):
    """Mamba-style sequence mixer with a safe PyTorch fallback.

    When mamba-ssm is installed this wraps the native Mamba layer. Otherwise it
    uses a gated depthwise-convolutional state mixer that keeps the same
    sequence-in/sequence-out contract. The fallback is intentionally lightweight
    so the desktop UI can run without custom CUDA extensions.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.native_backend = False
        if _NativeMamba is not None:
            try:
                self.mixer = _NativeMamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
                self.native_backend = True
            except Exception:
                self.mixer = None
        else:
            self.mixer = None

        if self.mixer is None:
            self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)
            self.depthwise_conv = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=5,
                padding=4,
                groups=hidden_dim,
            )
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm(x)
        if self.mixer is not None:
            mixed = self.mixer(z)
        else:
            values, gates = self.in_proj(z).chunk(2, dim=-1)
            conv = self.depthwise_conv(values.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
            mixed = self.out_proj(torch.nn.functional.silu(conv) * torch.sigmoid(gates))
        x = residual + self.dropout(mixed)
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class PriorWeightedMambaGSNet(nn.Module):
    """GP-WAITER-inspired prior-weighted CNN/Mamba branch with an MLP safety path."""

    attention_architecture = "gp_waiter_inspired_prior_weighted_mamba_v1"

    def __init__(
        self,
        marker_count: int,
        trait_count: int,
        prior_scores: np.ndarray | torch.Tensor | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
        hidden_layers: int = 2,
        block_size: int | None = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.marker_count = marker_count
        self.trait_count = trait_count
        if block_size is None:
            if marker_count >= 1_000_000:
                block_size = 5000
            elif marker_count >= 300_000:
                block_size = 2500
            elif marker_count >= 100_000:
                block_size = 1000
            elif marker_count >= 50_000:
                block_size = 500
            else:
                block_size = 128
        self.block_size = max(10, int(block_size))
        self.num_blocks = max(1, int(np.ceil(marker_count / self.block_size)))
        self.padded_marker_count = self.num_blocks * self.block_size

        hidden_dim = hidden_dim or min(256, hidden_units_from_markers(marker_count))
        if marker_count >= 1_000_000:
            hidden_dim = min(int(hidden_dim), 128)
        elif marker_count >= 300_000:
            hidden_dim = min(int(hidden_dim), 192)
        else:
            hidden_dim = min(int(hidden_dim), 512)
        self.hidden_dim = int(hidden_dim)

        self.marker_norm = nn.Identity() if marker_count >= 100_000 else nn.LayerNorm(marker_count)
        self.main_encoder = dense_dropout_stack(
            marker_count,
            self.hidden_dim,
            dropout,
            max(1, int(hidden_layers)),
            activation=activation,
        )

        prior_scores_t = BlockSNPTransformerGSNet._prepare_prior_scores(prior_scores, trait_count, marker_count)
        self.register_buffer("prior_scores", prior_scores_t)
        padded_prior = prior_scores_t
        if self.padded_marker_count > marker_count:
            padded_prior = torch.nn.functional.pad(padded_prior, (0, self.padded_marker_count - marker_count))
        prior_blocks = padded_prior.reshape(trait_count, self.num_blocks, self.block_size)
        block_prior = 0.7 * prior_blocks.max(dim=-1).values + 0.3 * prior_blocks.mean(dim=-1)
        self.register_buffer("block_prior_scores", block_prior)

        self.input_prior_strength_raw = nn.Parameter(torch.full((trait_count,), -1.5))
        self.pool_prior_strength_raw = nn.Parameter(torch.full((trait_count,), -1.0))
        self.prior_reliability_logit = nn.Parameter(torch.full((trait_count,), -1.0))
        self.prior_dropout_rate = 0.0

        self.block_projection = nn.Sequential(
            nn.Linear(self.block_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_cnn = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=max(1, self.hidden_dim)),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_blocks, self.hidden_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        mamba_layers = max(1, min(3, int(hidden_layers)))
        self.sequence_layers = nn.ModuleList([MambaLiteBlock(self.hidden_dim, dropout=dropout) for _ in range(mamba_layers)])
        self.sequence_backend = "mamba_ssm" if any(getattr(layer, "native_backend", False) for layer in self.sequence_layers) else "mamba_lite"
        self.pool_score = nn.Linear(self.hidden_dim, 1)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.mamba_fusion_logit = nn.Parameter(torch.tensor(-3.0))
        self.attention_runtime_scale = 1.0
        self.eval_blend_weights: torch.Tensor | None = None

        trait_dim = min(32, self.hidden_dim)
        self.trait_embedding = nn.Embedding(trait_count, trait_dim)
        self.trait_gate = nn.Sequential(
            nn.Linear(self.hidden_dim + trait_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            activation_layer(activation),
            nn.Dropout(dropout),
        )
        self.trait_interaction = TraitInteractionBlock(
            self.hidden_dim,
            dropout=dropout,
            trait_count=trait_count,
            activation=activation,
        )
        self.heads = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in range(trait_count)])

    def prior_reliability(self) -> torch.Tensor:
        return torch.sigmoid(self.prior_reliability_logit)

    def prior_strength(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.pool_prior_strength_raw)

    def input_prior_strength(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.input_prior_strength_raw)

    def effective_prior_strength(self) -> torch.Tensor:
        return self.prior_strength() * self.prior_reliability()

    def effective_input_prior_strength(self) -> torch.Tensor:
        return self.input_prior_strength() * self.prior_reliability()

    def mamba_fusion_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.mamba_fusion_logit)

    def attention_fusion_alpha(self) -> torch.Tensor:
        return self.mamba_fusion_alpha()

    def set_attention_runtime_scale(self, scale: float) -> None:
        self.attention_runtime_scale = float(np.clip(scale, 0.0, 1.0))

    def set_eval_blend_weights(self, weights) -> None:
        self.eval_blend_weights = torch.as_tensor(weights, dtype=torch.float32)

    def clear_eval_blend_weights(self) -> None:
        self.eval_blend_weights = None

    def set_prior_dropout_rate(self, rate: float) -> None:
        self.prior_dropout_rate = float(np.clip(rate, 0.0, 0.8))

    def _prepare_trait_weighted_input(self, x: torch.Tensor, trait_idx: int) -> torch.Tensor:
        prior = self.prior_scores[trait_idx].to(x.device)
        if self.training and self.prior_dropout_rate > 0:
            keep_prob = 1.0 - self.prior_dropout_rate
            prior = prior * ((torch.rand_like(prior) < keep_prob).float() / max(keep_prob, 1e-6))
        if torch.any(prior > 0):
            strength = self.effective_input_prior_strength()[trait_idx].to(x.device)
            x = x * (1.0 + strength * prior.view(1, -1))
        return x

    def _tokens_for_trait(self, x_norm: torch.Tensor, trait_idx: int) -> torch.Tensor:
        x_trait = self._prepare_trait_weighted_input(x_norm, trait_idx)
        if self.padded_marker_count > self.marker_count:
            x_trait = torch.nn.functional.pad(x_trait, (0, self.padded_marker_count - self.marker_count))
        blocks = x_trait.reshape(x_trait.shape[0], self.num_blocks, self.block_size)
        tokens = self.block_projection(blocks) + self.position_embedding
        tokens = tokens + self.local_cnn(tokens.transpose(1, 2)).transpose(1, 2)
        for layer in self.sequence_layers:
            tokens = layer(tokens)
        return tokens

    def trait_block_attention_from_tokens(self, tokens: torch.Tensor, trait_idx: int) -> torch.Tensor:
        base_logits = self.pool_score(tokens).squeeze(-1)
        prior_bias = (
            self.effective_prior_strength()[trait_idx].to(tokens.device)
            * self.block_prior_scores[trait_idx].to(tokens.device).view(1, -1)
        )
        return torch.softmax(base_logits + prior_bias, dim=-1)

    def _mamba_representations(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reps = []
        weights = []
        for trait_idx in range(self.trait_count):
            tokens = self._tokens_for_trait(x_norm, trait_idx)
            block_weights = self.trait_block_attention_from_tokens(tokens, trait_idx)
            pooled = torch.sum(tokens * block_weights[..., None], dim=1)
            reps.append(self.output_norm(pooled))
            weights.append(block_weights)
        return torch.stack(reps, dim=1), torch.stack(weights, dim=1)

    def trait_block_attention(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.marker_norm(x)
        _, block_weights = self._mamba_representations(x_norm)
        return block_weights

    def block_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.trait_block_attention(x).mean(dim=1)

    def trait_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        block_weights = self.trait_block_attention(x)
        marker_weights = block_weights.repeat_interleave(self.block_size, dim=2)
        marker_weights = marker_weights[:, :, : self.marker_count]
        return marker_weights / marker_weights.sum(dim=2, keepdim=True).clamp_min(1e-8)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        marker_weights = self.trait_attention_weights(x).mean(dim=1)
        return marker_weights / marker_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def attention_regularization(self) -> torch.Tensor:
        prior_strength_penalty = torch.mean(self.prior_strength() ** 2)
        input_strength_penalty = torch.mean(self.input_prior_strength() ** 2)
        reliability_penalty = torch.mean(self.prior_reliability() ** 2)
        fusion_penalty = self.mamba_fusion_alpha() ** 2
        return 0.01 * prior_strength_penalty + 0.01 * input_strength_penalty + 0.01 * reliability_penalty + 0.05 * fusion_penalty

    def _forward_with_attention_scale(self, x: torch.Tensor, attention_scale: float) -> torch.Tensor:
        x_norm = self.marker_norm(x)
        main_z = self.main_encoder(x_norm)
        trait_ids = torch.arange(self.trait_embedding.num_embeddings, device=x.device)
        trait_z = self.trait_embedding(trait_ids)
        trait_expanded = trait_z[None, :, :].expand(x.shape[0], trait_z.shape[0], trait_z.shape[1])
        main_expanded = main_z[:, None, :].expand(x.shape[0], self.trait_count, self.hidden_dim)

        if attention_scale <= 0:
            fused = main_expanded
        else:
            mamba_reps, _ = self._mamba_representations(x_norm)
            alpha = float(attention_scale) * self.mamba_fusion_alpha()
            fused = self.fusion_norm(main_expanded + alpha * mamba_reps)

        trait_reps = self.trait_interaction(self.trait_gate(torch.cat([fused, trait_expanded], dim=-1)))
        return torch.cat([head(trait_reps[:, idx, :]) for idx, head in enumerate(self.heads)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.eval_blend_weights is not None:
            main_pred = self._forward_with_attention_scale(x, attention_scale=0.0)
            mamba_pred = self._forward_with_attention_scale(x, attention_scale=1.0)
            weight_on_main = self.eval_blend_weights.to(x.device).view(1, -1)
            return weight_on_main * main_pred + (1.0 - weight_on_main) * mamba_pred
        return self._forward_with_attention_scale(x, attention_scale=self.attention_runtime_scale)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    observed = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / observed


def masked_trait_balanced_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average per-trait MSE across traits represented in the current batch."""
    observed_by_trait = mask.sum(dim=0)
    squared_error_by_trait = (((pred - target) ** 2) * mask).sum(dim=0)
    active_traits = observed_by_trait > 0
    if not bool(active_traits.any()):
        return pred.sum() * 0.0
    trait_losses = squared_error_by_trait[active_traits] / observed_by_trait[active_traits].clamp_min(1.0)
    return trait_losses.mean()


def correlation_regularizer(pred: torch.Tensor, target_corr: torch.Tensor | None) -> torch.Tensor:
    """Legacy diagnostic helper; the active training objective no longer calls it."""
    if target_corr is None or pred.shape[0] < 3 or pred.shape[1] < 2:
        return pred.new_tensor(0.0)

    centered = pred - pred.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, keepdim=True).clamp_min(1e-6)
    standardized = centered / scale
    pred_corr = standardized.T @ standardized / max(pred.shape[0] - 1, 1)
    return ((pred_corr - target_corr) ** 2).mean()


def observed_correlation_matrix(y: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    trait_count = y.shape[1]
    if trait_count < 2:
        return None

    corr = np.eye(trait_count, dtype=np.float32)
    for i in range(trait_count):
        for j in range(i + 1, trait_count):
            observed = (mask[:, i] > 0) & (mask[:, j] > 0)
            if observed.sum() < 3:
                value = 0.0
            else:
                value = float(np.corrcoef(y[observed, i], y[observed, j])[0, 1])
                if not np.isfinite(value):
                    value = 0.0
            corr[i, j] = value
            corr[j, i] = value
    return corr
