from __future__ import annotations

from .impu_module import MentorImputationMixin
from .module import MentorModule, bounded_hidden_dim, make_mlp, require_torch


METHOD_CONFIGS = {
    "deep_ae": {"display_name": "Deep_AE", "how_impu": "replace", "compt_loss": "all", "ae_type": "AE"},
    "sm_dae": {"display_name": "SM_DAE", "how_impu": "mid", "compt_loss": "adv", "ae_type": "AE"},
    "sdai": {"display_name": "SDAi", "how_impu": "replace", "compt_loss": "complete", "ae_type": "AE"},
}


class Deep_AE(MentorModule, MentorImputationMixin):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        latent_dim: int | None = None,
        dropout: float = 0.0,
        ae_type: str = "AE",
        how_impu: str = "replace",
        compt_loss: str = "all",
        device: str | torch.device | None = None,
    ):
        super().__init__(device=device)
        self.input_dim = int(input_dim)
        self.hidden_dim = bounded_hidden_dim(self.input_dim, hidden_dim)
        self.latent_dim = int(latent_dim or self.hidden_dim)
        self.dropout = float(dropout)
        self.ae_type = ae_type
        self.how_impu = how_impu
        self.compt_loss = compt_loss
        self.encoder = make_mlp([self.input_dim, self.hidden_dim, self.latent_dim], hidden_activation="lr", output_activation="lr", dropout=self.dropout)
        self.decoder = make_mlp([self.latent_dim, self.hidden_dim, self.input_dim], hidden_activation="lr", output_activation=None, dropout=self.dropout)
        self.loss = None
        self.move_to_device()

    def _get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        mask_missing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        torch, _ = require_torch()
        origin = x if target is None else target
        feature = self._get_latent(x)
        recon = self.decoder(feature)
        if target is not None and mask_missing is not None:
            self.loss = self._get_impu_loss(recon, origin, mask_missing)
        else:
            self.loss = torch.mean(torch.square(recon - origin))
        return recon


def build_deep_ae(method: str, input_dim: int, **kwargs) -> Deep_AE:
    if method not in METHOD_CONFIGS:
        raise ValueError(f"Unsupported Deep_AE family method: {method}")
    config = dict(METHOD_CONFIGS[method])
    config.update(kwargs)
    config.pop("display_name", None)
    return Deep_AE(input_dim=input_dim, **config)
