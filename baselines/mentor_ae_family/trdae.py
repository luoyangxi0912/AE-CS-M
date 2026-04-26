from __future__ import annotations

from .dae import Deep_AE
from .module import require_torch


class TRDAE(Deep_AE):
    def __init__(self, *args, trdae_exact_max_dim: int = 2048, **kwargs):
        kwargs.setdefault("how_impu", "grad")
        kwargs.setdefault("compt_loss", "all")
        super().__init__(*args, **kwargs)
        self.trdae_exact_max_dim = int(trdae_exact_max_dim)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        mask_missing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        torch, _ = require_torch()
        batch_size, input_dim = x.size(0), x.size(1)
        if input_dim > self.trdae_exact_max_dim:
            raise RuntimeError(
                "TRDAE exact leave-one-variable-out reconstruction is too large for this flattened window. "
                f"input_dim={input_dim}, limit={self.trdae_exact_max_dim}. Use a smaller baseline window_size."
            )
        eye = torch.eye(input_dim, device=x.device, dtype=x.dtype)
        x_epd = x.view(batch_size, 1, input_dim) * (1.0 - eye)
        x_in = x_epd.reshape(-1, input_dim)
        x_recon = self.decoder(self.encoder(x_in)).view(batch_size, input_dim, input_dim)
        recon = torch.diagonal(x_recon, dim1=1, dim2=2)
        origin = x if target is None else target
        if target is not None and mask_missing is not None:
            self.loss = self._get_impu_loss(recon, origin, mask_missing)
        else:
            self.loss = torch.mean(torch.square(recon - origin))
        return recon
