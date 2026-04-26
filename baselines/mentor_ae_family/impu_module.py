from __future__ import annotations

from .module import require_torch


class MentorImputationMixin:
    """Internal mentor-style missing-indicator helpers.

    Inside this baseline family, mask_missing == 1 means missing. The adapter
    owns conversion from AE-CS-M public mask_observed.
    """

    compt_loss: str = "all"
    how_impu: str = "replace"

    def _get_impu_loss(self, recon: torch.Tensor, origin: torch.Tensor, mask_missing: torch.Tensor) -> torch.Tensor:
        torch, functional = _torch_functional()
        mask_missing = mask_missing.to(dtype=recon.dtype, device=recon.device)
        mask_complete = 1.0 - mask_missing
        if self.compt_loss == "all":
            return functional.mse_loss(recon, origin)
        if self.compt_loss == "complete":
            return _masked_mse(recon, origin, mask_complete)
        if self.compt_loss == "missing":
            return _masked_mse(recon, origin, mask_missing)
        if self.compt_loss == "adv":
            missing_rate = torch.mean(mask_missing)
            r = torch.clamp(missing_rate, 0.0, 1.0) * 0.5
            return _masked_mse(recon, origin, mask_missing) * r * 2.0 + _masked_mse(recon, origin, mask_complete) * (1.0 - r) * 2.0
        raise ValueError(f"Unsupported mentor compt_loss: {self.compt_loss}")

    def update_imputation_values(
        self,
        current: torch.Tensor,
        recon: torch.Tensor,
        origin: torch.Tensor,
        mask_missing: torch.Tensor,
    ) -> torch.Tensor:
        mask_missing = mask_missing.to(dtype=current.dtype, device=current.device)
        if self.how_impu == "replace":
            imputed = recon
        elif self.how_impu == "mid":
            imputed = (recon + origin) / 2.0
        elif self.how_impu == "grad":
            imputed = recon
        else:
            raise ValueError(f"Unsupported mentor how_impu: {self.how_impu}")
        return current * (1.0 - mask_missing) + imputed * mask_missing


def _masked_mse(recon: torch.Tensor, origin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    torch, _ = _torch_functional()
    denom = torch.sum(mask).clamp_min(1.0)
    return torch.sum(torch.square((recon - origin) * mask)) / denom


def _torch_functional():
    torch, _ = require_torch()
    import torch.nn.functional as functional

    return torch, functional
