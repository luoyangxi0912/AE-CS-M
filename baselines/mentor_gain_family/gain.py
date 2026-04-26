from __future__ import annotations

from baselines.mentor_ae_family.module import MentorModule, bounded_hidden_dim, make_mlp, require_torch


class GAIN(MentorModule):
    """Mentor-style GAIN baseline with AE-CS-M internal naming.

    Internally this class follows the mentor GAIN convention:
    mask_observed = 1 for observed values and mask_missing = 1 - mask_observed.
    The public experiment layer never sees mentor missing-indicator semantics.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        alpha: float = 100.0,
        hint_drop_rate: float = 0.2,
        device: str | torch.device | None = None,
    ):
        super().__init__(device=device)
        self.input_dim = int(input_dim)
        self.hidden_dim = bounded_hidden_dim(self.input_dim, hidden_dim)
        self.dropout = float(dropout)
        self.alpha = float(alpha)
        self.hint_drop_rate = float(hint_drop_rate)
        self.g_x = make_mlp([self.input_dim, self.hidden_dim], hidden_activation="lr", output_activation="lr", dropout=self.dropout)
        self.g_z = make_mlp([self.input_dim, self.hidden_dim], hidden_activation="lr", output_activation="lr", dropout=self.dropout)
        self.g_m = make_mlp([self.input_dim, self.hidden_dim], hidden_activation="lr", output_activation="lr", dropout=self.dropout)
        self.generator = make_mlp([self.hidden_dim, self.hidden_dim, self.input_dim], hidden_activation="lr", output_activation=None, dropout=self.dropout)
        self.d_x = make_mlp([self.input_dim, self.hidden_dim], hidden_activation="lr", output_activation="lr", dropout=self.dropout)
        self.d_h = make_mlp([self.input_dim, self.hidden_dim], hidden_activation="lr", output_activation="lr", dropout=self.dropout)
        self.discriminator = make_mlp([self.hidden_dim, self.hidden_dim, self.input_dim], hidden_activation="lr", output_activation="s", dropout=self.dropout)
        self.loss = None
        self.move_to_device()

    def generator_parameters(self):
        modules = [self.g_x, self.g_z, self.g_m, self.generator]
        for module in modules:
            yield from module.parameters()

    def discriminator_parameters(self):
        modules = [self.d_x, self.d_h, self.discriminator]
        for module in modules:
            yield from module.parameters()

    def _noise(self, x: torch.Tensor) -> torch.Tensor:
        torch, _ = require_torch()
        return torch.rand_like(x)

    def _hint(self, mask_observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        torch, _ = require_torch()
        reveal = torch.ones_like(mask_observed)
        reveal[torch.rand_like(mask_observed) < self.hint_drop_rate] = 0.0
        hint = reveal * mask_observed + 0.5 * (1.0 - reveal)
        hidden_part = 1.0 - reveal
        return hint, hidden_part

    def generate(self, x_input: torch.Tensor, mask_observed: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        noise = self._noise(x_input) if noise is None else noise
        hidden = self.g_x(x_input) + self.g_z(noise * (1.0 - mask_observed)) + self.g_m(mask_observed)
        return self.generator(hidden)

    def complete(self, x_input: torch.Tensor, mask_observed: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        generated = self.generate(x_input, mask_observed, noise=noise)
        return x_input * mask_observed + generated * (1.0 - mask_observed)

    def discriminate(self, x_completed: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        validity = self.discriminator(self.d_x(x_completed) + self.d_h(hint))
        return validity.clamp(1e-6, 1.0 - 1e-6)

    def discriminator_loss(self, x: torch.Tensor, mask_observed: torch.Tensor) -> torch.Tensor:
        torch, _ = require_torch()
        x_input = x * mask_observed
        with torch.no_grad():
            x_completed = self.complete(x_input, mask_observed)
        hint, hidden_part = self._hint(mask_observed)
        validity = self.discriminate(x_completed, hint)
        loss = -torch.mean(
            hidden_part
            * (
                mask_observed * torch.log(validity)
                + (1.0 - mask_observed) * torch.log(1.0 - validity)
            )
        )
        self.loss = loss
        return loss

    def generator_loss(self, x: torch.Tensor, mask_observed: torch.Tensor) -> torch.Tensor:
        torch, _ = require_torch()
        x_input = x * mask_observed
        generated = self.generate(x_input, mask_observed)
        x_completed = x_input * mask_observed + generated * (1.0 - mask_observed)
        hint, hidden_part = self._hint(mask_observed)
        validity = self.discriminate(x_completed, hint)
        mask_missing = 1.0 - mask_observed
        adv_loss = -torch.mean(hidden_part * mask_missing * torch.log(validity))
        recon_denom = torch.clamp(mask_observed.sum(), min=1.0)
        recon_loss = torch.sum(((generated - x) * mask_observed) ** 2) / recon_denom
        loss = adv_loss + self.alpha * recon_loss
        self.loss = loss
        return loss

    def forward(self, x: torch.Tensor, mask_observed: torch.Tensor | None = None) -> torch.Tensor:
        if mask_observed is None:
            _, torch_nn = require_torch()
            raise ValueError("GAIN.forward() requires mask_observed.")
        return self.complete(x * mask_observed, mask_observed)
