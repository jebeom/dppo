import torch
from torch import nn


class SteeringHeadMLP(nn.Module):
    """
    Simple MLP mapping state history to latent noise action with shape (Ta, Da).

    Inputs:
        - state: tensor of shape (B, To, Do)

    Outputs:
        - noise_action: tensor of shape (B, Ta, Da)
    """

    def __init__(
        self,
        obs_dim: int,
        cond_steps: int,
        action_dim: int,
        horizon_steps: int,
        hidden_dims: list[int] = None,
        activation: nn.Module = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]
        if activation is None:
            activation = nn.Mish()

        input_dim = obs_dim * cond_steps
        output_dim = action_dim * horizon_steps
        layers: list[nn.Module] = []
        last_dim = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(last_dim, dim), nn.LayerNorm(dim), activation]
            last_dim = dim
        layers += [nn.Linear(last_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, To, Do) -> flatten history
        batch_size = state.shape[0]
        x = state.view(batch_size, -1)
        w = self.mlp(x)
        return w.view(batch_size, self.horizon_steps, self.action_dim)


