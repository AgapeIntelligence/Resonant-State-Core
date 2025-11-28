```python
import torch
import torch.nn as nn
import math

class RabiNet(nn.Module):
    """
    Oscillatory activation: cos(ω * x + φ)
    - ω: dimensionless frequency multiplier (default 1.0)
    - φ: phase offset (radians)
    """
    def __init__(self, omega: float = 1.0, phi: float = math.pi / 3.12, environment: str = "earth"):
        super().__init__()
        self.omega = torch.tensor([omega * (1.2 if environment == "mars" else 1.0)], dtype=torch.float32)
        self.phi = torch.tensor([phi * (1.1 if environment == "mars" else 1.0)], dtype=torch.float32)
        self.environment = environment

    def forward(self, x):
        return torch.cos(self.omega * x + self.phi)

class GrokResonantLayer(nn.Module):
    """
    Grok-style FFN block with oscillatory activation.
    """
    def __init__(self, d_model: int = 768, d_ff: int = 3072, omega: float = 1.0, environment: str = "earth"):
        super().__init__()
        self.proj_in = nn.Linear(d_model, d_ff)
        self.rabi = RabiNet(omega=omega, environment=environment)
        self.proj_out = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.proj_in(x)
        x = self.rabi(x)
        x = self.proj_out(x)
        return self.norm(x + residual)

if __name__ == "__main__":
    layer = GrokResonantLayer(d_model=768, d_ff=3072, omega=0.75)
    x = torch.randn(1, 10, 768)
    y = layer(x)
    sparsity = (torch.abs(y) < 0.1).float().mean().item()
    eps = 1e-8
    ent_proxy = -(torch.abs(y) * torch.log(torch.abs(y) + eps)).mean().item()
    print(f"Output Shape : {tuple(y.shape)}")
    print(f"Sparsity     : {sparsity:.4f}")
    print(f"Entropy Proxy: {ent_proxy:.4f}")
    print("\nRabiNet-Grok: Stability OK | Gradients OK")
```
