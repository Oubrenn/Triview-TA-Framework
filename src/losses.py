import torch
import torch.nn.functional as F


def tf_consistency_loss(z_time: torch.Tensor, z_freq: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Cosine similarity loss between embeddings."""
    z_time = F.normalize(z_time, dim=-1)
    z_freq = F.normalize(z_freq, dim=-1)
    logits = (z_time * z_freq).sum(dim=-1) / temperature
    return 1.0 - logits.mean()


def multi_domain_consistency_loss(
    z_time: torch.Tensor,
    z_freq: torch.Tensor,
    z_tf: torch.Tensor,
    temperature: float = 0.1,
    weights=None,
) -> torch.Tensor:
    """Pairwise consistency across time/frequency/time-frequency domains."""
    losses = [
        tf_consistency_loss(z_time, z_freq, temperature),
        tf_consistency_loss(z_time, z_tf, temperature),
        tf_consistency_loss(z_freq, z_tf, temperature),
    ]
    if weights is None:
        return sum(losses) / len(losses)
    if len(weights) != 3:
        raise ValueError("weights must have length 3 for (T-F, T-TF, F-TF).")
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive value.")
    return sum(w * l for w, l in zip(weights, losses)) / weight_sum


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> torch.Tensor:
    repr_loss = F.mse_loss(z1, z2)

    def std_loss(z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std))

    def cov_loss(z: torch.Tensor) -> torch.Tensor:
        n = z.shape[0]
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (n - 1)
        off_diag = cov.flatten()[:-1].view(z.shape[1] - 1, z.shape[1] + 1)[:, 1:].flatten()
        return (off_diag**2).sum() / z.shape[1]

    loss = sim_coeff * repr_loss + std_coeff * (std_loss(z1) + std_loss(z2)) + cov_coeff * (
        cov_loss(z1) + cov_loss(z2)
    )
    return loss


def ta_cfc_loss(z_anchor: torch.Tensor, z_warped: torch.Tensor, mode: str = "vicreg") -> torch.Tensor:
    if mode == "vicreg":
        return vicreg_loss(z_anchor, z_warped)
    if mode == "infonce":
        z_anchor = F.normalize(z_anchor, dim=-1)
        z_warped = F.normalize(z_warped, dim=-1)
        logits = z_anchor @ z_warped.T
        labels = torch.arange(z_anchor.shape[0], device=z_anchor.device)
        return F.cross_entropy(logits, labels)
    raise ValueError(f"Unknown TA-CFC mode: {mode}")


def color_regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-dimension MSE averaged over the band dimension (K) and batch."""
    per_dim = (pred - target) ** 2
    return per_dim.mean(dim=-1).mean()
