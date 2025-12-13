import torch

class RoPE:
  def __init__(self , head_dim , max_sequence = 2048):
    self.head_dim = head_dim
    self.max_sequence = max_sequence
    self.half_dim = head_dim // 2

    # finding the inverse frequency
    inv_freq = torch.tensor([1.0 / (10000 ** (i / self.half_dim)) for i in range(self.half_dim)] , dtype=torch.float32)

    # finding the positions
    positions = torch.arange(self.max_sequence).float().unsqueeze(1)

    # computing the theta
    angles = positions * inv_freq.unsqueeze(0)

    # computing the sin and cos
    self.sine = torch.sin(angles)
    self.cosine = torch.cos(angles)

  def get_cos_and_sine(self , seq_len , device):
    return (self.cosine[:seq_len , :].to(device) , self.sine[:seq_len , :].to(device))

  def apply(self, q, k, cos, sin):
       
        # Expand cos and sin for broadcasting
        cos = cos[:, None, None, :]
        sin = sin[:, None, None, :]

        # Split even and odd dimensions
        q_even = q[..., ::2]
        q_odd = q[..., 1::2]
        k_even = k[..., ::2]
        k_odd = k[..., 1::2]

        # Rotate
        q_rot = torch.stack(
            (q_even * cos - q_odd * sin,
             q_even * sin + q_odd * cos),
            dim=-1
        ).flatten(-2)

        k_rot = torch.stack(
            (k_even * cos - k_odd * sin,
             k_even * sin + k_odd * cos),
            dim=-1
        ).flatten(-2)

        return q_rot, k_rot