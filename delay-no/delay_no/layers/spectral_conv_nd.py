import torch
import torch.nn as nn

class SpectralConvND(nn.Module):
    """
    N-dimensional Spectral Convolution layer for Fourier Neural Operator
    """
    def __init__(self, in_ch, out_ch, n_modes: tuple[int], *, rank=None):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.n_dims = len(n_modes)
        self.n_modes = n_modes
        # low‑rank factorisation (optional)
        r = rank or min(in_ch, out_ch)
        self.coeff_real = nn.Parameter(torch.randn(r, *n_modes, in_ch))
        self.coeff_imag = nn.Parameter(torch.randn(r, *n_modes, in_ch))
        self.proj = nn.Parameter(torch.randn(out_ch, r))

    def compl_weight(self):
        # W = P @ (A + iB)
        w = torch.einsum("or, r...i -> o...i", self.proj,
                         self.coeff_real + 1j * self.coeff_imag)
        return w

    def forward(self, x):
        # x: (B,C,H1,…,Hn)
        # Convert range to tuple for dim parameter
        x_ft = torch.fft.rfftn(x, dim=tuple(range(-self.n_dims, 0)))
        W = self.compl_weight()                    # (O, k1,…,kn, I)
        
        # Apply convolution in Fourier space
        # Note: need to handle each mode separately
        out_ft = torch.zeros(x_ft.shape[:-self.n_dims] + (self.out_ch,) + 
                             x_ft.shape[-self.n_dims:], 
                             dtype=torch.cfloat, device=x.device)
        
        # For simplicity, handling 1D and 2D cases explicitly
        if self.n_dims == 1:
            out_ft[..., :self.n_modes[0]] = torch.einsum(
                "bi, oi -> bo", x_ft[..., :self.n_modes[0]], W)
        elif self.n_dims == 2:
            for i in range(min(self.n_modes[0], x_ft.shape[-2])):
                for j in range(min(self.n_modes[1], x_ft.shape[-1])):
                    out_ft[..., i, j] = torch.einsum(
                        "bi, oi -> bo", x_ft[..., i, j], W[..., i, j])
        else:
            # General N-dimensional case would require more complex indexing
            raise NotImplementedError("Only 1D and 2D spectral convolutions are implemented")
            
        # Convert range to tuple for dim parameter
        out = torch.fft.irfftn(out_ft, s=x.shape[-self.n_dims:], dim=tuple(range(-self.n_dims, 0)))
        return out
