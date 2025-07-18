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
        try:
            # Create complex weight with explicit casting to complex
            complex_coeff = torch.complex(self.coeff_real, self.coeff_imag)
            
            # Complex einsum needs special handling - convert projection to complex type
            proj_complex = torch.complex(
                self.proj,  # Real part
                torch.zeros_like(self.proj)  # Imaginary part (zero)
            )
            
            # Debug shape and dtype info
            if not hasattr(self, '_printed_weight_debug'):
                print(f"[compl_weight] proj shape: {proj_complex.shape}, dtype: {proj_complex.dtype}")
                print(f"[compl_weight] complex_coeff shape: {complex_coeff.shape}, dtype: {complex_coeff.dtype}")
                self._printed_weight_debug = True
            
            # Use einsum with both inputs as complex tensors
            w = torch.einsum("or, r...i -> o...i", proj_complex, complex_coeff)
            return w
            
        except Exception as e:
            print(f"Error in compl_weight: {str(e)}")
            print(f"proj: {self.proj.shape if hasattr(self, 'proj') else 'N/A'}, "
                  f"dtype: {self.proj.dtype if hasattr(self, 'proj') else 'N/A'}")
            print(f"coeff_real: {self.coeff_real.shape if hasattr(self, 'coeff_real') else 'N/A'}, "
                  f"dtype: {self.coeff_real.dtype if hasattr(self, 'coeff_real') else 'N/A'}")
            raise

    def forward(self, x):
        try:
            # Debug input shape
            if not hasattr(self, '_printed_debug') or not self._printed_debug:
                print(f"[SpectralConvND] Input shape: {x.shape}, dtype: {x.dtype}")
                self._printed_debug = True
            
            # x: (B,C,H1,…,Hn)
            # Convert range to tuple for dim parameter
            x_ft = torch.fft.rfftn(x, dim=tuple(range(-self.n_dims, 0)))
            
            # Get complex weights
            W = self.compl_weight()  # (O, k1,…,kn, I)
            
            if not hasattr(self, '_printed_weights') or not self._printed_weights:
                print(f"[SpectralConvND] Weight shape: {W.shape}, dtype: {W.dtype}")
                self._printed_weights = True
            
            # Apply convolution in Fourier space
            # Note: need to handle each mode separately
            out_ft = torch.zeros(x_ft.shape[:-self.n_dims] + (self.out_ch,) + 
                                x_ft.shape[-self.n_dims:], 
                                dtype=torch.complex64, device=x.device)
            
            # For simplicity, handling 1D and 2D cases explicitly
            if self.n_dims == 1:
                # Ensure compatible types
                x_ft_slice = x_ft[..., :self.n_modes[0]]
                out_ft[..., :self.n_modes[0]] = torch.einsum(
                    "bi, oi -> bo", x_ft_slice, W)
            elif self.n_dims == 2:
                for i in range(min(self.n_modes[0], x_ft.shape[-2])):
                    for j in range(min(self.n_modes[1], x_ft.shape[-1])):
                        # Ensure compatible types
                        x_ft_slice = x_ft[..., i, j]
                        w_slice = W[..., i, j]
                        
                        # Debug shape and dtype on first iteration
                        if i == 0 and j == 0 and (not hasattr(self, '_printed_slice_debug') or not self._printed_slice_debug):
                            print(f"[SpectralConvND] x_ft_slice: {x_ft_slice.shape}, {x_ft_slice.dtype}")
                            print(f"[SpectralConvND] w_slice: {w_slice.shape}, {w_slice.dtype}")
                            self._printed_slice_debug = True
                            
                        out_ft[..., i, j] = torch.einsum(
                            "bi, oi -> bo", x_ft_slice, w_slice)
            else:
                # General N-dimensional case would require more complex indexing
                raise NotImplementedError("Only 1D and 2D spectral convolutions are implemented")
                
            # Convert back from Fourier space
            out = torch.fft.irfftn(out_ft, s=x.shape[-self.n_dims:], dim=tuple(range(-self.n_dims, 0)))
            
            if not hasattr(self, '_printed_out_debug') or not self._printed_out_debug:
                print(f"[SpectralConvND] Output shape: {out.shape}, dtype: {out.dtype}")
                self._printed_out_debug = True
                
            return out
        except Exception as e:
            print(f"Error in SpectralConvND.forward: {str(e)}")
            print(f"Input shape: {x.shape if 'x' in locals() else 'N/A'}, ")
            if 'W' in locals():
                print(f"Weight shape: {W.shape}, dtype: {W.dtype}")
            if 'x_ft' in locals():
                print(f"FFT shape: {x_ft.shape}, dtype: {x_ft.dtype}")
            raise
