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
        
        # Print initialization dimensions
        print(f"[SpectralConvND.__init__] in_ch={in_ch}, out_ch={out_ch}, n_modes={n_modes}, r={r}")
        
        # For 2D spectral convolution, we need weights with shape [out_ch, n_modes[0], n_modes[1], in_ch]
        if self.n_dims == 1:
            self.coeff_real = nn.Parameter(torch.randn(r, n_modes[0], in_ch))
            self.coeff_imag = nn.Parameter(torch.randn(r, n_modes[0], in_ch))
        elif self.n_dims == 2:
            self.coeff_real = nn.Parameter(torch.randn(r, n_modes[0], n_modes[1], in_ch))
            self.coeff_imag = nn.Parameter(torch.randn(r, n_modes[0], n_modes[1], in_ch))
        
        # Projection matrix: [out_ch, r]
        self.proj = nn.Parameter(torch.randn(out_ch, r))

    def compl_weight(self):
        try:
            # Create complex coefficients
            complex_coeff = torch.complex(self.coeff_real, self.coeff_imag)
            
            # Convert proj to complex dtype for einsum operations
            proj_complex = self.proj.to(dtype=torch.complex64)
            
            # Compute complex weights using einsum
            # CRITICAL FIX: Rearrange dimensions to [out_ch, in_ch, modes...] for proper slicing
            if self.n_dims == 1:
                # proj: [out_ch, r], complex_coeff: [r, modes, in_ch]
                # Result: [out_ch, in_ch, modes] - rearranged for proper slicing
                weight = torch.einsum("or,rmi->oim", proj_complex, complex_coeff)
            elif self.n_dims == 2:
                # proj: [out_ch, r], complex_coeff: [r, modes_x, modes_y, in_ch]
                # Result: [out_ch, in_ch, modes_x, modes_y] - rearranged for proper slicing
                weight = torch.einsum("or,rmni->oimn", proj_complex, complex_coeff)
            else:
                raise NotImplementedError(f"Spectral convolution not implemented for {self.n_dims}D")
            
            return weight
            
        except Exception as e:
            print(f"Error in compl_weight: {str(e)}")
            print(f"proj shape: {self.proj.shape}")
            print(f"coeff_real shape: {self.coeff_real.shape}")
            print(f"coeff_imag shape: {self.coeff_imag.shape}")
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
            
            # Get complex weights with corrected layout: [out_ch, in_ch, modes...]
            W = self.compl_weight()
            
            if not hasattr(self, '_printed_weights') or not self._printed_weights:
                print(f"[SpectralConvND] Weight shape: {W.shape}, dtype: {W.dtype}")
                self._printed_weights = True
            
            # Apply convolution in Fourier space
            # Initialize output tensor with correct shape
            out_ft = torch.zeros(x_ft.shape[:-self.n_dims] + (self.out_ch,) + 
                                x_ft.shape[-self.n_dims:], 
                                dtype=torch.complex64, device=x.device)
            
            # Handle 1D and 2D cases with corrected weight indexing
            if self.n_dims == 1:
                # Get actual available modes (RFFT reduces last dimension)
                available_modes = min(self.n_modes[0], x_ft.shape[-1])
                
                # Extract slices: x_ft [..., :modes], W [out_ch, in_ch, :modes]
                x_ft_slice = x_ft[..., :available_modes]  # [B, in_ch, modes]
                w_slice = W[:, :, :available_modes]       # [out_ch, in_ch, modes]
                
                # Perform batched matrix multiplication over modes
                # einsum: batch, input_ch, modes -> batch, output_ch, modes
                out_ft[..., :available_modes] = torch.einsum("bim,oim->bom", x_ft_slice, w_slice)
                
            elif self.n_dims == 2:
                # Get actual available modes (RFFT reduces last dimension)
                modes_x = min(self.n_modes[0], x_ft.shape[-2])
                modes_y = min(self.n_modes[1], x_ft.shape[-1])
                
                if not hasattr(self, '_printed_debug_dims') or not self._printed_debug_dims:
                    print(f"[SpectralConvND] x_ft shape: {x_ft.shape}")
                    print(f"[SpectralConvND] W shape: {W.shape}")
                    print(f"[SpectralConvND] Using modes: {modes_x}, {modes_y}")
                    self._printed_debug_dims = True
                
                # Extract slices: x_ft [..., :modes_x, :modes_y], W [out_ch, in_ch, :modes_x, :modes_y]
                x_ft_slice = x_ft[..., :modes_x, :modes_y]  # [B, in_ch, modes_x, modes_y]
                w_slice = W[:, :, :modes_x, :modes_y]       # [out_ch, in_ch, modes_x, modes_y]
                
                # Perform batched matrix multiplication over spatial modes
                # einsum: batch, input_ch, modes_x, modes_y -> batch, output_ch, modes_x, modes_y
                out_ft[..., :modes_x, :modes_y] = torch.einsum("bimn,oimn->bomn", x_ft_slice, w_slice)
                
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
