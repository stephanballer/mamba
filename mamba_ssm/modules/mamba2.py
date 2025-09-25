# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin


class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: Union[int, Tuple[int, int]] = 4,
        conv_init: Optional[float] = None,
        expand: int = 2,
        headdim: int = 64,
        d_ssm: Optional[int] = None,  # If not None, SSM only on this many dims; rest uses gated MLP
        ngroups: int = 1,
        A_init_range: Tuple[float, float] = (1, 16),
        D_has_hdim: bool = False,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        bias: bool = False,
        conv_bias: bool = True,
        # Fused kernel and sharding options (1D-only)
        chunk_size: int = 256,
        use_mem_eff_path: bool = True,
        conv2d_causal: bool = False,
        layer_idx: Optional[int] = None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.conv2d_causal = conv2d_causal
        self.layer_idx = layer_idx

        # --- 1D vs 2D conv detection ---
        self.is_2d = isinstance(d_conv, (tuple, list))
        if self.is_2d:
            assert len(d_conv) == 2 and all(isinstance(k, int) and k >= 1 for k in d_conv)
            self.d_conv_hw = (int(d_conv[0]), int(d_conv[1]))
            self.kh, self.kw = self.d_conv_hw
        else:
            assert isinstance(d_conv, int) and d_conv >= 1
            self.d_conv = int(d_conv)

        # Order in in_proj output: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(
                self.d_model,
                d_in_proj * self.world_size,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state

        # --- Convolution module: depthwise 1D (causal) or depthwise 2D (non-causal/same) ---
        if not self.is_2d:
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=self.d_conv,
                groups=conv_dim,
                padding=self.d_conv - 1,  # causal padding; we crop later
                **factory_kwargs,
            )
            if self.conv_init is not None:
                nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        else:
            kh, kw = self.d_conv_hw
            pad = (0, 0) if self.conv2d_causal else (self.kh // 2, self.kw // 2)
            self.conv2d = nn.Conv2d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                kernel_size=(self.kh, self.kw),
                groups=conv_dim,          # depthwise
                padding=pad,              # 0 for causal (we’ll F.pad), “same” for non-causal
                bias=conv_bias,
                **factory_kwargs,
            )
            if self.conv_init is not None:
                nn.init.uniform_(self.conv2d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(
                self.d_inner * self.world_size,
                self.d_model,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs,
            )


    def forward(
        self,
        u: torch.Tensor,
        seqlen: Optional[int] = None,
        seq_idx: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        inference_params=None,
        hw: Optional[Tuple[int, int]] = None,  # (H, W) required for 2D conv
    ):
        """
        u: (B, L, D) if u is 3D; or u is (B*L, D) if packed (then pass seqlen=L).
        For 2D conv, pass hw=(H, W) such that H*W == L.
        Returns: same "packedness" as input (3D in, 3D out; 2D in, 2D out).
        """
        # --- detect packed vs batched input ---
        packed_input = (u.dim() == 2)
        if not packed_input:
            batch, seqlen, dim = u.shape
        else:
            assert seqlen is not None, "If u is (B*L, D), you must pass seqlen=L"
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen
    
        # --- 2D conv shape checks ---
        if self.is_2d:
            assert hw is not None and isinstance(hw, (tuple, list)) and len(hw) == 2, \
                "For 2D conv, pass hw=(H, W)."
            H, W = int(hw[0]), int(hw[1])
            assert H * W == seqlen, "H*W must equal sequence length."
    
        # --- inference cache (1D only) ---
        conv_state, ssm_state = None, None
        if inference_params is not None:
            if self.is_2d:
                raise NotImplementedError("2D conv does not support cached/varlen inference in this implementation.")
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # Updated inplace; only supported for 1D
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out
    
        # --- projection ---
        zxbcdt = self.in_proj(u)  # (B, L, ·) or (B*L, ·)
        if packed_input:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
    
        # Avoid -inf in fp16 for A
        A = -torch.exp(self.A_log.float())  # (nheads,)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
    
        # --- 1D fused fast path (unchanged) ---
        if (not self.is_2d) and self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if packed_input:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
            return out
    
        # --- Non-fused path (shared by 1D/2D) ---
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
    
        # --- Convolution branch ---
        assert self.activation in ["silu", "swish"]
        if not self.is_2d:
            # 1D path
            if conv_state is not None:
                if cu_seqlens is None:
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    xBC_t = F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
                    conv_state.copy_(xBC_t)  # (B, D, W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
    
            use_torch_conv = (
                xBC.device.type == "cpu"              # CPU: avoid CUDA-only custom op
                or causal_conv1d_fn is None           # package not available
                or self.activation not in ["silu", "swish"]
            )
            
            if use_torch_conv:
                # Plain depthwise Conv1d with causal crop
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)])
            else:
                # Fast CUDA custom op
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)

        else:
            # 2D conv branch (causal optional, no custom CUDA)
            B, L, Dtot = xBC.shape
            H, W = hw
            xBC_2d = rearrange(xBC, "b (h w) d -> b d h w", h=H, w=W).contiguous()
            
            if self.conv2d_causal:
                # asymmetric causal padding: (left, right, top, bottom)
                x_padded = F.pad(xBC_2d, (self.kw - 1, 0, self.kh - 1, 0))
                xBC_2d = F.conv2d(
                    x_padded,
                    self.conv2d.weight,
                    self.conv2d.bias,
                    stride=self.conv2d.stride,
                    padding=0,  # important: padding handled by F.pad
                    dilation=self.conv2d.dilation,
                    groups=self.conv2d.groups,  # depthwise
                )
                xBC_2d = self.act(xBC_2d)
            else:
                # non-causal "same" padding via the module (fast cuDNN)
                xBC_2d = self.act(self.conv2d(xBC_2d))
            
            xBC = rearrange(xBC_2d, "b d h w -> b (h w) d")
    
        # --- Scan (SSM) ---
        x, Bv, Cv = torch.split(
            xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(Bv, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(Cv, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
            z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            seq_idx=seq_idx if not self.is_2d else None,  # varlen seq idx not used for 2D here
            cu_seqlens=cu_seqlens if not self.is_2d else None,
            **dt_limit_kwargs,
            return_final_states=False,
            return_varlen_states=False,
        )
    
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
    
        if packed_input:
            y = rearrange(y, "b l d -> (b l) d")
        out = self.out_proj(y)
        return out

    # --- 1D decoding step path (unchanged). Not supported for 2D. ---
    def step(self, hidden_states, conv_state, ssm_state):
        if hasattr(self, "is_2d") and self.is_2d:
            raise NotImplementedError("step() decoding is 1D-only in this implementation.")
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )

        # Conv step (1D causal)
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, Bv, Cv = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, Bv, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), Cv)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)
        else:
            A_full = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt_full = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            Bv = rearrange(Bv, "b (g n) -> b g n", g=self.ngroups)
            Cv = rearrange(Cv, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt_full, A_full, Bv, Cv, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if getattr(self, "is_2d", False):
            raise NotImplementedError("allocate_inference_cache not supported for 2D in this implementation.")
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states: bool = False):
        if getattr(self, "is_2d", False):
            raise NotImplementedError("_get_states_from_cache not supported for 2D in this implementation.")
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

