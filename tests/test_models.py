import pytest
import torch
import torch.nn as nn
from src.models.architecture import MambaOnlyModel, CfCOnlyModel, MambaCfCModel
from src.models.mamba_fallback import MambaConfig, MambaBlock

def test_shapes_and_finite():
    B, L, D, C = 32, 30, 10, 53
    x = torch.randn(B, L, D)
    
    models = [
        MambaOnlyModel(D, C),
        CfCOnlyModel(D, C),
        MambaCfCModel(D, C)
    ]
    
    for model in models:
        out = model(x)
        assert out.shape == (B, C), f"Shape mismatch in {model.__class__.__name__}"
        assert torch.isfinite(out).all(), f"Non-finite output in {model.__class__.__name__}"

def test_hybrid_extra_parameters():
    D, C = 10, 53
    hybrid = MambaCfCModel(D, C)
    mamba_only = MambaOnlyModel(D, C)
    cfc_only = CfCOnlyModel(D, C)
    
    def get_param_count(model):
        return sum(p.numel() for p in model.parameters())
    
    hybrid_params = get_param_count(hybrid)
    mamba_params = get_param_count(mamba_only)
    cfc_params = get_param_count(cfc_only)
    
    # Hybrid has: embedding + mamba + projection + cfc + head
    # Mamba has: embedding + mamba + head
    # Difference should be exactly projection + cfc
    proj = nn.Linear(64, 64)
    proj_params = sum(p.numel() for p in proj.parameters())
    
    # CfCOnly has embedding(10->64) + cfc(64) + head(64->53)
    # The CfC parameters alone:
    cfc_core_params = sum(p.numel() for p in cfc_only.cfc.parameters())
    
    assert hybrid_params - mamba_params == proj_params + cfc_core_params, "Hybrid extra parameters do not match projection + CfC!"

def test_shared_structure():
    D, C = 10, 53
    hybrid = MambaCfCModel(D, C)
    mamba_only = MambaOnlyModel(D, C)
    
    # Assert structural classes are the same for the shared parts
    assert isinstance(hybrid.embedding, type(mamba_only.embedding))
    assert isinstance(hybrid.mamba_backbone, type(mamba_only.mamba_backbone))
    assert isinstance(hybrid.head, type(mamba_only.head))
    
    assert hybrid.embedding.in_features == mamba_only.embedding.in_features
    assert hybrid.embedding.out_features == mamba_only.embedding.out_features
    
    assert hybrid.head.in_features == mamba_only.head.in_features
    assert hybrid.head.out_features == mamba_only.head.out_features

def test_mamba_fallback_parallel_vs_sequential():
    config = MambaConfig(d_model=16, n_layers=1, d_state=16, pscan=True, use_cuda=False)
    block = MambaBlock(config)
    block.eval()
    
    batch_size = 2
    seq_len = 10
    
    x = torch.randn(batch_size, seq_len, config.d_inner)
    z = torch.randn(batch_size, seq_len, config.d_inner)
    
    with torch.no_grad():
        A = -torch.exp(block.A_log.float())
        D = block.D.float()
        
        deltaBC = block.x_proj(x)
        delta, B, C = torch.split(deltaBC, [config.dt_rank, config.d_state, config.d_state], dim=-1)
        delta, B, C = block._apply_layernorms(delta, B, C)
        
        delta = block.dt_proj.weight @ delta.transpose(1, 2)
        delta = delta.transpose(1, 2)
        delta = torch.nn.functional.softplus(delta + block.dt_proj.bias)
        
        y_pscan = block.selective_scan(x, delta, A, B, C, D)
        y_seq = block.selective_scan_seq(x, delta, A, B, C, D)
        
    torch.testing.assert_close(y_pscan, y_seq, rtol=1e-4, atol=1e-4, msg="pscan and sequential scan outputs differ")
