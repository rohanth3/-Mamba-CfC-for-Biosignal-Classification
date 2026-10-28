import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models.architecture import MambaOnlyModel, CfCOnlyModel, MambaCfCModel
from src.models.mamba_fallback import MambaConfig

def get_real_batch():
    try:
        data_dir = project_root / 'data' / 'processed' / 'db1'
        import h5py
        train_file = data_dir / 'train' / 'train_S1.h5'
        if not train_file.exists():
            torch.manual_seed(42)
            return torch.randn(32, 30, 10), torch.randint(0, 53, (32,))
        with h5py.File(train_file, 'r') as f:
            x = f['windows'][:32]
            y = f['labels'][:32]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    except Exception as e:
        torch.manual_seed(42)
        return torch.randn(32, 30, 10), torch.randint(0, 53, (32,))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_module_params(model):
    total = count_parameters(model)
    print(f"{'Module':<20} | {'Parameters':<10}")
    print("-" * 35)
    
    if hasattr(model, 'embedding'):
        print(f"{'Embedding':<20} | {count_parameters(model.embedding):<10}")
    
    if hasattr(model, 'mamba_backbone'):
        for i, layer in enumerate(model.mamba_backbone.layers):
            print(f"{f'Mamba Layer {i}':<20} | {count_parameters(layer):<10}")
            
    if hasattr(model, 'projection'):
        print(f"{'Projection':<20} | {count_parameters(model.projection):<10}")
        
    if hasattr(model, 'cfc'):
        print(f"{'CfC':<20} | {count_parameters(model.cfc):<10}")
        
    if hasattr(model, 'head'):
        print(f"{'Head':<20} | {count_parameters(model.head):<10}")
    
    print("-" * 35)
    print(f"{'Total':<20} | {total:<10}\n")

def overfit_batch(model, x, y, device, steps=300):
    torch.manual_seed(42)
    model.to(device)
    x, y = x.to(device), y.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        
        # Check gradients
        has_nans = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_nans = True
                break
                
        if has_nans:
            print(f"  [Warning] NaN gradients detected at step {step}!")
            
        optimizer.step()
        losses.append(loss.item())
        
    model.to('cpu')
    return losses[0], losses[-1]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    x, y = get_real_batch()
    print(f"Batch shape: X={x.shape}, Y={y.shape}")
    
    input_dim = x.shape[-1]
    num_classes = 53
    
    print("\n--- Mamba Config Used ---")
    print("d_model=64, d_state=16, d_conv=4, expand=2, dt_rank=4 (auto)")
    
    models = {
        'Mamba-only': MambaOnlyModel(input_dim, num_classes, d_model=64, num_layers=2),
        'Mamba-larger (upper-bound control)': MambaOnlyModel(input_dim, num_classes, d_model=64, num_layers=3),
        'Mamba-param-matched': MambaOnlyModel(input_dim, num_classes, d_model=88, num_layers=2, expand_factor=2),
        'CfC-only': CfCOnlyModel(input_dim, num_classes, d_model=64),
        'Mamba-CfC': MambaCfCModel(input_dim, num_classes, d_model=64, num_layers=2, cfc_units=64)
    }
    
    for name, model in models.items():
        print(f"\n========================================")
        print(f"Testing {name}")
        print(f"========================================")
        print_module_params(model)
        
        # Mean vs Last pooling
        print(f"--- Overfit Check (300 steps) ---")
        for pool_mode in ['mean', 'last']:
            model.pooling.mode = pool_mode
            init_loss, final_loss = overfit_batch(model, x, y, device, steps=300)
            print(f"Pooling: {pool_mode:<5} | Initial Loss: {init_loss:.4f} -> Final Loss: {final_loss:.4f}")

if __name__ == '__main__':
    main()
