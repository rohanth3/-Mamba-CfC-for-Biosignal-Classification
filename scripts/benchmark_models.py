import torch
import torch.nn as nn
from src.models.architecture import MambaOnlyModel, CfCOnlyModel, MambaCfCModel
import time
import numpy as np
import torch.optim as optim

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def benchmark_latency(model, batch_size, device='cuda'):
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(batch_size, 30, 10).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    if device == 'cuda':
        torch.cuda.synchronize()
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    latencies = []
    for _ in range(50):
        start_event.record()
        _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
        
    return np.mean(latencies), np.std(latencies)

def benchmark_memory(model, batch_size, device='cuda'):
    model = model.to(device)
    model.train()
    
    dummy_input = torch.randn(batch_size, 30, 10).to(device)
    dummy_target = torch.randint(0, 53, (batch_size,)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    torch.cuda.reset_peak_memory_stats()
    
    optimizer.zero_grad()
    out = model(dummy_input)
    loss = criterion(out, dummy_target)
    loss.backward()
    optimizer.step()
    
    return torch.cuda.max_memory_allocated() / (1024**2)

def investigate_pooling(model_class, kwargs, pooling_modes, device='cuda', steps=300):
    results = {}
    for pool in pooling_modes:
        setup_seed(42)
        model = model_class(**kwargs, pooling=pool).to(device)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        dummy_input = torch.randn(128, 30, 10).to(device)
        dummy_target = torch.randint(0, 53, (128,)).to(device)
        
        losses = []
        for step in range(steps):
            optimizer.zero_grad()
            out = model(dummy_input)
            loss = criterion(out, dummy_target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        results[pool] = losses
    return results

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    models = {
        'Mamba-only': (MambaOnlyModel, {'input_dim': 10, 'num_classes': 53, 'd_model': 64, 'num_layers': 2, 'expand_factor': 2}),
        'Mamba-3L (depth control)': (MambaOnlyModel, {'input_dim': 10, 'num_classes': 53, 'd_model': 64, 'num_layers': 3, 'expand_factor': 2}),
        'Mamba-param-matched': (MambaOnlyModel, {'input_dim': 10, 'num_classes': 53, 'd_model': 88, 'num_layers': 2, 'expand_factor': 2}),
        'CfC-only': (CfCOnlyModel, {'input_dim': 10, 'num_classes': 53}),
        'Mamba-CfC': (MambaCfCModel, {'input_dim': 10, 'num_classes': 53, 'd_model': 64, 'num_layers': 2})
    }
    
    print("=== Latency & Memory Benchmark ===")
    for b_size in [32, 256]:
        print(f"\nBatch Size: {b_size}")
        for name, (cls, kwargs) in models.items():
            model = cls(**kwargs, pool_mode='mean')
            lat_mean, lat_std = benchmark_latency(model, b_size, device)
            mem_peak = benchmark_memory(model, b_size, device)
            print(f"{name:25s}: Latency = {lat_mean:6.2f} +/- {lat_std:5.2f} ms | Peak Mem = {mem_peak:7.2f} MB")
            
    print("\n=== Pooling Investigation ===")
    for name, (cls, kwargs) in models.items():
        if name in ['Mamba-only', 'CfC-only', 'Mamba-CfC']:
            print(f"\nModel: {name}")
            try:
                # Update investigate_pooling to use pool_mode
                results = {}
                for pool in ['mean', 'last']:
                    setup_seed(42)
                    model = cls(**kwargs, pool_mode=pool).to(device)
                    model.train()
                    
                    optimizer = optim.Adam(model.parameters(), lr=1e-3)
                    criterion = nn.CrossEntropyLoss()
                    
                    dummy_input = torch.randn(128, 30, 10).to(device)
                    dummy_target = torch.randint(0, 53, (128,)).to(device)
                    
                    losses = []
                    for step in range(300):
                        optimizer.zero_grad()
                        out = model(dummy_input)
                        loss = criterion(out, dummy_target)
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())
                        
                    results[pool] = losses
                
                for pool, losses in results.items():
                    print(f"  Pool: {pool:5s} | Step 0 Loss: {losses[0]:.4f} | Step 299 Loss: {losses[-1]:.4f}")
            except Exception as e:
                print(f"  Error with model {name}: {e}")
