import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score
import json

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataset import NinaproDataset, get_balanced_sampler
from src.models.architecture import MambaOnlyModel, CfCOnlyModel, MambaCfCModel

def get_model(model_name, pool_mode='last', device='cuda'):
    input_dim = 10
    num_classes = 53
    
    if model_name == 'mamba-only':
        model = MambaOnlyModel(input_dim, num_classes, d_model=64, num_layers=2, pool_mode=pool_mode)
    elif model_name == 'cfc-only':
        model = CfCOnlyModel(input_dim, num_classes, d_model=64, pool_mode=pool_mode)
    elif model_name == 'mamba-cfc':
        model = MambaCfCModel(input_dim, num_classes, d_model=64, num_layers=2, pool_mode=pool_mode)
    elif model_name == 'mamba-3l':
        model = MambaOnlyModel(input_dim, num_classes, d_model=64, num_layers=3, pool_mode=pool_mode)
    elif model_name == 'mamba-width':
        model = MambaOnlyModel(input_dim, num_classes, d_model=88, num_layers=2, pool_mode=pool_mode)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return model.to(device)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())
        
    import numpy as np
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    avg_loss = total_loss / len(dataloader.dataset)
    
    return avg_loss, macro_f1, bal_acc

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            
    import numpy as np
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    avg_loss = total_loss / len(dataloader.dataset)
    
    return avg_loss, macro_f1, bal_acc

def main():
    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument('--model', type=str, required=True, choices=['mamba-only', 'cfc-only', 'mamba-cfc', 'mamba-3l', 'mamba-width'])
    parser.add_argument('--pool_mode', type=str, default='last', choices=['mean', 'last'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run for 1 batch to test pipeline')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_dataset = NinaproDataset('data/processed/train.h5')
    val_dataset = NinaproDataset('data/processed/val.h5')
    
    if args.fast_dev_run:
        train_dataset.X = train_dataset.X[:500]
        train_dataset.labels = train_dataset.labels[:500]
        val_dataset.X = val_dataset.X[:500]
        val_dataset.labels = val_dataset.labels[:500]
        args.epochs = 2
    
    print("Setting up dataloaders...")
    train_sampler = get_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = get_model(args.model, args.pool_mode, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Starting training for {args.model} with {args.pool_mode} pooling...")
    best_val_f1 = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        train_loss, train_f1, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} [{elapsed:.1f}s]")
        print(f"  Train Loss: {train_loss:.4f} | Macro-F1: {train_f1:.4f} | Bal-Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Macro-F1: {val_f1:.4f} | Bal-Acc: {val_acc:.4f}", flush=True)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'checkpoints/{args.model}_{args.pool_mode}_best.pt')
            print("  [*] Saved new best model", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.", flush=True)
                break
            
    # Save history
    with open(f'reports/{args.model}_{args.pool_mode}_history.json', 'w') as f:
        json.dump(history, f)
        
if __name__ == '__main__':
    main()
