import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report
import numpy as np
import json

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataset import NinaproDataset
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate Models")
    parser.add_argument('--models', nargs='+', default=['mamba-only', 'cfc-only', 'mamba-cfc', 'mamba-3l', 'mamba-width'])
    parser.add_argument('--pool_mode', type=str, default='last')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading test dataset...")
    test_dataset = NinaproDataset('data/processed/test.h5')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    results = {}
    
    for model_name in args.models:
        model_path = f'checkpoints/{model_name}_{args.pool_mode}_best.pt'
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}, checkpoint not found.")
            continue
            
        print(f"\nEvaluating {model_name}...")
        model = get_model(model_name, args.pool_mode, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(all_targets, all_preds)
        
        results[model_name] = {
            'macro_f1': float(macro_f1),
            'balanced_accuracy': float(bal_acc)
        }
        
        print(f"  Macro-F1: {macro_f1:.4f}")
        print(f"  Bal-Acc:  {bal_acc:.4f}")
        
    os.makedirs('reports', exist_ok=True)
    with open('reports/final_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nEvaluation complete. Results saved to reports/final_results.json.")

if __name__ == '__main__':
    main()
