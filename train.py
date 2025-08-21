import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import XRDNet
from data_utils import load_dataset_from_cifs_enhanced

# 解析命令行参数
parser = argparse.ArgumentParser(description="XRD-AutoAnalyzer-PyTorch")
parser.add_argument("--data_dir", type=str, default="Novel_Space/All_CIFs",
                    help="The directory path containing all reference phase CIF files ")
parser.add_argument("--train_ratio", type=float, default=0.8,
                    help="Training set split ratio, the rest is used for validation (0<ratio<1)")
parser.add_argument("--augment_each", type=int, default=50,
                    help="Number of samples generated for each phase (including original + augmented)")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--epochs", type=int, default=100, help="epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
parser.add_argument("--out_dir", type=str, default=".",
                    help="The output directory for saving the model file and the class mapping.")
args = parser.parse_args()

def create_references_directory(source_dir, references_dir="References"):
    """
    Automatically create and populate the References directory with copies
    of CIF files from the source directory.
    """
    # Create References directory if it doesn't exist
    os.makedirs(references_dir, exist_ok=True)
    print(f"Creating/updating References directory: {references_dir}")
    
    # Get list of CIF files from source directory
    cif_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.cif')]
    
    # Copy each CIF file to the References directory
    for cif_file in cif_files:
        source_path = os.path.join(source_dir, cif_file)
        target_path = os.path.join(references_dir, cif_file)
        
        # Remove existing file if it exists
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass  # Ignore errors when removing
        
        # Copy the file
        try:
            import shutil
            shutil.copy2(source_path, target_path)
            print(f"  Copied file: {cif_file}")
        except Exception as copy_error:
            print(f"  Failed to copy file {cif_file}: {copy_error}")
    
    print(f"References directory populated with {len(cif_files)} CIF files")

print("Pre-creating References directory...")
create_references_directory(args.data_dir)

# Load dataset from CIF files
print("Loading dataset from CIFs in:", args.data_dir)
(train_X, train_y), (val_X, val_y), class_info = load_dataset_from_cifs_enhanced(
    args.data_dir, train_ratio=args.train_ratio, augment_each=args.augment_each)
num_classes = len(class_info["class_names"])
print(f"Total samples: {len(train_X)+len(val_X)}, Classes: {num_classes}")
print(f"Train samples: {len(train_X)}, Val samples: {len(val_X)}")

# numpy to PyTorch tensors
train_X = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_y)
val_X = torch.from_numpy(val_X)
val_y = torch.from_numpy(val_y)

# Ensure the input is in the shape (N, 1, L)
input_length = train_X.shape[2]  # Should be (N, 1, L)
print(f"Input spectrum length: {input_length}")

# Load data into DataLoader
train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# temp_model initialization
temp_model = XRDNet(n_phases=num_classes, dropout_rate=0.7)
flattened_size = temp_model.get_num_features(input_length)
print(f"Flattened feature size: {flattened_size}")

# Create the actual model with dynamic flattened size
model = XRDNet(n_phases=num_classes, dropout_rate=0.7, n_dense=[min(3100, flattened_size), min(1200, flattened_size//2)]).to(device)
del temp_model  
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training loop
best_val_acc = 0.0
best_val_loss = float('inf')
patience_counter = 0
patience_limit = 15

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(train_dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            val_loss += F.cross_entropy(logits, y_batch).item() * X_batch.size(0)
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
    val_loss /= len(val_dataset)
    val_acc = correct / len(val_dataset)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch}/{args.epochs}: TrainLoss={avg_loss:.4f}, ValLoss={val_loss:.4f}, ValAcc={val_acc*100:.2f}%")
    
    # Early stopping based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model based on validation loss
        os.makedirs(args.out_dir, exist_ok=True)
        model_path = os.path.join(args.out_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        # Save class info
        import json
        class_info_path = os.path.join(args.out_dir, "classes.json")
        with open(class_info_path, "w") as f:
            json.dump(class_info, f, indent=2, ensure_ascii=False)
        print(f"  [Info] 模型已更新，保存至 {model_path}, 类别信息保存至 {class_info_path}")
    else:
        patience_counter += 1
        if patience_counter >= patience_limit:
            print(f"  [Info] 验证损失在 {patience_limit} 个epoch内未改善，提前停止训练")
            break
            
    # Save the best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model_path_acc = os.path.join(args.out_dir, "model_best_acc.pt")
        torch.save(model.state_dict(), model_path_acc)
        print(f"  [Info] 基于准确率的最佳模型已保存至 {model_path_acc}")

print(f"训练完成。最佳验证准确率: {best_val_acc*100:.2f}%")
