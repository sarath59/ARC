import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import List, Tuple
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
import math

print("Starting program...")



# ARC Dataset class with data augmentation
print("Starting program...")

# Set device to CPU
device = torch.device("cpu")
print("Using CPU")

# Custom padding function
def custom_pad(tensor_list):
    max_h = max([t.size(0) for t in tensor_list])
    max_w = max([t.size(1) for t in tensor_list])
    padded_list = []
    for tensor in tensor_list:
        h_padding = max_h - tensor.size(0)
        w_padding = max_w - tensor.size(1)
        padded_tensor = F.pad(tensor, (0, w_padding, 0, h_padding))
        padded_list.append(padded_tensor)
    return torch.stack(padded_list)

# ARC Dataset class with data augmentation
class ARCDataset(Dataset):
    def __init__(self, challenges_file, solutions_file=None):
        with open(challenges_file, 'r') as f:
            self.challenges = json.load(f)
        self.task_ids = list(self.challenges.keys())

        if solutions_file:
            with open(solutions_file, 'r') as f:
                self.solutions = json.load(f)
        else:
            self.solutions = {}

        print(f"Number of tasks: {len(self.task_ids)}")

    def __len__(self):
        return len(self.task_ids)

    def __getitem__(self, idx):
        task_id = self.task_ids[idx]
        task = self.challenges[task_id]
        train_input = torch.tensor(task['train'][0]['input'], dtype=torch.float32)
        train_output = torch.tensor(task['train'][0]['output'], dtype=torch.float32)
        test_input = torch.tensor(task['test'][0]['input'], dtype=torch.float32)
        if task_id in self.solutions:
            test_output = torch.tensor(self.solutions[task_id][0], dtype=torch.float32)
        else:
            test_output = torch.zeros_like(test_input)

        # Apply data augmentation
        train_input, train_output = self.augment_data(train_input, train_output)

        return train_input, train_output, test_input, test_output, task_id

    def augment_data(self, input_grid, output_grid):
        # Randomly apply transformations
        if np.random.random() < 0.5:
            input_grid = torch.rot90(input_grid, k=np.random.randint(1, 4))
            output_grid = torch.rot90(output_grid, k=np.random.randint(1, 4))
        if np.random.random() < 0.5:
            input_grid = torch.flip(input_grid, [0])
            output_grid = torch.flip(output_grid, [0])
        if np.random.random() < 0.5:
            input_grid = torch.flip(input_grid, [1])
            output_grid = torch.flip(output_grid, [1])
        return input_grid, output_grid

def collate_fn(batch):
    train_inputs, train_outputs, test_inputs, test_outputs, task_ids = zip(*batch)
    train_inputs_padded = custom_pad(train_inputs)
    train_outputs_padded = custom_pad(train_outputs)
    test_inputs_padded = custom_pad(test_inputs)
    test_outputs_padded = custom_pad(test_outputs)
    return train_inputs_padded, train_outputs_padded, test_inputs_padded, test_outputs_padded, task_ids

# Enhanced ARC Model
class EnhancedARCModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=10):
        super(EnhancedARCModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.input_projection = nn.Linear(1, embed_dim)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.hierarchical = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.output_projection = nn.Linear(embed_dim, 1)

    def forward(self, x, target_shape):
        batch_size, height, width = x.shape
        x = x.view(batch_size, -1, 1)  # Flatten the input
        x = self.input_projection(x)  # Project to embedding dimension
        x = self.encoder(x)           # Apply the Transformer encoder
        x = self.hierarchical(x)      # Hierarchical processing
        x = self.output_projection(x) # Project back to 1 channel
        x = x.view(batch_size, height, width) # Reshape to original spatial dimensions

        # Adjust the output size to match target dimensions
        if (height, width) != (target_shape[1], target_shape[2]):
            x = F.adaptive_avg_pool2d(x.unsqueeze(1), (target_shape[1], target_shape[2])).squeeze(1)
        return x
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Training function
def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()
    for batch_idx, (train_input, train_output, _, _, task_id) in enumerate(train_loader):
        train_input, train_output = train_input.to(device), train_output.to(device)
        optimizer.zero_grad()

        output = model(train_input, train_output.shape)

        # Resize output to match the target shape using interpolation
        output_resized = F.interpolate(output.unsqueeze(1), size=(train_output.size(1), train_output.size(2)), mode='bilinear', align_corners=False).squeeze(1)

        loss = criterion(output_resized, train_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            print(f"Input shape: {train_input.shape}, Resized Output shape: {output_resized.shape}, Target shape: {train_output.shape}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Average Training Loss: {avg_loss:.6f}")
    return avg_loss
# Evaluation function

def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (eval_input, eval_output, _, _, task_id) in enumerate(eval_loader):
            eval_input, eval_output = eval_input.to(device), eval_output.to(device)
            
            # Ensure eval_input is 3D (batch_size, height, width)
            if eval_input.dim() == 2:
                eval_input = eval_input.unsqueeze(0)
            
            output = model(eval_input, eval_output.shape)
            loss = criterion(output, eval_output)
            total_loss += loss.item()

    avg_loss = total_loss / len(eval_loader)
    print(f"Evaluation Loss: {avg_loss:.6f}")
    return avg_loss


# Test function
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (test_input, _, _, test_output, task_id) in enumerate(test_loader):
            test_input, test_output = test_input.to(device), test_output.to(device)
            
            # Ensure test_input is 3D (batch_size, height, width)
            if test_input.dim() == 2:
                test_input = test_input.unsqueeze(0)
            
            output = model(test_input, test_output.shape)
            
            # Round the output to the nearest integer
            output_rounded = output.round()
            
            # Check if the rounded output matches the target
            correct += (output_rounded == test_output).all(dim=(1,2)).sum().item()
            total += test_input.size(0)

            if batch_idx % 10 == 0:
                print(f'Testing Task {task_id[0]}')
                print(f'Input shape: {test_input.shape}, Output shape: {output.shape}, Target shape: {test_output.shape}')

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# Main execution function
# Main execution function
def main():
    print("Entering main function...")
    print("Loading datasets...")
    dataset = ARCDataset('arc-agi_training_challenges.json', 'arc-agi_training_solutions.json')
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    test_data = ARCDataset('arc-agi_test_challenges.json')

    print("Creating data loaders...")
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print("Initializing model...")
    model = EnhancedARCModel(embed_dim=256, num_heads=8, num_layers=9).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    num_epochs = 100
    best_val_loss = float('inf')

    print("Starting training loop...")
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    print("Training completed.")
    print("Starting testing phase...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_accuracy = test(model, test_loader, device)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()