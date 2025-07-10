import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import subprocess

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x

class CMIBehaviorModel(nn.Module):
    def __init__(self, input_channels=337, num_classes=None, seq_len=1000):
        super(CMIBehaviorModel, self).__init__()
        
        # CNN feature extraction layers
        self.cnn_blocks = nn.Sequential(
            CNNBlock(input_channels, 64, kernel_size=7, padding=3),
            nn.MaxPool1d(2),
            CNNBlock(64, 128, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
            CNNBlock(128, 256, kernel_size=3, padding=1),
            nn.MaxPool1d(2),
            CNNBlock(256, 512, kernel_size=3, padding=1),
            nn.MaxPool1d(2)
        )
        
        self.feature_dim = 512
        self.seq_len_after_conv = seq_len // 16
        
        # Attention layers
        self.attention_blocks = nn.Sequential(
            AttentionBlock(self.feature_dim, num_heads=8),
            AttentionBlock(self.feature_dim, num_heads=8)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head - will be set based on task
        self.classifier = None
        if num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
    
    def forward(self, x):
        # CNN feature extraction
        x = self.cnn_blocks(x)
        
        # Prepare for attention
        x = x.transpose(1, 2)
        
        # Apply attention
        x = self.attention_blocks(x)
        
        # Back to CNN format
        x = x.transpose(1, 2)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        if self.classifier:
            x = self.classifier(x)
        
        return x

class CMIDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, mode='train', seq_len=1000, prediction_mode='behavior'):
        self.csv_path = csv_path
        self.mode = mode
        self.seq_len = seq_len
        self.prediction_mode = prediction_mode
        
        # Load data
        self.data = self._load_data_with_sudo()
        
        # Prepare feature columns
        self.feature_columns = self._get_feature_columns()
        
        # Prepare sequences
        self.sequences = self._prepare_sequences()
        
        # Fit scaler on training data
        if mode == 'train' and len(self.sequences) > 0:
            self.scaler = StandardScaler()
            all_features = []
            for seq in self.sequences:
                all_features.extend(seq['features'].tolist())
            
            if len(all_features) > 0:
                all_features = np.array(all_features)
                self.scaler.fit(all_features)
            else:
                self.scaler = None
        
    def _load_data_with_sudo(self):
        try:
            # Use pandas standard CSV reading
            df = pd.read_csv(self.csv_path)
            return df
        except Exception as e:
            print(f"Error loading data from {self.csv_path}: {e}")
            return pd.DataFrame()
    
    def _get_feature_columns(self):
        # Define sensor feature columns
        feature_cols = []
        
        # Accelerometer and rotation
        if 'acc_x' in self.data.columns:
            feature_cols.extend(['acc_x', 'acc_y', 'acc_z'])
        if 'rot_w' in self.data.columns:
            feature_cols.extend(['rot_w', 'rot_x', 'rot_y', 'rot_z'])
        
        # Thermal sensors
        thm_cols = [col for col in self.data.columns if col.startswith('thm_')]
        feature_cols.extend(thm_cols)
        
        # ToF sensors
        tof_cols = [col for col in self.data.columns if col.startswith('tof_')]
        feature_cols.extend(tof_cols)
        
        return feature_cols
    
    def _prepare_sequences(self):
        sequences = []
        
        # Group by sequence_id
        if 'sequence_id' in self.data.columns:
            for seq_id, group in self.data.groupby('sequence_id'):
                if len(group) >= 50:  # Reduce minimum sequence length
                    # Extract features
                    try:
                        features = group[self.feature_columns].values.astype(float)
                        
                        # Handle missing values (-1.0 in ToF data)
                        features = np.where(features == -1.0, 0.0, features)
                        features = np.nan_to_num(features, nan=0.0)
                        
                        # Extract label based on prediction mode
                        if self.prediction_mode == 'behavior' and 'behavior' in group.columns:
                            # Use the most common behavior in the sequence
                            behavior_values = group['behavior'].dropna()
                            if len(behavior_values) > 0:
                                label_str = behavior_values.mode().iloc[0]
                                # Convert behavior string to numeric
                                label = abs(hash(label_str)) % 2  # Simple hash to number 0-1
                            else:
                                label = 0
                        elif self.prediction_mode == 'gesture' and 'gesture' in group.columns:
                            gesture_values = group['gesture'].dropna()
                            if len(gesture_values) > 0:
                                label_str = gesture_values.mode().iloc[0]
                                label = abs(hash(label_str)) % 2
                            else:
                                label = 0
                        else:
                            label = 0
                        
                        sequences.append({
                            'sequence_id': seq_id,
                            'features': features,
                            'label': label
                        })
                    except Exception as e:
                        print(f"Error processing sequence {seq_id}: {e}")
                        continue
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        features = seq['features']
        label = seq['label']
        
        # Normalize features
        if hasattr(self, 'scaler') and self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Convert to tensor and transpose for CNN input
        features_tensor = torch.FloatTensor(features).transpose(0, 1)
        
        # Pad or truncate sequence to seq_len
        if features_tensor.shape[1] < self.seq_len:
            padding = torch.zeros(features_tensor.shape[0], self.seq_len - features_tensor.shape[1])
            features_tensor = torch.cat([features_tensor, padding], dim=1)
        elif features_tensor.shape[1] > self.seq_len:
            features_tensor = features_tensor[:, :self.seq_len]
        
        return features_tensor, torch.LongTensor([int(label)])

def train_model():
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Create dataset
    train_dataset = CMIDataset('/datasets/cmi-detect-behavior-with-sensor-data/train.csv', mode='train')
    
    if len(train_dataset) == 0:
        print("No data loaded. Please check data access.")
        return None
    
    print(f"Loaded {len(train_dataset)} sequences")
    print(f"Feature dimensions: {len(train_dataset.feature_columns)}")
    
    # Determine number of classes and ensure proper range
    labels = [seq['label'] for seq in train_dataset.sequences]
    unique_labels = sorted(set(labels))
    num_classes = max(len(unique_labels), 2)  # Ensure at least 2 classes
    
    # Create label mapping to ensure 0-based indexing
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Update labels in sequences
    for seq in train_dataset.sequences:
        seq['label'] = label_mapping.get(seq['label'], 0)
    
    print(f"Number of classes: {num_classes}")
    print(f"Label mapping: {label_mapping}")
    print(f"Label range: 0 to {num_classes-1}")
    
    # Create model
    model = CMIBehaviorModel(
        input_channels=len(train_dataset.feature_columns),
        num_classes=num_classes,
        seq_len=train_dataset.seq_len
    )
    model.to(device)
    
    # Create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=0
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    num_epochs = 10
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), '/notebooks/cmidetect/cmi_real_data_model.pth')
    print("Model saved successfully")
    
    return model

if __name__ == "__main__":
    model = train_model()