#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
import socket
import warnings
warnings.filterwarnings('ignore')

# Define paths
DATA_ROOT = "."  # Current directory as root
PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, "processed_data")
FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, "features")
METADATA_DIR = os.path.join(PROCESSED_DATA_DIR, "metadata")
MODELS_DIR = os.path.join(DATA_ROOT, "models")
RESULTS_DIR = os.path.join(DATA_ROOT, "results")

# Create directories if they don't exist
for directory in [MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Environment variables for better performance
os.environ["NCCL_DEBUG"] = "WARN"  # Changed from INFO to WARN to reduce output
os.environ["NCCL_SOCKET_IFNAME"] = "lo"  # Use loopback interface for localhost
os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand if not needed

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

#----------------------------
# Dataset Class
#----------------------------
class AnomalyDetectionDataset(Dataset):
    """PyTorch dataset for anomaly detection using HDF5 features"""
    
    def __init__(self, h5_file, video_names=None, transform=None, target_transform=None):
        """
        Initialize dataset - but don't open the h5 file yet
        Args:
            h5_file: Path to HDF5 file with features
            video_names: List of video names to include (None for all)
            transform: Transform to apply to features
            target_transform: Transform to apply to labels
        """
        self.h5_file = h5_file
        self.transform = transform
        self.target_transform = target_transform
        self.videos = []
        self.feature_dim = None
        self.label_map = {}
        self.idx_to_label = {}
        
        # Load metadata without keeping the h5 file open
        self._load_metadata(video_names)
        
    def _load_metadata(self, video_names=None):
        """Load metadata from h5 file without keeping it open"""
        with h5py.File(self.h5_file, 'r') as h5:
            # Get list of videos
            all_videos = list(h5['videos'].keys())
            
            # Filter videos if video_names is provided
            if video_names is not None:
                # Check if we need to handle name mapping (for duplicate names)
                if 'name_map' in h5['metadata']:
                    # Create mapping between original names and unique names in h5 file
                    name_map = {}
                    for unique_name in all_videos:
                        original_name = h5['videos'][unique_name].attrs.get('original_video_name', unique_name)
                        name_map[original_name] = unique_name
                    
                    # Filter using the provided names but map to unique names
                    for name in video_names:
                        if name in name_map:
                            self.videos.append(name_map[name])
                        elif name in all_videos:  # If the name is already a unique name
                            self.videos.append(name)
                else:
                    # No name mapping needed
                    self.videos = [v for v in all_videos if v in video_names]
            else:
                self.videos = all_videos
            
            # Get feature dimension
            self.feature_dim = h5['metadata'].attrs['feature_dim']
            
            # Get label mapping
            for label, idx in h5['metadata']['label_map'].attrs.items():
                self.label_map[label] = idx
                
            self.idx_to_label = {idx: label for label, idx in self.label_map.items()}
    
    def __len__(self):
        """Get dataset length"""
        return len(self.videos)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset - open and close h5 file each time"""
        with h5py.File(self.h5_file, 'r') as h5:
            video_name = self.videos[idx]
            video_group = h5['videos'][video_name]
            
            # Get features
            features = video_group['video_features_mean_norm'][...]
            
            # Get label
            label_idx = video_group.attrs['label_idx']
            is_anomaly = video_group.attrs['is_anomaly']
            
            # Get original video name if it exists
            original_name = video_group.attrs.get('original_video_name', video_name)
        
        # Apply transforms if provided
        if self.transform:
            features = self.transform(features)
        
        if self.target_transform:
            label_idx = self.target_transform(label_idx)
            is_anomaly = self.target_transform(is_anomaly)
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        label_idx = torch.tensor(label_idx, dtype=torch.long)
        is_anomaly = torch.tensor(is_anomaly, dtype=torch.float)
        
        return features, label_idx, is_anomaly, original_name

#----------------------------
# Model Definitions
#----------------------------

class AnomalyDetector(nn.Module):
    """
    Multi-task model for anomaly detection and action classification
    with BCEWithLogitsLoss (no sigmoid in forward) for mixed precision compatibility
    """
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.5):
        super(AnomalyDetector, self).__init__()
        
        # Shared layers
        shared_layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims[:-1]:
            shared_layers.append(nn.Linear(prev_dim, dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        self.shared_features = nn.Sequential(*shared_layers)
        
        # Task-specific layers - NOTE: removed sigmoid for anomaly classifier
        # The sigmoid will be part of the BCEWithLogitsLoss
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], 1)  # No sigmoid here
        )
        
        self.action_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], num_classes)
        )
    
    def forward(self, x):
        # Apply shared layers
        shared_features = self.shared_features(x)
        
        # Apply task-specific layers
        anomaly_logits = self.anomaly_classifier(shared_features).squeeze()
        action_pred = self.action_classifier(shared_features)
        
        return anomaly_logits, action_pred

#----------------------------
# Training Functions
#----------------------------

def setup_distributed(rank, world_size, port=12355, max_retries=10):
    """
    Initialize distributed training with retries for port availability
    Args:
        rank: Process rank
        world_size: Number of processes
        port: Starting port number for communication
        max_retries: Maximum number of port retries
    """
    # Try multiple ports with retries
    original_port = port
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Set up the master address and port
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(port)
            
            # Check if port is available before initializing
            if rank == 0:  # Only rank 0 needs to check
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.bind(('localhost', port))
                    s.close()
                    print(f"[rank{rank}]: Port {port} is available")
                except socket.error:
                    # Port is in use, try the next one
                    print(f"[rank{rank}]: Port {port} is in use, trying next port")
                    port += 1
                    retry_count += 1
                    continue
            
            # Initialize the process group with timeout
            dist.init_process_group(
                backend='nccl',  # Use NCCL for GPU-to-GPU communication
                init_method=f"tcp://localhost:{port}",
                world_size=world_size,
                rank=rank,
                timeout=timedelta(minutes=5)  # Longer timeout for FSDP
            )
            
            print(f"[rank{rank}]: Successfully initialized process group on port {port}")
            # Set GPU
            torch.cuda.set_device(rank)
            return True
            
        except Exception as e:
            retry_count += 1
            port = original_port + retry_count
            print(f"[rank{rank}]: Failed to initialize process group (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                time.sleep(2)  # Wait before retrying
            else:
                print(f"[rank{rank}]: Max retries reached. Could not initialize process group.")
                raise

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def train_epoch(model, loader, optimizer, criterion_anomaly, criterion_action, device, epoch, total_epochs, scaler=None):
    """
    Train for one epoch with mixed precision if enabled
    """
    model.train()
    
    loss_values = []
    anomaly_loss_values = []
    action_loss_values = []
    
    correct_anomaly = 0
    total_anomaly = 0
    correct_action = 0
    total_action = 0
    
    # Get rank for progress display
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Display progress only on rank 0
    if rank == 0:
        progress = tqdm(enumerate(loader), total=len(loader), 
                      desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    else:
        progress = enumerate(loader)
    
    # Initialize optimizer
    optimizer.zero_grad()
    
    # Training loop
    for i, (features, action_labels, anomaly_labels, _) in progress:
        # Move to device
        features = features.to(device, non_blocking=True)
        action_labels = action_labels.to(device, non_blocking=True)
        anomaly_labels = anomaly_labels.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                anomaly_logits, action_predictions = model(features)
                
                # Calculate losses - Note we're now using logits (without sigmoid)
                # The sigmoid is applied in BCEWithLogitsLoss
                anomaly_loss = criterion_anomaly(anomaly_logits, anomaly_labels)
                action_loss = criterion_action(action_predictions, action_labels)
                
                # Combined loss with weighting
                loss = 0.7 * anomaly_loss + 0.3 * action_loss
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            
            # Update weights with scaled gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # Regular forward pass without mixed precision
            anomaly_logits, action_predictions = model(features)
            
            # Calculate losses
            anomaly_loss = criterion_anomaly(anomaly_logits, anomaly_labels)
            action_loss = criterion_action(action_predictions, action_labels)
            
            # Combined loss with weighting
            loss = 0.7 * anomaly_loss + 0.3 * action_loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
        
        # Record losses
        loss_values.append(loss.item())
        anomaly_loss_values.append(anomaly_loss.item())
        action_loss_values.append(action_loss.item())
        
        # Calculate accuracy
        with torch.no_grad():
            # Anomaly detection accuracy - apply sigmoid here for evaluation
            anomaly_predictions = torch.sigmoid(anomaly_logits)
            anomaly_pred_binary = (anomaly_predictions >= 0.5).float()
            correct_anomaly += (anomaly_pred_binary == anomaly_labels).sum().item()
            total_anomaly += anomaly_labels.size(0)
            
            # Action classification accuracy
            _, action_pred = torch.max(action_predictions, 1)
            correct_action += (action_pred == action_labels).sum().item()
            total_action += action_labels.size(0)
        
        # Update progress bar on rank 0
        if rank == 0 and (i % 10 == 0 or i == len(loader) - 1):
            avg_loss = np.mean(loss_values[-10:]) if len(loss_values) >= 10 else np.mean(loss_values)
            anomaly_acc = correct_anomaly / total_anomaly if total_anomaly > 0 else 0
            action_acc = correct_action / total_action if total_action > 0 else 0
            
            progress.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'ano_acc': f"{anomaly_acc:.4f}",
                'act_acc': f"{action_acc:.4f}"
            })
    
    # Calculate epoch statistics
    epoch_loss = np.mean(loss_values)
    epoch_anomaly_loss = np.mean(anomaly_loss_values)
    epoch_action_loss = np.mean(action_loss_values)
    
    epoch_anomaly_acc = correct_anomaly / total_anomaly if total_anomaly > 0 else 0
    epoch_action_acc = correct_action / total_action if total_action > 0 else 0
    
    return {
        'loss': epoch_loss,
        'anomaly_loss': epoch_anomaly_loss,
        'action_loss': epoch_action_loss,
        'anomaly_acc': epoch_anomaly_acc,
        'action_acc': epoch_action_acc
    }

def validate(model, loader, criterion_anomaly, criterion_action, device, scaler=None):
    """
    Validate model
    """
    model.eval()
    
    loss_values = []
    anomaly_loss_values = []
    action_loss_values = []
    
    correct_anomaly = 0
    total_anomaly = 0
    correct_action = 0
    total_action = 0
    
    all_anomaly_labels = []
    all_anomaly_predictions = []
    all_action_labels = []
    all_action_predictions = []
    
    with torch.no_grad():
        # Get rank for progress display
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Display progress only on rank 0
        if rank == 0:
            progress = tqdm(loader, desc="Validation")
        else:
            progress = loader
        
        for features, action_labels, anomaly_labels, _ in progress:
            # Move to device
            features = features.to(device, non_blocking=True)
            action_labels = action_labels.to(device, non_blocking=True)
            anomaly_labels = anomaly_labels.to(device, non_blocking=True)
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    anomaly_logits, action_predictions = model(features)
                    
                    # Calculate losses
                    anomaly_loss = criterion_anomaly(anomaly_logits, anomaly_labels)
                    action_loss = criterion_action(action_predictions, action_labels)
                    
                    # Combined loss with weighting
                    loss = 0.7 * anomaly_loss + 0.3 * action_loss
            else:
                # Regular forward pass without mixed precision
                anomaly_logits, action_predictions = model(features)
                
                # Calculate losses
                anomaly_loss = criterion_anomaly(anomaly_logits, anomaly_labels)
                action_loss = criterion_action(action_predictions, action_labels)
                
                # Combined loss with weighting
                loss = 0.7 * anomaly_loss + 0.3 * action_loss
            
            # Record losses
            loss_values.append(loss.item())
            anomaly_loss_values.append(anomaly_loss.item())
            action_loss_values.append(action_loss.item())
            
            # Calculate accuracy
            # Apply sigmoid for anomaly detection
            anomaly_predictions = torch.sigmoid(anomaly_logits)
            anomaly_pred_binary = (anomaly_predictions >= 0.5).float()
            correct_anomaly += (anomaly_pred_binary == anomaly_labels).sum().item()
            total_anomaly += anomaly_labels.size(0)
            
            # Action classification accuracy
            _, action_pred = torch.max(action_predictions, 1)
            correct_action += (action_pred == action_labels).sum().item()
            total_action += action_labels.size(0)
            
            # Store predictions and labels for metrics
            all_anomaly_labels.append(anomaly_labels.cpu().numpy())
            all_anomaly_predictions.append(anomaly_predictions.cpu().numpy())
            all_action_labels.append(action_labels.cpu().numpy())
            all_action_predictions.append(torch.softmax(action_predictions, dim=1).cpu().numpy())
    
    # Calculate validation statistics
    val_loss = np.mean(loss_values)
    val_anomaly_loss = np.mean(anomaly_loss_values)
    val_action_loss = np.mean(action_loss_values)
    
    val_anomaly_acc = correct_anomaly / total_anomaly if total_anomaly > 0 else 0
    val_action_acc = correct_action / total_action if total_action > 0 else 0
    
    # Combine all predictions and labels
    all_anomaly_labels = np.concatenate(all_anomaly_labels) if all_anomaly_labels else np.array([])
    all_anomaly_predictions = np.concatenate(all_anomaly_predictions) if all_anomaly_predictions else np.array([])
    all_action_labels = np.concatenate(all_action_labels) if all_action_labels else np.array([])
    all_action_predictions = np.concatenate(all_action_predictions) if all_action_predictions else np.array([])
    
    # Calculate F1 score for anomaly detection
    anomaly_pred_binary = (all_anomaly_predictions >= 0.5).astype(float)
    
    try:
        anomaly_f1 = f1_score(all_anomaly_labels, anomaly_pred_binary)
        anomaly_precision = precision_score(all_anomaly_labels, anomaly_pred_binary)
        anomaly_recall = recall_score(all_anomaly_labels, anomaly_pred_binary)
    except:
        anomaly_f1 = 0.0
        anomaly_precision = 0.0
        anomaly_recall = 0.0
    
    return {
        'loss': val_loss,
        'anomaly_loss': val_anomaly_loss,
        'action_loss': val_action_loss,
        'anomaly_acc': val_anomaly_acc,
        'action_acc': val_action_acc,
        'anomaly_f1': anomaly_f1,
        'anomaly_precision': anomaly_precision,
        'anomaly_recall': anomaly_recall,
        'all_anomaly_labels': all_anomaly_labels,
        'all_anomaly_predictions': all_anomaly_predictions,
        'all_action_labels': all_action_labels,
        'all_action_predictions': all_action_predictions
    }

def train_model_fsdp(
    rank, 
    world_size,
    h5_file,
    train_video_names,
    test_video_names,
    num_classes,
    model_save_path,
    results_save_path,
    epochs=30,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=1e-4,
    hidden_dims=[1024, 512, 256, 128],
    dropout_rate=0.5,
    patience=5,
    port=12355,
    use_mixed_precision=True,
    sharding_strategy="FULL_SHARD"
):
    """
    Train the model using Fully Sharded Data Parallel (FSDP)
    """
    try:
        # Set up distributed training with port retry logic
        setup_distributed(rank, world_size, port)
        
        # Print info about current process
        print(f"[rank{rank}]: Starting FSDP training with world size {world_size}")
        
        # Set device
        device = torch.device(f'cuda:{rank}')
        
        # Create datasets
        train_dataset = AnomalyDetectionDataset(h5_file, train_video_names)
        test_dataset = AnomalyDetectionDataset(h5_file, test_video_names)
        
        # Get feature dimension
        feature_dim = train_dataset.feature_dim
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        # Calculate per-GPU batch size
        samples_per_gpu = max(32, batch_size // world_size)
        effective_batch_size = samples_per_gpu * world_size
        
        if rank == 0:
            print(f"Batch size: {batch_size}, Per-GPU batch size: {samples_per_gpu}, "
                  f"Effective batch size: {effective_batch_size}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=samples_per_gpu,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=samples_per_gpu * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Set up mixed precision
        if use_mixed_precision:
            # Configure mixed precision policy
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                buffer_dtype=torch.float16,
                reduce_dtype=torch.float16
            )
            
            # Create gradient scaler for mixed precision training
            scaler = torch.cuda.amp.GradScaler()
        else:
            mixed_precision_policy = None
            scaler = None
        
        # Set sharding strategy
        if sharding_strategy == "FULL_SHARD":
            strategy = ShardingStrategy.FULL_SHARD
        elif sharding_strategy == "SHARD_GRAD_OP":
            strategy = ShardingStrategy.SHARD_GRAD_OP
        elif sharding_strategy == "NO_SHARD":
            strategy = ShardingStrategy.NO_SHARD
        else:
            strategy = ShardingStrategy.FULL_SHARD
        
        # Create model
        model = AnomalyDetector(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        ).to(device)
        
        # Wrap model with FSDP
        model = FSDP(
            model,
            device_id=rank,
            sharding_strategy=strategy,
            mixed_precision=mixed_precision_policy
        )
        
        # Define loss functions - NOTE: Using BCEWithLogitsLoss instead of BCELoss
        # This is safe to use with autocast for mixed-precision training
        criterion_anomaly = nn.BCEWithLogitsLoss()
        criterion_action = nn.CrossEntropyLoss()
        
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Define learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True if rank == 0 else False
        )
        
        # Training loop
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_stats = []
        val_stats = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Set epoch for distributed sampler
            train_sampler.set_epoch(epoch)
            
            # Train for one epoch
            train_epoch_stats = train_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion_anomaly, 
                criterion_action, 
                device,
                epoch,
                epochs,
                scaler
            )
            
            # Validate
            val_epoch_stats = validate(
                model, 
                test_loader, 
                criterion_anomaly, 
                criterion_action, 
                device,
                scaler
            )
            
            # Update learning rate
            scheduler.step(val_epoch_stats['loss'])
            
            # Record training and validation stats
            train_stats.append(train_epoch_stats)
            val_stats.append(val_epoch_stats)
            
            # Print stats on rank 0
            if rank == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train loss: {train_epoch_stats['loss']:.4f}, Anomaly acc: {train_epoch_stats['anomaly_acc']:.4f}, Action acc: {train_epoch_stats['action_acc']:.4f}")
                print(f"  Val loss: {val_epoch_stats['loss']:.4f}, Anomaly acc: {val_epoch_stats['anomaly_acc']:.4f}, Action acc: {val_epoch_stats['action_acc']:.4f}")
                print(f"  Anomaly F1: {val_epoch_stats['anomaly_f1']:.4f}, Precision: {val_epoch_stats['anomaly_precision']:.4f}, Recall: {val_epoch_stats['anomaly_recall']:.4f}")
            
            # Check if this is the best model
            if val_epoch_stats['loss'] < best_val_loss:
                best_val_loss = val_epoch_stats['loss']
                early_stop_counter = 0
                
                # Save best model on rank 0
                if rank == 0:
                    try:
                        # Save model metadata
                        save_state = {
                            'epoch': epoch,
                            'train_loss': train_epoch_stats['loss'],
                            'val_loss': val_epoch_stats['loss'],
                            'val_anomaly_acc': val_epoch_stats['anomaly_acc'],
                            'val_action_acc': val_epoch_stats['action_acc'],
                            'val_anomaly_f1': val_epoch_stats['anomaly_f1'],
                            'model_config': {
                                'input_dim': feature_dim,
                                'hidden_dims': hidden_dims,
                                'num_classes': num_classes,
                                'dropout_rate': dropout_rate
                            }
                        }
                        
                        # Save metadata
                        torch.save(save_state, model_save_path)
                        print(f"Saved best model metadata with val_loss: {best_val_loss:.4f}")
                        
                    except Exception as e:
                        print(f"Error saving model: {e}")
            else:
                early_stop_counter += 1
                
                if early_stop_counter >= patience:
                    if rank == 0:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Save training and validation stats on rank 0
        if rank == 0:
            # Convert train_stats and val_stats to DataFrames
            train_df = pd.DataFrame(train_stats)
            val_df = pd.DataFrame([{k: v for k, v in stats.items() if not isinstance(v, np.ndarray)} for stats in val_stats])
            
            # Create dictionary with final results
            final_results = {
                'num_gpus': world_size,
                'training_method': 'FSDP',
                'sharding_strategy': sharding_strategy,
                'mixed_precision': use_mixed_precision,
                'epochs': epoch + 1,
                'batch_size': batch_size,
                'effective_batch_size': effective_batch_size,
                'per_gpu_batch_size': samples_per_gpu,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout_rate,
                'feature_dim': feature_dim,
                'num_classes': num_classes,
                'training_time': training_time,
                'training_time_per_epoch': training_time / (epoch + 1),
                'best_val_loss': best_val_loss,
                'final_train_loss': train_stats[-1]['loss'],
                'final_val_loss': val_stats[-1]['loss'],
                'final_train_anomaly_acc': train_stats[-1]['anomaly_acc'],
                'final_val_anomaly_acc': val_stats[-1]['anomaly_acc'],
                'final_train_action_acc': train_stats[-1]['action_acc'],
                'final_val_action_acc': val_stats[-1]['action_acc'],
                'final_val_anomaly_f1': val_stats[-1]['anomaly_f1'],
                'final_val_anomaly_precision': val_stats[-1]['anomaly_precision'],
                'final_val_anomaly_recall': val_stats[-1]['anomaly_recall'],
            }
            
            # Save final results and training/validation DataFrames
            with open(results_save_path, 'w') as f:
                json.dump(final_results, f, indent=2, cls=NumpyEncoder)
            
            train_df.to_csv(os.path.join(os.path.dirname(results_save_path), f'train_stats_fsdp_{world_size}gpu.csv'), index=False)
            val_df.to_csv(os.path.join(os.path.dirname(results_save_path), f'val_stats_fsdp_{world_size}gpu.csv'), index=False)
            
            print(f"Training complete in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
            print(f"Results saved to {results_save_path}")
    except Exception as e:
        print(f"[rank{rank}]: Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up distributed training
        cleanup_distributed()

def run_fsdp_benchmark(
    h5_file,
    split_stats_file,
    gpu_counts=[1, 2, 4],
    epochs=30,
    batch_size=128,
    learning_rate=0.001,
    weight_decay=1e-4,
    hidden_dims=[1024, 512, 256, 128],
    dropout_rate=0.5,
    patience=5,
    use_mixed_precision=True,
    sharding_strategy="FULL_SHARD"
):
    """
    Run training benchmark with FSDP using different numbers of GPUs
    """
    # Load split stats
    with open(split_stats_file, 'r') as f:
        split_stats = json.load(f)
    
    train_video_names = split_stats['train_video_names']
    test_video_names = split_stats['test_video_names']
    
    # Get number of classes without keeping dataset open
    temp_dataset = AnomalyDetectionDataset(h5_file, train_video_names)
    num_classes = len(temp_dataset.label_map)
    dataset_size = len(train_video_names)
    del temp_dataset  # Explicitly delete to free resources
    
    print(f"Training FSDP benchmark with GPU counts: {gpu_counts}")
    print(f"Dataset: {h5_file}")
    print(f"Training videos: {len(train_video_names)}")
    print(f"Testing videos: {len(test_video_names)}")
    print(f"Number of classes: {num_classes}")
    print(f"Mixed precision: {use_mixed_precision}")
    print(f"Sharding strategy: {sharding_strategy}")
    
    # Optimize GPU counts for small datasets
    available_gpus = torch.cuda.device_count()
    optimized_gpu_counts = []
    
    for gpu_count in gpu_counts:
        # For very small datasets, limit GPU count to maintain at least 10 samples per GPU
        min_samples_per_gpu = dataset_size // gpu_count
        if dataset_size < 200 and min_samples_per_gpu < 10:
            print(f"Warning: Dataset is too small for efficient training with {gpu_count} GPUs.")
            if gpu_count > 2:  # Still test with 1-2 GPUs regardless
                print(f"Skipping {gpu_count} GPU configuration.")
                continue
        
        # Make sure we have enough GPUs available
        if gpu_count > available_gpus:
            print(f"Warning: Requested {gpu_count} GPUs but only {available_gpus} available.")
            print(f"Skipping this configuration.")
            continue
            
        optimized_gpu_counts.append(gpu_count)
    
    if not optimized_gpu_counts:
        print("No valid GPU configurations to test!")
        return [], {}
    
    # Run training for each GPU count
    benchmark_results = []
    
    # Use a different base port for each run to avoid address reuse issues
    base_port = 29500  # Higher port number less likely to be in use
    
    for i, gpu_count in enumerate(optimized_gpu_counts):
        port = base_port + (i * 100)  # Use large intervals between ports
        
        print(f"\n{'='*80}")
        print(f"FSDP Training with {gpu_count} GPUs (port: {port})")
        print(f"{'='*80}")
        
        # Create paths for model and results
        model_path = os.path.join(MODELS_DIR, f"anomaly_detector_fsdp_{gpu_count}gpu.pt")
        results_path = os.path.join(RESULTS_DIR, f"training_results_fsdp_{gpu_count}gpu.json")
        
        # Scale batch size with GPU count
        effective_batch_size = batch_size
        if gpu_count > 1:
            effective_batch_size = min(batch_size * 2, batch_size * gpu_count)
            print(f"Scaling batch size to {effective_batch_size} for {gpu_count} GPUs")
        
        try:
            # Make sure spawn is set for CUDA support
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, ignore
            pass
        
        try:
            # Launch distributed training
            mp.spawn(
                train_model_fsdp,
                args=(
                    gpu_count,
                    h5_file,
                    train_video_names,
                    test_video_names,
                    num_classes,
                    model_path,
                    results_path,
                    epochs,
                    effective_batch_size,
                    learning_rate,
                    weight_decay,
                    hidden_dims,
                    dropout_rate,
                    patience,
                    port,
                    use_mixed_precision,
                    sharding_strategy
                ),
                nprocs=gpu_count,
                join=True
            )
            
            # Add delay between runs
            if gpu_count > 1:
                print(f"Waiting for resources to be released...")
                time.sleep(20)  # Wait 20 seconds between runs
        
        except Exception as e:
            print(f"Error during FSDP training with {gpu_count} GPUs: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Force cleanup
            time.sleep(10)
        
        # Load training results if file exists
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            benchmark_results.append(results)
        else:
            print(f"Warning: Results file {results_path} not found. Skipping in benchmark summary.")
    
    # Create benchmark summary
    benchmark_summary = {
        'gpu_counts': [],
        'training_times': [],
        'training_times_per_epoch': [],
        'best_val_losses': [],
        'final_val_anomaly_accs': [],
        'final_val_action_accs': [],
        'final_val_anomaly_f1s': [],
        'speedups': []
    }
    
    if benchmark_results:
        # Calculate speedup relative to 1 GPU
        single_gpu_time = next((r['training_time'] for r in benchmark_results if r['num_gpus'] == 1), None)
        
        for results in benchmark_results:
            gpu_count = results['num_gpus']
            training_time = results['training_time']
            
            benchmark_summary['gpu_counts'].append(gpu_count)
            benchmark_summary['training_times'].append(training_time)
            benchmark_summary['training_times_per_epoch'].append(results['training_time_per_epoch'])
            benchmark_summary['best_val_losses'].append(results['best_val_loss'])
            benchmark_summary['final_val_anomaly_accs'].append(results['final_val_anomaly_acc'])
            benchmark_summary['final_val_action_accs'].append(results['final_val_action_acc'])
            benchmark_summary['final_val_anomaly_f1s'].append(results['final_val_anomaly_f1'])
            
            # Calculate speedup
            if single_gpu_time is not None:
                speedup = single_gpu_time / training_time
            else:
                speedup = 1.0  # No speedup for single GPU
            
            benchmark_summary['speedups'].append(speedup)
        
        # Create benchmark summary dataframe
        benchmark_df = pd.DataFrame({
            'GPU Count': benchmark_summary['gpu_counts'],
            'Training Time (s)': benchmark_summary['training_times'],
            'Time Per Epoch (s)': benchmark_summary['training_times_per_epoch'],
            'Best Val Loss': benchmark_summary['best_val_losses'],
            'Val Anomaly Acc': benchmark_summary['final_val_anomaly_accs'],
            'Val Action Acc': benchmark_summary['final_val_action_accs'],
            'Val Anomaly F1': benchmark_summary['final_val_anomaly_f1s'],
            'Speedup': benchmark_summary['speedups']
        })
        
        # Save benchmark summary
        benchmark_df.to_csv(os.path.join(RESULTS_DIR, 'fsdp_benchmark_summary.csv'), index=False)
        
        # Save full benchmark results with custom encoder
        with open(os.path.join(RESULTS_DIR, 'fsdp_benchmark_results.json'), 'w') as f:
            json.dump(benchmark_results, f, indent=2, cls=NumpyEncoder)
        
        create_benchmark_visualizations(benchmark_df, benchmark_summary, prefix="fsdp")
        
        print("\nFSDP Benchmark Summary:")
        print(benchmark_df)
        print(f"\nFSDP Benchmark results saved to {RESULTS_DIR}")
    else:
        print("No benchmark results were collected.")
    
    return benchmark_results, benchmark_summary

def create_benchmark_visualizations(benchmark_df, benchmark_summary, prefix=""):
    """
    Create visualizations for benchmark results
    Args:
        benchmark_df: DataFrame with benchmark results
        benchmark_summary: Dictionary with benchmark summary
        prefix: Prefix for file names
    """
    if not benchmark_df.empty:
        prefix = f"{prefix}_" if prefix else ""
        
        # Plot training time vs GPU count
        plt.figure(figsize=(10, 6))
        plt.plot(benchmark_summary['gpu_counts'], benchmark_summary['training_times'], 'o-', linewidth=2)
        plt.xlabel('Number of GPUs')
        plt.ylabel('Training Time (seconds)')
        plt.title('FSDP Training Time vs Number of GPUs')
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, f'{prefix}training_time_vs_gpus.png'))
        plt.close()
        
        # Plot speedup vs GPU count
        plt.figure(figsize=(10, 6))
        plt.plot(benchmark_summary['gpu_counts'], benchmark_summary['speedups'], 'o-', linewidth=2, label='Actual Speedup')
        plt.plot(benchmark_summary['gpu_counts'], benchmark_summary['gpu_counts'], '--', linewidth=2, label='Ideal Linear Speedup')
        plt.xlabel('Number of GPUs')
        plt.ylabel('Speedup')
        plt.title('FSDP Speedup vs Number of GPUs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, f'{prefix}speedup_vs_gpus.png'))
        plt.close()
        
        # Plot accuracy and F1 vs GPU count
        plt.figure(figsize=(10, 6))
        plt.plot(benchmark_summary['gpu_counts'], benchmark_summary['final_val_anomaly_accs'], 'o-', linewidth=2, label='Anomaly Detection Accuracy')
        plt.plot(benchmark_summary['gpu_counts'], benchmark_summary['final_val_action_accs'], 's-', linewidth=2, label='Action Classification Accuracy')
        plt.plot(benchmark_summary['gpu_counts'], benchmark_summary['final_val_anomaly_f1s'], '^-', linewidth=2, label='Anomaly Detection F1 Score')
        plt.xlabel('Number of GPUs')
        plt.ylabel('Metric Value')
        plt.title('FSDP Model Performance vs Number of GPUs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, f'{prefix}performance_vs_gpus.png'))
        plt.close()
        
        # Calculate training efficiency (speedup / GPU count)
        efficiency = [s / g for s, g in zip(benchmark_summary['speedups'], benchmark_summary['gpu_counts'])]
        
        plt.figure(figsize=(10, 6))
        plt.plot(benchmark_summary['gpu_counts'], efficiency, 'o-', linewidth=2)
        plt.xlabel('Number of GPUs')
        plt.ylabel('Efficiency (Speedup / GPU Count)')
        plt.title('FSDP Training Efficiency vs Number of GPUs')
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, f'{prefix}efficiency_vs_gpus.png'))
        plt.close()
        
        # Create combined visualization - Training time and speedup
        plt.figure(figsize=(12, 8))
        
        # Left y-axis for training time
        ax1 = plt.gca()
        ax1.plot(benchmark_summary['gpu_counts'], benchmark_summary['training_times'], 'o-', color='blue', linewidth=2, label='Training Time')
        ax1.set_xlabel('Number of GPUs')
        ax1.set_ylabel('Training Time (seconds)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Right y-axis for speedup
        ax2 = ax1.twinx()
        ax2.plot(benchmark_summary['gpu_counts'], benchmark_summary['speedups'], 's-', color='red', linewidth=2, label='Actual Speedup')
        ax2.plot(benchmark_summary['gpu_counts'], benchmark_summary['gpu_counts'], '--', color='green', linewidth=2, label='Ideal Speedup')
        ax2.set_ylabel('Speedup', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        
        plt.title('FSDP Training Time and Speedup vs Number of GPUs')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{prefix}time_and_speedup.png'))
        plt.close()
    else:
        print("No data to create visualizations")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FSDP Multi-GPU Training for Shoplifting Detection')
    parser.add_argument('--h5_file', type=str, default=os.path.join(PROCESSED_DATA_DIR, "ucf_crime_features.h5"),
                       help='Path to HDF5 file with features')
    parser.add_argument('--split_stats', type=str, default=os.path.join(METADATA_DIR, "train_test_split.json"),
                       help='Path to train/test split JSON')
    parser.add_argument('--gpu_counts', type=int, nargs='+', default=[1, 2, 4],
                       help='GPU counts to benchmark')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--mixed_precision', dest='mixed_precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--no_mixed_precision', dest='mixed_precision', action='store_false',
                       help='Do not use mixed precision training')
    parser.add_argument('--sharding_strategy', type=str, default="FULL_SHARD",
                       choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
                       help='Sharding strategy for FSDP')
    parser.set_defaults(mixed_precision=True)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Print configuration
    print(f"FSDP Training with the following configuration:")
    print(f"  H5 File: {args.h5_file}")
    print(f"  Split Stats: {args.split_stats}")
    print(f"  GPU Counts: {args.gpu_counts}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Patience: {args.patience}")
    print(f"  Mixed Precision: {args.mixed_precision}")
    print(f"  Sharding Strategy: {args.sharding_strategy}")
    
    # Run FSDP training benchmark
    fsdp_results, fsdp_summary = run_fsdp_benchmark(
        args.h5_file,
        args.split_stats,
        args.gpu_counts,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        hidden_dims=[1024, 512, 256, 128],
        dropout_rate=0.5,
        patience=args.patience,
        use_mixed_precision=args.mixed_precision,
        sharding_strategy=args.sharding_strategy
    )