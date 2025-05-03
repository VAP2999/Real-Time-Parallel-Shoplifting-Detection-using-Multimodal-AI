
import wandb
from torch.utils.tensorboard import SummaryWriter
import psutil
import GPUtil
from datetime import datetime
import os
import time
import json
import math
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
import socket
import warnings
import traceback
warnings.filterwarnings('ignore')

# Define paths
DATA_ROOT = "/home/patel.vanshi/PRO_DATASET" 
DATA_ROOT1 = "/home/patel.vanshi/HPC_PROJECT" # Current directory as root
PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, "processed_data")
FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, "features")
METADATA_DIR = os.path.join(PROCESSED_DATA_DIR, "metadata")
MODELS_DIR = os.path.join(DATA_ROOT, "models")
RESULTS_DIR = os.path.join(DATA_ROOT1, "results")

# Create directories if they don't exist
for directory in [MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set NCCL environment variables for better performance - reduced debug output
os.environ['NCCL_DEBUG'] = 'WARN'  # Only show warning messages
os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand if not needed

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
    """PyTorch dataset for anomaly detection using HDF5 features with caching"""
    
    def __init__(self, h5_file, video_names=None, transform=None, target_transform=None):
        """Initialize dataset - but don't open the h5 file yet"""
        self.h5_file = h5_file
        self.transform = transform
        self.target_transform = target_transform
        self.videos = []
        self.feature_dim = None
        self.label_map = {}
        self.idx_to_label = {}
        self.metadata_cache = {}
        self.feature_cache = {}  # Cache for features
        self.cache_size = 200  # Max number of features to cache
        
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
            
            # Pre-cache metadata for each video to avoid repeated HDF5 opens
            for video_name in self.videos:
                video_group = h5['videos'][video_name]
                self.metadata_cache[video_name] = {
                    'label_idx': video_group.attrs['label_idx'],
                    'is_anomaly': video_group.attrs['is_anomaly'],
                    'original_name': video_group.attrs.get('original_video_name', video_name)
                }
                
                # Pre-cache the first few features for faster initial loading
                if len(self.feature_cache) < self.cache_size:
                    self.feature_cache[video_name] = video_group['video_features_mean_norm'][...]
    
    def __len__(self):
        """Get dataset length"""
        return len(self.videos)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset - using cache when possible"""
        video_name = self.videos[idx]
        
        # Check if features are in cache
        if video_name in self.feature_cache:
            features = self.feature_cache[video_name]
        else:
            # Get features from HDF5 file if not in cache
            with h5py.File(self.h5_file, 'r') as h5:
                features = h5['videos'][video_name]['video_features_mean_norm'][...]
                
                # Update cache if there's space - replace a random entry if full
                if len(self.feature_cache) < self.cache_size:
                    self.feature_cache[video_name] = features
                elif np.random.random() < 0.1:  # 10% chance to update cache
                    # Remove a random key
                    remove_key = list(self.feature_cache.keys())[np.random.randint(0, len(self.feature_cache))]
                    del self.feature_cache[remove_key]
                    self.feature_cache[video_name] = features
        
        # Get metadata from cache
        metadata = self.metadata_cache[video_name]
        label_idx = metadata['label_idx']
        is_anomaly = metadata['is_anomaly']
        original_name = metadata['original_name']
        
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
# Model Definition
#----------------------------

class AnomalyDetector(nn.Module):
    """Multi-task model for anomaly detection and action classification with BN"""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.5):
        super(AnomalyDetector, self).__init__()
        
        # Shared layers with Batch Normalization for better distributed training
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))  # Add BatchNorm for better gradient flow
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Task-specific layers
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
        
        self.action_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], num_classes)
        )
    
    def forward(self, x):
        # Apply shared layers
        x = self.shared_layers(x)
        
        # Apply task-specific layers
        anomaly_pred = self.anomaly_classifier(x).squeeze()
        action_pred = self.action_classifier(x)
        
        return anomaly_pred, action_pred

#----------------------------
# Distributed Training Setup
#----------------------------

def find_free_port():
    """Find a free port to use for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Binding to port 0 will cause the OS to find an available port
        sock.bind(('localhost', 0))
        return sock.getsockname()[1]

def setup_distributed(rank, world_size, port):
    """Initialize distributed training with minimal logging"""
    # Set up the master address and port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Suppress stdout and stderr during initialization to reduce noise
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    from io import StringIO
    
    # Initialize the process group with a timeout
    max_retries = 5
    retry_delay = 5  # seconds
    
    for retry in range(max_retries):
        try:
            # Capture and suppress stdout/stderr during init
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank,
                    timeout=timedelta(seconds=120)  # Longer timeout for stability
                )
            
            # Only print a short status message from rank 0
            if rank == 0:
                print(f"[GPU {rank}]: Process group initialized successfully")
            break
        except Exception as e:
            if retry == max_retries - 1:
                raise RuntimeError(f"Failed to initialize process group after {max_retries} attempts: {e}")
            if rank == 0:
                print(f"[GPU {rank}]: Retry {retry+1}/{max_retries}: Process group initialization failed, retrying...")
            time.sleep(retry_delay)
    
    # Set CUDA device - explicitly free memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_device(rank)
        
        # Print minimal device info only from rank 0
        if rank == 0:
            print(f"Training on {world_size} GPUs")

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

#----------------------------
# Monitoring and Logging Functions
#----------------------------

def log_system_stats():
    """Log system statistics for monitoring"""
    stats = {}
    
    # CPU stats
    stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
    stats['ram_percent'] = psutil.virtual_memory().percent
    
    # GPU stats if available
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            try:
                gpu = GPUtil.getGPUs()[i]
                stats[f'gpu{i}_util'] = gpu.load * 100
                stats[f'gpu{i}_memory_util'] = gpu.memoryUtil * 100
                stats[f'gpu{i}_temp'] = gpu.temperature
            except:
                # Skip if we can't get GPU info
                pass
    
    return stats

def init_wandb(rank, config):
    """Initialize Weights & Biases for the given rank"""
    if rank == 0:  # Initialize only on the main process
        # Create a unique run name with timestamp
        run_name = f"ddp-training-{config['num_gpus']}gpu-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Initialize W&B with your project name
        try:
            wandb.init(
                project="ddp-anomaly-detection",  # Change to your project name
                name=run_name,
                config=config,
                return_previous=True
            )
            
            print(f"W&B initialized on rank {rank} with run name: {run_name}")
            return True
        except Exception as e:
            print(f"W&B initialization failed: {e}. Training will continue without W&B logging.")
            return False
    return False

def log_metrics(rank, metrics, step=None, prefix=""):
    """Log metrics to W&B from the given rank"""
    if rank == 0:  # Log only from the main process
        try:
            # Add system stats
            if step is not None and step % 10 == 0:  # Log system stats less frequently
                system_stats = log_system_stats()
                for key, value in system_stats.items():
                    metrics[f"system/{key}"] = value
            
            # Add prefix to metric names if provided
            if prefix:
                metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
                
            # Log to W&B
            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except Exception as e:
            print(f"Warning: Failed to log metrics to W&B: {e}")

def finish_wandb(rank):
    """Finish W&B logging"""
    if rank == 0 and wandb.run is not None:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: Failed to finish W&B run: {e}")

#----------------------------
# Training Functions
#----------------------------

def train_epoch(model, loader, optimizer, criterion_anomaly, criterion_action, device, epoch, total_epochs):
    """Train for one epoch with optimized DDP operations and W&B logging"""
    model.train()
    
    # Use CUDA streams for overlapping operations
    default_stream = torch.cuda.current_stream()
    
    loss_values = []
    anomaly_loss_values = []
    action_loss_values = []
    
    correct_anomaly = 0
    total_anomaly = 0
    correct_action = 0
    total_action = 0
    
    # Get rank for progress display
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # Display progress only on rank 0
    if rank == 0:
        progress = tqdm(enumerate(loader), total=len(loader), 
                      desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    else:
        progress = enumerate(loader)
    
    optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
    
    # For logging batch-level metrics
    batch_metrics = {}
    global_step = epoch * len(loader)
    
    for i, (features, action_labels, anomaly_labels, _) in progress:
        # Move to device with non_blocking for overlap
        with torch.cuda.stream(default_stream):
            features = features.to(device, non_blocking=True)
            action_labels = action_labels.to(device, non_blocking=True)
            anomaly_labels = anomaly_labels.to(device, non_blocking=True)
            
            # Forward pass
            anomaly_predictions, action_predictions = model(features)
            
            # Calculate losses
            anomaly_loss = criterion_anomaly(anomaly_predictions, anomaly_labels)
            action_loss = criterion_action(action_predictions, action_labels)
            
            # Combined loss with weighting
            loss = 0.7 * anomaly_loss + 0.3 * action_loss
            
            # Backward pass
            loss.backward()
            
            # Use gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Record losses
        loss_values.append(loss.item())
        anomaly_loss_values.append(anomaly_loss.item())
        action_loss_values.append(action_loss.item())
        
        # Calculate accuracy
        # Anomaly detection accuracy
        anomaly_pred_binary = (anomaly_predictions >= 0.5).float()
        correct_anomaly += (anomaly_pred_binary == anomaly_labels).sum().item()
        total_anomaly += anomaly_labels.size(0)
        
        # Action classification accuracy
        _, action_pred = torch.max(action_predictions, 1)
        correct_action += (action_pred == action_labels).sum().item()
        total_action += action_labels.size(0)
        
        # Update progress bar on rank 0
        if rank == 0 and (i % 5 == 0 or i == len(loader) - 1):
            avg_loss = np.mean(loss_values[-10:]) if len(loss_values) >= 10 else np.mean(loss_values)
            anomaly_acc = correct_anomaly / total_anomaly if total_anomaly > 0 else 0
            action_acc = correct_action / total_action if total_action > 0 else 0
            
            progress.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'ano_acc': f"{anomaly_acc:.4f}",
                'act_acc': f"{action_acc:.4f}"
            })
            
            # Log batch-level metrics to W&B
            if i % 10 == 0:  # Log every 10 batches to reduce overhead
                current_step = global_step + i
                batch_metrics = {
                    'train/batch_loss': avg_loss,
                    'train/batch_anomaly_acc': anomaly_acc,
                    'train/batch_action_acc': action_acc,
                    'train/batch_anomaly_loss': np.mean(anomaly_loss_values[-10:]) if anomaly_loss_values else 0,
                    'train/batch_action_loss': np.mean(action_loss_values[-10:]) if action_loss_values else 0,
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                }
                
                # Log GPU utilization
                if torch.cuda.is_available():
                    for gpu_id in range(torch.cuda.device_count()):
                        if gpu_id < len(GPUtil.getGPUs()):
                            gpu = GPUtil.getGPUs()[gpu_id]
                            batch_metrics[f'system/gpu{gpu_id}_utilization'] = gpu.load * 100
                            batch_metrics[f'system/gpu{gpu_id}_memory'] = gpu.memoryUtil * 100
                
                # Log to W&B
                log_metrics(rank, batch_metrics, current_step)
    
    # Wait for all ops to finish
    torch.cuda.synchronize()
    
    # Calculate epoch statistics - synchronize across processes
    total_loss = torch.tensor(np.mean(loss_values), device=device)
    anomaly_acc = torch.tensor(correct_anomaly / total_anomaly if total_anomaly > 0 else 0, device=device)
    action_acc = torch.tensor(correct_action / total_action if total_action > 0 else 0, device=device)
    
    if world_size > 1:
        # Gather stats from all processes
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(anomaly_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(action_acc, op=dist.ReduceOp.SUM)
        
        total_loss = total_loss / world_size
        anomaly_acc = anomaly_acc / world_size
        action_acc = action_acc / world_size
    
    epoch_stats = {
        'loss': total_loss.item(),
        'anomaly_loss': np.mean(anomaly_loss_values),
        'action_loss': np.mean(action_loss_values),
        'anomaly_acc': anomaly_acc.item(),
        'action_acc': action_acc.item()
    }
    
    # Log epoch-level metrics to W&B
    if rank == 0:
        epoch_metrics = {
            'train/epoch_loss': epoch_stats['loss'],
            'train/epoch_anomaly_loss': epoch_stats['anomaly_loss'],
            'train/epoch_action_loss': epoch_stats['action_loss'],
            'train/epoch_anomaly_acc': epoch_stats['anomaly_acc'],
            'train/epoch_action_acc': epoch_stats['action_acc'],
            'train/epoch': epoch
        }
        log_metrics(rank, epoch_metrics, global_step + len(loader))
    
    return epoch_stats

def validate(model, loader, criterion_anomaly, criterion_action, device, epoch=None, global_step=None):
    """Validate model with optimized operations and W&B logging"""
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
    
    # Get rank for progress display
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    with torch.no_grad():
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
            
            # Forward pass
            anomaly_predictions, action_predictions = model(features)
            
            # Calculate losses
            anomaly_loss = criterion_anomaly(anomaly_predictions, anomaly_labels)
            action_loss = criterion_action(action_predictions, action_labels)
            
            # Combined loss with weighting
            loss = 0.7 * anomaly_loss + 0.3 * action_loss
            
            # Record losses
            loss_values.append(loss.item())
            anomaly_loss_values.append(anomaly_loss.item())
            action_loss_values.append(action_loss.item())
            
            # Calculate accuracy
            # Anomaly detection accuracy
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
    val_loss = np.mean(loss_values) if loss_values else 0
    val_anomaly_loss = np.mean(anomaly_loss_values) if anomaly_loss_values else 0
    val_action_loss = np.mean(action_loss_values) if action_loss_values else 0
    
    val_anomaly_acc = correct_anomaly / total_anomaly if total_anomaly > 0 else 0
    val_action_acc = correct_action / total_action if total_action > 0 else 0
    
    # Combine all predictions and labels
    all_anomaly_labels = np.concatenate(all_anomaly_labels) if all_anomaly_labels else np.array([])
    all_anomaly_predictions = np.concatenate(all_anomaly_predictions) if all_anomaly_predictions else np.array([])
    all_action_labels = np.concatenate(all_action_labels) if all_action_labels else np.array([])
    all_action_predictions = np.concatenate(all_action_predictions) if all_action_predictions else np.array([])
    
    # Calculate F1 score for anomaly detection
    try:
        anomaly_pred_binary = (all_anomaly_predictions >= 0.5).astype(float)
        anomaly_f1 = f1_score(all_anomaly_labels, anomaly_pred_binary)
        anomaly_precision = precision_score(all_anomaly_labels, anomaly_pred_binary)
        anomaly_recall = recall_score(all_anomaly_labels, anomaly_pred_binary)
    except Exception as e:
        print(f"Warning: Error calculating metrics: {e}")
        anomaly_f1 = 0.0
        anomaly_precision = 0.0
        anomaly_recall = 0.0
    
    val_stats = {
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
    
    # Log validation metrics to W&B on rank 0
    if rank == 0:
        val_metrics = {
            'val/loss': val_stats['loss'],
            'val/anomaly_loss': val_stats['anomaly_loss'],
            'val/action_loss': val_stats['action_loss'],
            'val/anomaly_acc': val_stats['anomaly_acc'],
            'val/action_acc': val_stats['action_acc'],
            'val/anomaly_f1': val_stats['anomaly_f1'],
            'val/anomaly_precision': val_stats['anomaly_precision'],
            'val/anomaly_recall': val_stats['anomaly_recall'],
            'val/epoch': epoch if epoch is not None else 0
        }
        
        # Create confusion matrix for anomaly detection
        if len(all_anomaly_labels) > 0 and wandb.run is not None:
            try:
                wandb.log({"val/anomaly_conf_mat": wandb.plot.confusion_matrix(
                    y_true=all_anomaly_labels.astype(int),
                    preds=anomaly_pred_binary.astype(int),
                    class_names=["Normal", "Anomaly"]
                )})
            except Exception as e:
                print(f"Warning: Could not log confusion matrix: {e}")
        
        # Log metrics - use global_step if provided
        if global_step is not None:
            log_metrics(rank, val_metrics, global_step)
        else:
            # Direct logging without system stats
            if wandb.run is not None:
                wandb.log(val_metrics)
    
    return val_stats

def train_model(
    rank, 
    world_size,
    h5_file,
    train_video_names,
    test_video_names,
    num_classes,
    model_save_path,
    results_save_path,
    epochs=16,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=1e-4,
    hidden_dims=[1024, 512, 256, 128],
    dropout_rate=0.5,
    patience=10,
    port=12355
):
    """Train the model using distributed data parallel with optimized parameters and W&B logging"""
    try:
        # Set up distributed training
        setup_distributed(rank, world_size, port)
        
        # Print minimal info about current process (only from rank 0)
        if rank == 0:
            print(f"Starting training on {world_size} GPU{'s' if world_size > 1 else ''}")
        
        # Set device
        device = torch.device(f'cuda:{rank}')
        
        # Create datasets
        train_dataset = AnomalyDetectionDataset(h5_file, train_video_names)
        test_dataset = AnomalyDetectionDataset(h5_file, test_video_names)
        
        # Get feature dimension
        feature_dim = train_dataset.feature_dim
        
        # Calculate per-GPU batch size - use integer division
        if world_size == 1:
            per_gpu_batch_size = batch_size
        elif world_size == 2:
            per_gpu_batch_size = batch_size * 2
        else:
            # For 3+ GPUs, divide batch size by increasing factor to avoid memory issues
            # This is critical for performance on more GPUs
            per_gpu_batch_size = int(batch_size / (1 + 0.3 * (world_size - 2)))
        
        # Ensure batch size is an integer
        per_gpu_batch_size = max(1, int(per_gpu_batch_size))
        
        if rank == 0:
            print(f"Batch size: {per_gpu_batch_size} per GPU (total: {per_gpu_batch_size * world_size})")
        
        # Initialize W&B on rank 0
        wandb_config = {
            'model': 'AnomalyDetector',
            'num_gpus': world_size,
            'batch_size': per_gpu_batch_size,
            'global_batch_size': per_gpu_batch_size * world_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate,
            'feature_dim': feature_dim,
            'num_classes': num_classes,
            'max_epochs': epochs,
            'patience': patience
        }
        
        init_wandb(rank, wandb_config)
        
        # Create optimized data loading pipeline
        # Create train sampler with optimized shuffling
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42 + rank  # Different seed per rank for better data distribution
        )
        
        # Create train/test datasets with efficient caching
        # We'll use different workers to parallelize data loading
        num_workers_per_gpu = 2  # Fewer workers to reduce contention
        
        # Create data loaders with pin memory and persistent workers
        train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch_size,
            sampler=train_sampler,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2  # Prefetch fewer batches to reduce memory pressure
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=per_gpu_batch_size,  # Same batch size for validation
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Optimize CUDA operations
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        
        # Make sure to set the same seed for model initialization
        torch.manual_seed(42 + rank)  # Different seed per rank for better parameter initialization
        torch.cuda.manual_seed_all(42 + rank)
        
        # Create model with optimized architecture
        model = AnomalyDetector(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        ).to(device)
        
        # Initialize weights properly
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        model.apply(init_weights)
        
        # SyncBatchNorm helps with consistent normalization in DDP
        if world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
        # Wrap model in DDP with optimized settings
        model = DDP(
            model, 
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,  # Reduces communication overhead
            find_unused_parameters=False  # Better performance
        )
        
        # Define loss functions
        criterion_anomaly = nn.BCELoss()
        criterion_action = nn.CrossEntropyLoss()
        
        # Define optimizer with learning rate scaled by world size
        # Square root scaling works better for maintaining effective learning rate
        # Reduce LR as GPU count increases to maintain stability
        scaling_factor = math.sqrt(world_size) if world_size <= 2 else math.sqrt(2)
        scaled_lr = learning_rate * scaling_factor
        
        # Use AdamW with weight decay for better generalization
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=scaled_lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warm-up
        total_steps = epochs * len(train_loader)
        warmup_steps = min(200, total_steps // 20)  # 5% warmup period
        
        def lr_schedule(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps))))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
        
        # Training loop
        best_val_loss = float('inf')
        best_val_f1 = 0.0
        early_stop_counter = 0
        train_stats = []
        val_stats = []
        
        start_time = time.time()
        
        # Log model graph to W&B if on rank 0
        if rank == 0 and wandb.run is not None:
            try:
                # Log model architecture as text
                model_summary = str(model)
                wandb.log({"model/architecture": wandb.Html(f"<pre>{model_summary}</pre>")})
            except Exception as e:
                print(f"Warning: Could not log model architecture: {e}")
        
        for epoch in range(epochs):
            try:
                # Set epoch for distributed sampler
                train_sampler.set_epoch(epoch)
                
                # Record learning rate
                current_lr = optimizer.param_groups[0]['lr']
                if rank == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Learning rate: {current_lr:.6f}")
                
                # Train for one epoch
                train_epoch_stats = train_epoch(
                    model, 
                    train_loader, 
                    optimizer, 
                    criterion_anomaly, 
                    criterion_action, 
                    device,
                    epoch,
                    epochs
                )
                
                # Step the learning rate scheduler after each epoch
                scheduler.step()
                
                # Calculate global step for validation
                global_step = (epoch + 1) * len(train_loader)
                
                # Validate
                val_epoch_stats = validate(
                    model, 
                    test_loader, 
                    criterion_anomaly, 
                    criterion_action, 
                    device,
                    epoch=epoch,
                    global_step=global_step
                )
                
                # Add learning rate to stats
                train_epoch_stats['learning_rate'] = current_lr
                
                # Record training and validation stats
                train_stats.append(train_epoch_stats)
                val_stats.append({k: v for k, v in val_epoch_stats.items() 
                                  if not isinstance(v, np.ndarray)})
                
                # Print stats on rank 0
                if rank == 0:
                    print(f"Epoch {epoch+1}/{epochs}")
                    print(f"  Train loss: {train_epoch_stats['loss']:.4f}, Anomaly acc: {train_epoch_stats['anomaly_acc']:.4f}, Action acc: {train_epoch_stats['action_acc']:.4f}")
                    print(f"  Val loss: {val_epoch_stats['loss']:.4f}, Anomaly acc: {val_epoch_stats['anomaly_acc']:.4f}, Action acc: {val_epoch_stats['action_acc']:.4f}")
                    print(f"  Anomaly F1: {val_epoch_stats['anomaly_f1']:.4f}, Precision: {val_epoch_stats['anomaly_precision']:.4f}, Recall: {val_epoch_stats['anomaly_recall']:.4f}")
                
                # Check if this is the best model - consider both loss and F1 score
                is_best_loss = val_epoch_stats['loss'] < best_val_loss
                is_best_f1 = val_epoch_stats['anomaly_f1'] > best_val_f1
                
                if is_best_loss or is_best_f1:
                    if is_best_loss:
                        best_val_loss = val_epoch_stats['loss']
                    if is_best_f1:
                        best_val_f1 = val_epoch_stats['anomaly_f1']
                    
                    early_stop_counter = 0
                    
                    # Save best model on rank 0
                    if rank == 0:
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_epoch_stats['loss'],
                            'val_loss': val_epoch_stats['loss'],
                            'val_anomaly_acc': val_epoch_stats['anomaly_acc'],
                            'val_action_acc': val_epoch_stats['action_acc'],
                            'val_anomaly_f1': val_epoch_stats['anomaly_f1'],
                        }
                        
                        try:
                            torch.save(checkpoint, model_save_path)
                            
                            # Log model to W&B
                            if wandb.run is not None:
                                wandb.save(model_save_path)
                            
                            if is_best_f1:
                                print(f"Saved best model with F1 score: {best_val_f1:.4f}")
                                if wandb.run is not None:
                                    wandb.run.summary["best_val_f1"] = best_val_f1
                            else:
                                print(f"Saved best model with loss: {best_val_loss:.4f}")
                                if wandb.run is not None:
                                    wandb.run.summary["best_val_loss"] = best_val_loss
                        except Exception as e:
                            print(f"Warning: Could not save model: {e}")
                else:
                    early_stop_counter += 1
                    
                    if early_stop_counter >= patience:
                        if rank == 0:
                            print(f"Early stopping triggered after {epoch+1} epochs")
                            if wandb.run is not None:
                                wandb.run.summary["stopped_epoch"] = epoch + 1
                        break
                
                # Synchronize all processes after each epoch
                if world_size > 1:
                    dist.barrier()
                
            except Exception as e:
                print(f"[rank{rank}]: Exception during epoch {epoch+1}: {str(e)}")
                traceback.print_exc()
                if world_size > 1:
                    # Synchronize processes before continuing
                    try:
                        dist.barrier()
                    except:
                        pass
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Save training and validation stats on rank 0
        if rank == 0:
            # Convert train_stats and val_stats to DataFrames without NumPy arrays
            train_df = pd.DataFrame(train_stats)
            val_df = pd.DataFrame(val_stats)
            
            # Create dictionary with final results
            final_results = {
                'num_gpus': world_size,
                'num_epochs': epoch + 1,  # Store actual epoch count
                'batch_size': batch_size,
                'per_gpu_batch_size': per_gpu_batch_size,
                'effective_batch_size': per_gpu_batch_size * world_size,
                'learning_rate': scaled_lr,
                'base_learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout_rate,
                'feature_dim': feature_dim,
                'num_classes': num_classes,
                'training_time': training_time,
                'training_time_per_epoch': training_time / (epoch + 1),
                'best_val_loss': best_val_loss,
                'best_val_f1': best_val_f1,
                'final_train_loss': train_stats[-1]['loss'] if train_stats else 0,
                'final_val_loss': val_stats[-1]['loss'] if val_stats else 0,
                'final_train_anomaly_acc': train_stats[-1]['anomaly_acc'] if train_stats else 0,
                'final_val_anomaly_acc': val_stats[-1]['anomaly_acc'] if val_stats else 0,
                'final_train_action_acc': train_stats[-1]['action_acc'] if train_stats else 0,
                'final_val_action_acc': val_stats[-1]['action_acc'] if val_stats else 0,
                'final_val_anomaly_f1': val_stats[-1]['anomaly_f1'] if val_stats else 0,
                'final_val_anomaly_precision': val_stats[-1]['anomaly_precision'] if val_stats else 0,
                'final_val_anomaly_recall': val_stats[-1]['anomaly_recall'] if val_stats else 0,
            }
            
            # Log final summary to W&B
            if wandb.run is not None:
                for key, value in final_results.items():
                    wandb.run.summary[key] = value
            
            # Save final results and training/validation DataFrames
            try:
                with open(results_save_path, 'w') as f:
                    json.dump(final_results, f, indent=2, cls=NumpyEncoder)
                
                train_df.to_csv(os.path.join(os.path.dirname(results_save_path), f'train_stats_{world_size}gpu.csv'), index=False)
                val_df.to_csv(os.path.join(os.path.dirname(results_save_path), f'val_stats_{world_size}gpu.csv'), index=False)
                
                # Log CSV files to W&B
                if wandb.run is not None:
                    wandb.save(os.path.join(os.path.dirname(results_save_path), f'train_stats_{world_size}gpu.csv'))
                    wandb.save(os.path.join(os.path.dirname(results_save_path), f'val_stats_{world_size}gpu.csv'))
            except Exception as e:
                print(f"Warning: Could not save results: {e}")
            
            print(f"Training complete in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
            print(f"Best validation loss: {best_val_loss:.4f}, Best F1: {best_val_f1:.4f}")
            print(f"Results saved to {results_save_path}")
            
            # Finish W&B logging
            finish_wandb(rank)
    except Exception as e:
        print(f"[rank{rank}]: Exception occurred: {str(e)}")
        traceback.print_exc()
        
        # Finish W&B even if there's an exception
        finish_wandb(rank)
    finally:
        # Clean up distributed training
        cleanup_distributed()

def create_benchmark_visualizations(benchmark_df, benchmark_summary, benchmark_id=None):
    """Create visualizations for benchmark results with improved styling and upload to W&B"""
    # Initialize W&B for visualization summary if not already initialized
    try:
        if wandb.run is None:
            wandb.init(
                project="ddp-anomaly-detection",  # Change 
                name=f"Benchmark-Viz-{benchmark_id if benchmark_id else datetime.now().strftime('%Y%m%d-%H%M%S')}",
                notes="GPU count benchmark visualizations",
                tags=["benchmark-viz"]
            )
    except Exception as e:
        print(f"Warning: Could not initialize W&B for visualizations: {str(e)}")
    
    # Set a consistent style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Common colors for consistency
    colors = {
        'time': '#1f77b4',       # Blue
        'speedup': '#ff7f0e',    # Orange
        'ideal': '#2ca02c',      # Green
        'anomaly': '#d62728',    # Red
        'action': '#9467bd',     # Purple
        'f1': '#8c564b',         # Brown
        'efficiency': '#e377c2'  # Pink
    }
    
    # 1. Training time vs GPU count
    plt.figure(figsize=(12, 7))
    plt.plot(benchmark_summary['gpu_counts'], benchmark_summary['training_times'], 'o-', 
             color=colors['time'], linewidth=3, markersize=10)
    
    # Add data labels
    for i, (x, y) in enumerate(zip(benchmark_summary['gpu_counts'], benchmark_summary['training_times'])):
        plt.text(x, y*1.03, f"{y:.1f}s", ha='center', fontsize=12)
    
    plt.xlabel('Number of GPUs', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=14)
    plt.title('Training Time vs Number of GPUs', fontsize=16, fontweight='bold')
    plt.xticks(benchmark_summary['gpu_counts'], fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save locally and log to W&B
    fig_path = os.path.join(RESULTS_DIR, 'training_time_vs_gpus.png')
    plt.savefig(fig_path, dpi=300)
    try:
        if wandb.run is not None:
            wandb.log({"plots/training_time_vs_gpus": wandb.Image(fig_path)})
    except:
        pass
    plt.close()
    
    # 2. Speedup vs GPU count with efficiency
    plt.figure(figsize=(12, 7))
    
    # Calculate efficiency (speedup / GPU count)
    efficiency = [s / g for s, g in zip(benchmark_summary['speedups'], benchmark_summary['gpu_counts'])]
    
    # Plot speedup
    plt.plot(benchmark_summary['gpu_counts'], benchmark_summary['speedups'], 'o-', 
             color=colors['speedup'], linewidth=3, markersize=10, label='Actual Speedup')
    
    # Plot ideal speedup
    plt.plot(benchmark_summary['gpu_counts'], benchmark_summary['gpu_counts'], '--', 
             color=colors['ideal'], linewidth=2, label='Ideal Linear Speedup')
    
    # Add data labels for speedup
    for i, (x, y) in enumerate(zip(benchmark_summary['gpu_counts'], benchmark_summary['speedups'])):
        plt.text(x, y*1.05, f"{y:.2f}x", ha='center', fontsize=12, fontweight='bold')
    
    # Add efficiency percentage on the same plot
    for i, (x, y, eff) in enumerate(zip(benchmark_summary['gpu_counts'], benchmark_summary['speedups'], efficiency)):
        plt.text(x, y*0.85, f"{eff*100:.1f}% eff.", ha='center', fontsize=10, alpha=0.7)
    
    plt.xlabel('Number of GPUs', fontsize=14)
    plt.ylabel('Speedup', fontsize=14)
    plt.title('Training Speedup vs Number of GPUs', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.xticks(benchmark_summary['gpu_counts'], fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save locally and log to W&B
    fig_path = os.path.join(RESULTS_DIR, 'speedup_vs_gpus.png')
    plt.savefig(fig_path, dpi=300)
    try:
        if wandb.run is not None:
            wandb.log({"plots/speedup_vs_gpus": wandb.Image(fig_path)})
    except:
        pass
    plt.close()
    
    # 3. Model performance metrics vs GPU count
    plt.figure(figsize=(12, 7))
    
    # Plot multiple metrics
    plt.plot(benchmark_summary['gpu_counts'], benchmark_summary['final_val_anomaly_accs'], 'o-', 
             color=colors['anomaly'], linewidth=2, markersize=8, label='Anomaly Detection Accuracy')
    plt.plot(benchmark_summary['gpu_counts'], benchmark_summary['final_val_action_accs'], 's-', 
             color=colors['action'], linewidth=2, markersize=8, label='Action Classification Accuracy')
    plt.plot(benchmark_summary['gpu_counts'], benchmark_summary['final_val_anomaly_f1s'], '^-', 
             color=colors['f1'], linewidth=2, markersize=8, label='Anomaly Detection F1 Score')
    
    # Add F1 score data labels
    for i, (x, y) in enumerate(zip(benchmark_summary['gpu_counts'], benchmark_summary['final_val_anomaly_f1s'])):
        plt.text(x, y-0.03, f"F1: {y:.3f}", ha='center', fontsize=9)
    
    plt.xlabel('Number of GPUs', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.title('Model Performance vs Number of GPUs', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.xticks(benchmark_summary['gpu_counts'], fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.0)  # Adjust Y limits for better visualization
    plt.tight_layout()
    
    # Save locally and log to W&B
    fig_path = os.path.join(RESULTS_DIR, 'performance_vs_gpus.png')
    plt.savefig(fig_path, dpi=300)
    try:
        if wandb.run is not None:
            wandb.log({"plots/performance_vs_gpus": wandb.Image(fig_path)})
    except:
        pass
    plt.close()
    
    # 4. Training efficiency visualization
    plt.figure(figsize=(12, 7))
    
    # Create bar chart for efficiency
    bars = plt.bar(benchmark_summary['gpu_counts'], efficiency, color=colors['efficiency'], alpha=0.7)
    
    # Add percentage labels
    for bar, eff in zip(bars, efficiency):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{eff*100:.1f}%", ha='center', fontsize=12)
    
    plt.axhline(y=1.0, color=colors['ideal'], linestyle='--', label='Ideal Efficiency (100%)')
    
    plt.xlabel('Number of GPUs', fontsize=14)
    plt.ylabel('Efficiency (Speedup / GPU Count)', fontsize=14)
    plt.title('Training Efficiency vs Number of GPUs', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.xticks(benchmark_summary['gpu_counts'], fontsize=12)
    plt.ylim(0, 1.2)  # Set y limit to highlight efficiency percentage
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save locally and log to W&B
    fig_path = os.path.join(RESULTS_DIR, 'efficiency_vs_gpus.png')
    plt.savefig(fig_path, dpi=300)
    try:
        if wandb.run is not None:
            wandb.log({"plots/efficiency_vs_gpus": wandb.Image(fig_path)})
    except:
        pass
    plt.close()
    
    # 5. Combined time and throughput plot
    plt.figure(figsize=(14, 8))
    
    # Calculate samples per second (throughput)
    # Using effective batch size * batches per epoch / time per epoch
    throughput = []
    for i, gpu_count in enumerate(benchmark_summary['gpu_counts']):
        effective_batch = benchmark_summary['effective_batch_sizes'][i]
        time_per_epoch = benchmark_summary['training_times_per_epoch'][i]
        # Estimate number of batches per epoch based on dataset size
        # This is an approximation - adjust based on your dataset
        est_batches_per_epoch = 100  # Placeholder - replace with actual if known
        samples_per_sec = (effective_batch * est_batches_per_epoch) / time_per_epoch
        throughput.append(samples_per_sec)
    
    # Left y-axis for training time
    ax1 = plt.gca()
    line1 = ax1.plot(benchmark_summary['gpu_counts'], benchmark_summary['training_times'], 'o-', 
             color=colors['time'], linewidth=3, markersize=10, label='Training Time')
    ax1.set_xlabel('Number of GPUs', fontsize=14)
    ax1.set_ylabel('Training Time (seconds)', fontsize=14, color=colors['time'])
    ax1.tick_params(axis='y', labelcolor=colors['time'])
    
    # Add data labels for training time
    for i, (x, y) in enumerate(zip(benchmark_summary['gpu_counts'], benchmark_summary['training_times'])):
        ax1.text(x, y*1.05, f"{y:.1f}s", ha='center', fontsize=10, color=colors['time'])
    
    # Right y-axis for throughput
    ax2 = ax1.twinx()
    line2 = ax2.plot(benchmark_summary['gpu_counts'], throughput, 's--', 
             color=colors['speedup'], linewidth=3, markersize=10, label='Throughput')
    ax2.set_ylabel('Samples processed per second', fontsize=14, color=colors['speedup'])
    ax2.tick_params(axis='y', labelcolor=colors['speedup'])
    
    # Add data labels for throughput
    for i, (x, y) in enumerate(zip(benchmark_summary['gpu_counts'], throughput)):
        ax2.text(x, y*0.95, f"{y:.0f}", ha='center', fontsize=10, color=colors['speedup'])
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=12, loc='upper center')
    
    plt.title('Training Time and Throughput vs Number of GPUs', fontsize=16, fontweight='bold')
    plt.xticks(benchmark_summary['gpu_counts'], fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save locally and log to W&B
    fig_path = os.path.join(RESULTS_DIR, 'time_and_throughput.png')
    plt.savefig(fig_path, dpi=300)
    try:
        if wandb.run is not None:
            wandb.log({"plots/time_and_throughput": wandb.Image(fig_path)})
    except:
        pass
    plt.close()
    
    # 6. Batch size comparison
    plt.figure(figsize=(12, 7))
    bars = plt.bar(benchmark_summary['gpu_counts'], benchmark_summary['effective_batch_sizes'], color='#17becf', alpha=0.7)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 5, 
                f"{int(height)}", ha='center', fontsize=12)
    
    plt.xlabel('Number of GPUs', fontsize=14)
    plt.ylabel('Effective Batch Size', fontsize=14)
    plt.title('Effective Batch Size vs Number of GPUs', fontsize=16, fontweight='bold')
    plt.xticks(benchmark_summary['gpu_counts'], fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save locally and log to W&B
    fig_path = os.path.join(RESULTS_DIR, 'batch_size_vs_gpus.png')
    plt.savefig(fig_path, dpi=300)
    try:
        if wandb.run is not None:
            wandb.log({"plots/batch_size_vs_gpus": wandb.Image(fig_path)})
    except:
        pass
    plt.close()
    
    # Create a table for the summary metrics in W&B
    try:
        if wandb.run is not None:
            summary_table = wandb.Table(dataframe=benchmark_df)
            wandb.log({"benchmark_summary_table": summary_table})
            
            # Create visualization for the summary table
            fields_to_plot = {
                'Training Time (s)': 'Training Time (seconds)',
                'Speedup': 'Training Speedup',
                'Val Anomaly F1': 'Validation F1 Score'
            }
            
            for field, title in fields_to_plot.items():
                fig = plt.figure(figsize=(10, 6))
                plt.bar(benchmark_df['GPU Count'].astype(str), benchmark_df[field], alpha=0.7)
                plt.title(title)
                plt.xlabel('Number of GPUs')
                plt.ylabel(field)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                # Log bar chart to W&B
                wandb.log({f"summary/{title.replace(' ', '_').lower()}": wandb.Image(fig)})
                plt.close(fig)
            
            # Finish the run
            wandb.finish()
    except Exception as e:
        print(f"Warning: Could not log summary table to W&B: {str(e)}")

def run_training_benchmark(
    h5_file,
    split_stats_file,
    gpu_counts=[1, 2, 4],
    epochs=16,
    batch_size=128,
    learning_rate=0.001,
    weight_decay=1e-4,
    hidden_dims=[1024, 512, 256, 128],
    dropout_rate=0.5,
    patience=10
):
    """Run training benchmark with different numbers of GPUs with better error handling"""
    # Load split stats
    try:
        with open(split_stats_file, 'r') as f:
            split_stats = json.load(f)
        
        train_video_names = split_stats['train_video_names']
        test_video_names = split_stats['test_video_names']
    except Exception as e:
        print(f"Error loading split stats: {e}")
        print(f"Checking if the file exists: {os.path.exists(split_stats_file)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Available files in metadata dir: {os.listdir(METADATA_DIR) if os.path.exists(METADATA_DIR) else 'METADATA_DIR not found'}")
        return [], {}
    
    # Get number of classes without keeping dataset open
    try:
        temp_dataset = AnomalyDetectionDataset(h5_file, train_video_names)
        num_classes = len(temp_dataset.label_map)
        dataset_size = len(train_video_names)
        del temp_dataset  # Explicitly delete to free resources
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Checking if the file exists: {os.path.exists(h5_file)}")
        return [], {}
    
    print(f"Training benchmark with GPU counts: {gpu_counts}")
    print(f"Dataset: {h5_file}")
    print(f"Training videos: {len(train_video_names)}")
    print(f"Testing videos: {len(test_video_names)}")
    print(f"Number of classes: {num_classes}")
    
    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if available_gpus == 0:
        print("No CUDA devices available. Cannot run multi-GPU training.")
        return [], {}
    
    # Filter GPU counts based on available devices
    gpu_counts = [count for count in gpu_counts if count <= available_gpus]
    
    if not gpu_counts:
        print(f"No valid GPU configurations (requested GPUs exceed available {available_gpus} GPUs)")
        return [], {}
    
    # Initialize results storage
    benchmark_results = []
    
    # Create a W&B benchmark run to group all individual runs
    benchmark_id = f"gpu-benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    try:
        # Initialize main benchmark W&B run
        wandb.init(
            project="ddp-anomaly-detection",  # Change to your project name
            name=f"Benchmark-Summary-{benchmark_id}",
            notes="GPU count benchmark summary run",
            tags=["benchmark-summary"]
        )
        
        wandb.config.update({
            "benchmark_id": benchmark_id,
            "gpu_counts": gpu_counts,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
            "patience": patience,
            "available_gpus": available_gpus,
            "dataset_size": dataset_size,
        })
        
        # Complete this initial run since we'll use it only for grouping
        wandb.finish()
    except Exception as e:
        print(f"Error initializing W&B benchmark group: {str(e)}")
        # Continue without W&B logging
        benchmark_id = None
    
    for i, gpu_count in enumerate(gpu_counts):
        # Clear CUDA cache before each run
        torch.cuda.empty_cache()
        
        # Sleep between runs to allow system to stabilize
        if i > 0:
            print(f"Waiting for system to stabilize before next run...")
            time.sleep(10)
        
        print(f"\n{'='*80}")
        print(f"Training with {gpu_count} GPUs")
        print(f"{'='*80}")
        
        # Create paths for model and results
        model_path = os.path.join(MODELS_DIR, f"anomaly_detector_{gpu_count}gpu.pt")
        results_path = os.path.join(RESULTS_DIR, f"training_results_{gpu_count}gpu.json")
        
        # Find a free port to use for this run - critical for avoiding address in use errors
        port = find_free_port()
        print(f"Using port {port} for process group communication")
        
        # Train model with distributed data parallel
        import torch.multiprocessing as mp
        
        # Use spawn start method for CUDA support
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, ignore
            pass
        
        # Adjust effective batch size based on GPU count
        effective_batch_size = batch_size
        if gpu_count > 1:
            # For multi-GPU, use a reasonable batch size but don't scale too much
            effective_batch_size = min(batch_size * 2, batch_size * gpu_count)
        
        try:
            mp.spawn(
                train_model,
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
                    port
                ),
                nprocs=gpu_count,
                join=True
            )
            
            # Add delay between runs to ensure resources are released
            time.sleep(5)
            
            # Clear CUDA cache after the run
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error during training with {gpu_count} GPUs: {str(e)}")
            import traceback
            traceback.print_exc()
        
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
        'epochs_completed': [],
        'effective_batch_sizes': [],
        'best_val_losses': [],
        'best_val_f1s': [],
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
            benchmark_summary['epochs_completed'].append(results.get('num_epochs', epochs))
            benchmark_summary['effective_batch_sizes'].append(results['effective_batch_size'])
            benchmark_summary['best_val_losses'].append(results['best_val_loss'])
            benchmark_summary['best_val_f1s'].append(results.get('best_val_f1', 0))
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
            'Epochs Completed': benchmark_summary['epochs_completed'],
            'Effective Batch Size': benchmark_summary['effective_batch_sizes'],
            'Best Val Loss': benchmark_summary['best_val_losses'],
            'Best Val F1': benchmark_summary['best_val_f1s'],
            'Val Anomaly Acc': benchmark_summary['final_val_anomaly_accs'],
            'Val Action Acc': benchmark_summary['final_val_action_accs'],
            'Val Anomaly F1': benchmark_summary['final_val_anomaly_f1s'],
            'Speedup': benchmark_summary['speedups']
        })
        
        # Save benchmark summary
        benchmark_df.to_csv(os.path.join(RESULTS_DIR, 'benchmark_summary.csv'), index=False)
        
        # Save full benchmark results with custom encoder
        with open(os.path.join(RESULTS_DIR, 'benchmark_results.json'), 'w') as f:
            json.dump(benchmark_results, f, indent=2, cls=NumpyEncoder)
        
        # Generate visualizations and upload to W&B
        create_benchmark_visualizations(benchmark_df, benchmark_summary, benchmark_id)
        
        print("\nBenchmark Summary:")
        print(benchmark_df)
        print(f"\nBenchmark results saved to {RESULTS_DIR}")
    else:
        print("No benchmark results were collected.")
    
    return benchmark_results, benchmark_summary

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Optimized Multi-GPU Training for Anomaly Detection')
    parser.add_argument('--h5_file', type=str, default=os.path.join(PROCESSED_DATA_DIR, "ucf_crime_features.h5"),
                       help='Path to HDF5 file with features')
    parser.add_argument('--split_stats', type=str, default=os.path.join(METADATA_DIR, "train_test_split.json"),
                       help='Path to train/test split JSON')
    parser.add_argument('--gpu_counts', type=int, nargs='+', default=[1, 2, 3, 4],
                       help='GPU counts to benchmark')
    parser.add_argument('--epochs', type=int, default=16,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Global batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Base learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with additional logging')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Enable error handling and debugging
    try:
        args = parse_args()
        
        # Print minimal system information
        print(f"PyTorch {torch.__version__} | CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        print(f"Available GPUs: {torch.cuda.device_count()} × {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
        
        # Print configuration
        print(f"\nTraining with the following configuration:")
        print(f"  H5 File: {args.h5_file}")
        print(f"  Split Stats: {args.split_stats}")
        print(f"  GPU Counts to test: {args.gpu_counts}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Global Batch Size: {args.batch_size}")
        print(f"  Base Learning Rate: {args.learning_rate}")
        print(f"  Weight Decay: {args.weight_decay}")
        print(f"  Patience: {args.patience}")
        print(f"  Debug Mode: {'Enabled' if args.debug else 'Disabled'}")
        
        # Create file paths for output files
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Run training benchmark
        run_training_benchmark(
            args.h5_file,
            args.split_stats,
            args.gpu_counts,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            hidden_dims=[1024, 512, 256, 128],
            dropout_rate=0.5,
            patience=args.patience
        )
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()





