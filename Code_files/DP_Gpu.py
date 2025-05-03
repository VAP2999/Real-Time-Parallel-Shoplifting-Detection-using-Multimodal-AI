import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import random

# Configure paths to dataset
DATA_DIR = "/home/patel.vanshi/PRO_DATASET"
H5_PATH = f"{DATA_DIR}/processed_data/ucf_crime_features.h5"

# ================================================================================
# DATASET FUNCTIONS
# ================================================================================

def get_feature_datasets(h5_path):
    """Get paths to all feature datasets in the H5 file"""
    with h5py.File(h5_path, 'r') as f:
        if 'videos' in f:
            features_paths = []
            videos_group = f['videos']
            
            for video_name in videos_group:
                # Try different potential paths
                potential_paths = [
                    f"videos/{video_name}",  # Direct dataset
                    f"videos/{video_name}/features",  # Common structure
                    f"videos/{video_name}/data",  # Alternative structure
                ]
                
                found = False
                for path in potential_paths:
                    try:
                        if path in f and isinstance(f[path], h5py.Dataset):
                            features_paths.append(path)
                            found = True
                            break
                    except Exception:
                        continue
                
                # If no standard path works
                if not found:
                    try:
                        video_group = videos_group[video_name]
                        if isinstance(video_group, h5py.Group):
                            # Find first dataset in group
                            for subname in video_group:
                                if isinstance(video_group[subname], h5py.Dataset):
                                    features_paths.append(f"videos/{video_name}/{subname}")
                                    break
                    except Exception:
                        continue
        
            print(f"Found {len(features_paths)} feature datasets")
            return features_paths
    
    # Fallback
    print("No feature datasets found")
    return []


def get_feature_dimension(h5_path):
    """Get feature dimension by examining actual data"""
    with h5py.File(h5_path, 'r') as f:
        # Check for metadata
        if 'metadata' in f and 'feature_dim' in f['metadata'].attrs:
            dim = f['metadata'].attrs['feature_dim']
            print(f"Found feature dimension in metadata: {dim}")
            return dim
        
        # Try to find a dataset and get its dimension
        if 'videos' in f:
            videos_group = f['videos']
            for video_name in list(videos_group.keys())[:100]:
                try:
                    if isinstance(videos_group[video_name], h5py.Dataset):
                        data = videos_group[video_name][:]
                        if data.ndim == 2:  # (timesteps, features)
                            return data.shape[1]
                        else:  # (features,)
                            return data.shape[0]
                except Exception:
                    pass
                
                # Try features subdataset
                try:
                    if 'features' in videos_group[video_name]:
                        data = videos_group[video_name]['features'][:]
                        if data.ndim == 2:
                            return data.shape[1]
                        else:
                            return data.shape[0]
                except Exception:
                    pass
    
    # Default dimension if we couldn't determine
    print("Could not determine feature dimension, using default of 2048")
    return 2048


def safe_get_features(h5_file, path, feature_dim):
    """Safely get features from a dataset path"""
    try:
        # Check if path exists and is a dataset
        if path in h5_file and isinstance(h5_file[path], h5py.Dataset):
            data = h5_file[path][:]
            
            # Convert to appropriate format
            if data.ndim == 2:  # Multiple vectors
                return torch.from_numpy(data.mean(axis=0)).float()  # Average across time
            else:  # Single vector
                return torch.from_numpy(data).float()
        
        # If path is a group
        elif path in h5_file and isinstance(h5_file[path], h5py.Group):
            # Try standard feature paths
            for subpath in ['features', 'data', 'embedding']:
                full_path = f"{path}/{subpath}"
                if full_path in h5_file and isinstance(h5_file[full_path], h5py.Dataset):
                    data = h5_file[full_path][:]
                    if data.ndim == 2:
                        return torch.from_numpy(data.mean(axis=0)).float()
                    else:
                        return torch.from_numpy(data).float()
            
            # If no standard path
            group = h5_file[path]
            for key in group:
                if isinstance(group[key], h5py.Dataset):
                    data = group[key][:]
                    if data.ndim == 2:
                        return torch.from_numpy(data.mean(axis=0)).float()
                    else:
                        return torch.from_numpy(data).float()
    
    except Exception as e:
        # Silently fail and return zeros - too many errors will spam output
        pass
    
    # Return zeros if all else fails
    return torch.zeros(feature_dim)


# ================================================================================
# MODEL DESIGN 
# ================================================================================

class HeavyComputationModel(nn.Module):
    """A computationally intensive model that will benefit from parallelism"""
    
    def __init__(self, feature_dim=2048, hidden_dim=2048):
        super().__init__()
        
        # Create a very deep network with large hidden dimensions
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # More complex classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, 2)
        )
    
    def forward(self, x):
        # Simpler but still intensive method that avoids dimension issues
        batch_size = x.size(0)
        
        # Add computationally intensive operations
        for _ in range(10):
            # Element-wise operations (sin/cos are computationally intensive on GPU)
            x = x + 0.01 * torch.sin(x)
            x = x + 0.01 * torch.cos(x)
            x = x + 0.01 * torch.tanh(x)
            
            # Additional non-linear transformations
            x = torch.relu(x)
            x = x + torch.relu(-x)
            
            # Add some channel-wise multiplications
            chunks = x.chunk(4, dim=1)
            if len(chunks) >= 4:
                a, b, c, d = chunks[:4]
                # Element-wise multiplications between chunks
                x1 = a * b
                x2 = b * c
                x3 = c * d
                x4 = d * a
                # Concatenate results back
                x = torch.cat([x1, x2, x3, x4], dim=1)
        
        features = self.encoder(x)
        return self.classifier(features)


# ================================================================================
# DATA PREPARATION
# ================================================================================

class SyntheticDataset(torch.utils.data.Dataset):
    """Create a synthetic dataset with a large number of samples"""
    def __init__(self, num_samples, feature_dim):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        
        # Generate random features and labels once
        self.features = torch.randn(num_samples, feature_dim)
        self.labels = torch.randint(0, 2, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ================================================================================
# GPU TRAINING FUNCTION
# ================================================================================

def train_epoch_proper_parallel(model, data_loader, optimizer, criterion, device):
    """Train for one epoch with proper parallelization practices"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Synchronize before starting timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for features, labels in data_loader:
        # Move data to device
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # Synchronize before ending timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    epoch_time = end_time - start_time
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / max(1, total)
    
    return epoch_time, avg_loss, accuracy


def benchmark_gpu_properly(num_gpus, feature_dim=2048, num_samples=10000, batch_size=256, num_epochs=3):
    """Benchmark training with proper GPU parallelization for the given number of GPUs"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run GPU benchmark.")
        return float('inf')
    
    available_gpus = torch.cuda.device_count()
    if num_gpus > available_gpus:
        print(f"Requested {num_gpus} GPUs but only {available_gpus} are available.")
        num_gpus = available_gpus
    
    device = torch.device('cuda')
    
    # Create the model
    model = HeavyComputationModel(feature_dim=feature_dim)
    model = model.to(device)
    
    # Use DataParallel for multiple GPUs
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"Using DataParallel with {num_gpus} GPUs: {device_ids}")
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create synthetic dataset that's large enough to be meaningful
    dataset = SyntheticDataset(num_samples, feature_dim)
    
    # Scale batch size with GPU count - this is important for efficient parallelization
    effective_batch_size = batch_size * num_gpus
    
    # Create data loader with enough workers for parallel data loading
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=min(4 * num_gpus, 16),  # Scale workers with GPUs, but cap at reasonable number
        pin_memory=True  # Important for faster data transfer to GPU
    )
    
    print(f"Created DataLoader with batch size {effective_batch_size} and {len(data_loader)} batches")
    
    # Warm-up to avoid timing initialization overhead
    print("Performing warm-up pass...")
    dummy_features = torch.rand(effective_batch_size, feature_dim).to(device)
    dummy_labels = torch.randint(0, 2, (effective_batch_size,)).to(device)
    _ = model(dummy_features)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Training loop
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_time, avg_loss, accuracy = train_epoch_proper_parallel(
            model, data_loader, optimizer, criterion, device
        )
        
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Time: {epoch_time:.2f}s, "
              f"Loss: {avg_loss:.4f}, "
              f"Acc: {accuracy:.2f}%")
    
    # Return average epoch time (excluding first epoch)
    if len(epoch_times) > 1:
        return sum(epoch_times[1:]) / len(epoch_times[1:])
    else:
        return epoch_times[0]


def run_improved_gpu_benchmark():
    """Run a GPU benchmark that should show proper scaling"""
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run GPU benchmark.")
        return
    
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    # Define GPU configurations to test
    gpu_configs = [1]
    for i in [2, 3, 4]:
        if i <= available_gpus:
            gpu_configs.append(i)
    
    # Print configurations to be tested
    print(f"Testing GPU configurations: {gpu_configs}")
    
    # Results storage
    times = []
    speedups = []
    efficiencies = []
    
    # Parameters for a substantial workload
    feature_dim = 2048
    num_samples = 20000  # Large enough to be meaningful
    batch_size = 64      # Base batch size per GPU
    num_epochs = 10       # Run multiple epochs for stability
    num_trials = 2       # Run multiple trials for each config
    
    # Run benchmark for each GPU config
    for num_gpus in gpu_configs:
        print(f"\n{'='*50}")
        print(f"Benchmarking with {num_gpus} GPU{'s' if num_gpus > 1 else ''}...")
        print(f"{'='*50}")
        
        # Run multiple trials to get more stable results
        trial_times = []
        
        for trial in range(num_trials):
            print(f"Trial {trial+1}/{num_trials}")
            
            # Clear CUDA cache between trials
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Run benchmark for this GPU configuration
            trial_time = benchmark_gpu_properly(
                num_gpus,
                feature_dim=feature_dim,
                num_samples=num_samples,
                batch_size=batch_size,
                num_epochs=num_epochs
            )
            
            trial_times.append(trial_time)
            print(f"Trial {trial+1} time: {trial_time:.2f}s")
        
        # Use the minimum time from trials (best performance)
        min_time = min(trial_times)
        print(f"Best time across trials: {min_time:.2f}s")
        
        times.append(min_time)
        
        # Calculate speedup and efficiency
        if len(times) == 1:
            # First configuration (baseline)
            speedups.append(1.0)
            efficiencies.append(1.0)
        else:
            speedup = times[0] / min_time
            efficiency = speedup / num_gpus
            speedups.append(speedup)
            efficiencies.append(efficiency)
    
    # Print results
    print("\nBenchmark Results:")
    print("=" * 40)
    print(f"{'GPUs':<6} | {'Time (s)':<10} | {'Speedup':<8} | {'Efficiency':<10}")
    print("-" * 40)
    
    for i, num_gpus in enumerate(gpu_configs):
        print(f"{num_gpus:<6} | {times[i]:<10.2f} | {speedups[i]:<8.2f} | {efficiencies[i]:<10.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 6))
    
    # Speedup plot
    plt.subplot(1, 3, 1)
    plt.plot(gpu_configs, speedups, 'o-', linewidth=2, markersize=8)
    plt.plot(gpu_configs, gpu_configs, 'k--', label='Ideal', linewidth=1.5)
    plt.title('Speedup vs. Number of GPUs', fontsize=14)
    plt.xlabel('Number of GPUs', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Efficiency plot
    plt.subplot(1, 3, 2)
    plt.plot(gpu_configs, efficiencies, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=1.0, color='k', linestyle='--', label='Ideal', linewidth=1.5)
    plt.title('Efficiency vs. Number of GPUs', fontsize=14)
    plt.xlabel('Number of GPUs', fontsize=12)
    plt.ylabel('Efficiency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Execution time plot
    plt.subplot(1, 3, 3)
    plt.plot(gpu_configs, times, 'o-', linewidth=2, markersize=8)
    plt.title('Execution Time vs. Number of GPUs', fontsize=14)
    plt.xlabel('Number of GPUs', fontsize=12)
    plt.ylabel('Time (s)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("improved_gpu_benchmark_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'gpu_configs': gpu_configs,
        'times': times,
        'speedups': speedups,
        'efficiencies': efficiencies
    }


if __name__ == "__main__":
    results = run_improved_gpu_benchmark()