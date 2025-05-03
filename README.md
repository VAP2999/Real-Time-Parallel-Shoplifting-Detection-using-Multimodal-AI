# Real-Time Parallel Shoplifting Detection using Multimodal AI

Advanced parallel computing framework for real-time shoplifting detection combining ResNet-50 visual features with pose estimation data. Implements and benchmarks Joblib, Dask, DP, DDP and FSDP parallelization strategies across multi-GPU clusters, achieving 81% detection accuracy with optimized throughput.

## Project Overview

This project addresses the significant challenge of shoplifting detection in retail environments using high-performance parallel computing techniques and multimodal AI. By combining visual features with pose estimation data and implementing various parallelization strategies, we've developed a system that can efficiently process surveillance video data in real-time while maintaining high detection accuracy.

## Key Features

- **Multimodal Feature Extraction**: Combines ResNet-50 visual features with PoseLift pose estimation data
- **Multiple Parallelization Strategies**:
  - Joblib and Dask for CPU-based data loading and preprocessing
  - Data Parallel (DP) for basic multi-GPU training
  - Distributed Data Parallel (DDP) for process-based efficient parallelism
  - Fully Sharded Data Parallel (FSDP) for memory-optimized parameter sharding
- **Optimized Data Pipeline**: Compresses 32GB of raw video data to 4.2GB HDF5 format for efficient storage and retrieval
- **Performance Monitoring**: Real-time visualization of GPU utilization and temperature using Weights & Biases
- **Multi-task Learning Model**: Combined anomaly detection and action classification

## Performance Highlights

- **High Accuracy**: 81% anomaly detection accuracy with 0.78 F1 score
- **Efficient Scaling**: 39% reduction in training time when scaling from 1 to 3 GPUs
- **Memory Optimization**: FSDP with SHARD_GRAD_OP strategy offers the best balance of memory efficiency and computational performance
- **Mixed Precision Acceleration**: Additional 1.5x speedup with minimal accuracy impact

## Dataset

We utilized the UCF Crime dataset, a comprehensive collection of real-world surveillance videos:
- Total videos: 1,900 with 128 hours of footage
- Original size: 104GB, optimized subset: 32GB 
- Includes multiple crime categories with focus on shoplifting events
- Processed into 4.2GB HDF5 format for efficient training

## Technical Implementation

### Parallel Data Loading & Preprocessing

- **Joblib**: Parallel video file processing with configurable worker counts
- **Dask**: LocalCluster configuration with customizable workers per CPU
- **Multi-threaded Frame Extraction**: Efficient temporal sampling at configurable intervals

### Feature Extraction Pipeline

- ResNet-50 pre-trained model with final classification layer removed
- Parallel distribution of feature extraction tasks across available GPUs
- Normalization and alignment of features across modalities

### Model Architecture

- Multi-task learning with shared feature extraction layers
- Task-specific branches for anomaly detection and action classification
- Batch normalization and dropout for improved training stability

### Distributed Training Optimization

- DDP implementation with NCCL backend for GPU-to-GPU communication
- FSDP with different sharding strategies for memory efficiency
- Mixed precision training using Automatic Mixed Precision (AMP)

## Hardware Requirements

- NVIDIA GPUs (tested on NVIDIA Tesla V100 and NVIDIA RTX A5000)
- CUDA 12.3+
- PyTorch 1.13.1+

## Software Dependencies

- Python 3.8.10+
- PyTorch 1.13.1+
- CUDA 12.3+
- Joblib, Dask, NumPy, OpenCV, H5py
- Weights & Biases (for monitoring)

## Key Research Findings

- DDP and FSDP consistently outperform DP across all GPU configurations
- 2-GPU configurations show the highest efficiency (≈62%) 
- 3-GPU setups provide fastest absolute training times with acceptable efficiency (≈50%)
- Mixed precision training delivers substantial speedups with minimal accuracy impact

## Future Work

- Testing on larger GPU clusters (8+ GPUs)
- Edge deployment optimization through model compression
- Exploration of ZeRO-3 for even more efficient distributed training
- Integration of additional modalities for improved detection accuracy

## References

- [UCF Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/)
- [Distributed Training in PyTorch](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)

## Contributors

- Vanshi Patel
- Malav Gajera
