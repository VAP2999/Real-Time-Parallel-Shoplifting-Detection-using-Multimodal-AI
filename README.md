# Real-Time-Parallel-Shoplifting-Detection-using-Multimodal-AI

Overview
This project implements an efficient, real-time shoplifting detection system using advanced parallel computing techniques and multimodal AI. The system combines computer vision and pose estimation to detect suspicious activities in retail environments with high accuracy while optimizing computational resource utilization.
Key Features

Multimodal Analysis: Combines visual features with pose estimation data for improved detection accuracy
High-Performance Computing: Implements and compares various parallelization strategies (Joblib, Dask, DP, DDP, FSDP)
Real-Time Processing: Optimized for low-latency detection in surveillance video streams
Scalable Architecture: Efficiently utilizes multiple CPUs and GPUs for parallel processing
High Accuracy: Achieves 81% accuracy and 0.78 F1 score for shoplifting detection

System Architecture
The system is composed of several key components:

Parallel Data Loading: Optimized video data loading using Joblib and Dask
Feature Extraction: Deep learning-based extraction of visual and pose features
Anomaly Detection: Multi-task model for shoplifting detection and action classification
Distributed Training: Efficient model training across multiple GPUs

Show Image
Installation
Prerequisites

Python 3.8+
CUDA 11.6+ (for GPU acceleration)
GPU with at least 8GB memory (16GB recommended)

Setup
bash# Clone the repository
git clone https://github.com/vap2999/ShopliftingDetection.git
cd ShopliftingDetection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Dataset
This project uses the UCF Crime dataset, focusing on shoplifting videos:

151 shoplifting videos
1,000+ normal videos
Average video duration: 240 seconds
Average resolution: 320x240 pixels

To prepare your dataset:
bash# Download UCF Crime dataset (requires registration)
python scripts/download_dataset.py --output_dir data/raw

# Process dataset for training
python scripts/preprocess_data.py --input_dir data/raw --output_dir data/processed
Usage
Model Training
bash# Single GPU training
python train.py --config configs/single_gpu.yaml

# Multi-GPU training with DDP
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/ddp.yaml

# FSDP training
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/fsdp.yaml
Inference
bash# Run inference on a video file
python detect.py --model checkpoints/best_model.pth --video path/to/video.mp4 --output output/result.mp4

# Run on a live camera feed
python detect.py --model checkpoints/best_model.pth --camera 0 --display
Performance Results
Parallelization MethodGPUsTraining Time (s)SpeedupEfficiencyF1 ScoreDP12.601.001.000.755DP42.181.190.300.753DDP1388.701.001.000.755DDP3261.031.490.500.751FSDP1310.091.001.000.783FSDP3188.651.640.550.791
Project Structure
ShopliftingDetection/
├── configs/               # Configuration files
├── data/                  # Dataset storage
├── models/                # Model definitions
│   ├── backbones/         # Feature extraction networks
│   ├── detection/         # Anomaly detection modules
│   └── pose/              # Pose estimation integration
├── utils/                 # Utility functions
│   ├── data_loading/      # Parallel data loading implementations
│   ├── distributed/       # Distributed training utilities
│   └── visualization/     # Result visualization tools
├── scripts/               # Helper scripts
├── train.py               # Training script
├── detect.py              # Inference script
└── requirements.txt       # Dependencies
Parallelization Strategies
The project implements and compares several parallelization strategies:

Joblib Parallelization: CPU-based parallel processing for data loading and preprocessing
Dask Distributed Computing: Scalable distributed computing framework for data processing
Data Parallel (DP): Simple model replication across GPUs
Distributed Data Parallel (DDP): Process-based parallelism with efficient gradient synchronization
Fully Sharded Data Parallel (FSDP): Memory-efficient parameter sharding across GPUs

Contributors

Vanshi Patel (002057295)
Malav Gajera (002050537)

Citation
If you use this code in your research, please cite our work:
@article{patel2025realtime,
  title={Real-Time Parallel Shoplifting Detection using Multimodal AI},
  author={Patel, Vanshi and Gajera, Malav},
  journal={Northeastern University Technical Report},
  year={2025}
}
Acknowledgments

UCF Crime dataset from the Center for Research in Computer Vision
This research was conducted at Northeastern University
