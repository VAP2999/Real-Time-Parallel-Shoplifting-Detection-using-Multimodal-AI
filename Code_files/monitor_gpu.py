# save as unified_gpu_monitor.py
import torch
import time
import wandb
import psutil
import GPUtil
import threading
import argparse
from datetime import datetime
import os

# Create directories for results
os.makedirs("./results", exist_ok=True)

def log_system_metrics(interval=0.5, log_to_wandb=True, project_name="gpu-monitoring"):
    """Monitor system metrics and log to W&B"""
    # Initialize W&B if requested
    if log_to_wandb:
        run_name = f"gpu-monitor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        try:
            wandb.init(project=project_name, name=run_name)
            
            # Add system info to wandb config
            wandb.config.update({
                "system": {
                    "hostname": os.uname()[1],
                    "cpu_count": psutil.cpu_count(),
                    "cpu_count_physical": psutil.cpu_count(logical=False),
                    "total_memory": psutil.virtual_memory().total / (1024 ** 3),  # GB
                    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                    "pytorch_version": torch.__version__
                }
            })
            
            # Add GPU info if available
            if torch.cuda.is_available():
                gpu_info = {}
                for i in range(torch.cuda.device_count()):
                    gpu_info[f"gpu{i}"] = {
                        "name": torch.cuda.get_device_name(i),
                        "capability": torch.cuda.get_device_capability(i)
                    }
                wandb.config.update({"gpu_info": gpu_info})
        except Exception as e:
            print(f"Warning: W&B initialization failed: {e}. Will continue with local logging only.")
            log_to_wandb = False
    
    print("\nStarting system monitoring")
    print("Press Ctrl+C to stop monitoring")
    
    # Start monitoring
    start_time = time.time()
    try:
        while True:
            # Get current timestamp and elapsed time
            current_time = time.time()
            elapsed = current_time - start_time
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # Create metrics dictionary
            metrics = {
                'timestamp': timestamp,
                'elapsed_time': elapsed,
            }
            
            # Get CPU and RAM stats
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                ram_percent = psutil.virtual_memory().percent
                metrics['cpu_percent'] = cpu_percent
                metrics['ram_percent'] = ram_percent
            except Exception as e:
                print(f"Error getting CPU stats: {e}")
            
            # Get GPU stats if available
            gpu_stats_str = ""
            if torch.cuda.is_available():
                try:
                    gpu_count = torch.cuda.device_count()
                    gpus = GPUtil.getGPUs()
                    
                    for i in range(min(gpu_count, len(gpus))):
                        gpu = gpus[i]
                        metrics[f'gpu{i}_util'] = gpu.load * 100
                        metrics[f'gpu{i}_memory_util'] = gpu.memoryUtil * 100
                        metrics[f'gpu{i}_memory_used'] = gpu.memoryUsed
                        metrics[f'gpu{i}_memory_total'] = gpu.memoryTotal
                        metrics[f'gpu{i}_temp'] = gpu.temperature
                        
                        # Add to status string
                        gpu_stats_str += f"GPU {i}: {gpu.load*100:.1f}% util, {gpu.memoryUtil*100:.1f}% mem, {gpu.temperature:.1f}°C | "
                except Exception as e:
                    print(f"Error getting GPU stats: {e}")
            
            # Print current stats to console
            status_line = f"Time: {elapsed:.1f}s | CPU: {metrics.get('cpu_percent', 0):.1f}% | RAM: {metrics.get('ram_percent', 0):.1f}% | {gpu_stats_str}"
            print(status_line, end="\r")
            
            # Log to wandb if initialized
            if log_to_wandb and wandb.run is not None:
                wandb.log(metrics)
            
            # Sleep until next interval
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        # Finish W&B run if initialized
        if log_to_wandb and wandb.run is not None:
            wandb.finish()

def stress_gpu(gpu_id=0, duration=60, tensor_size=10000):
    """Stress test a specific GPU"""
    if not torch.cuda.is_available():
        print(f"CUDA not available. Cannot stress test GPU.")
        return
        
    if gpu_id >= torch.cuda.device_count():
        print(f"GPU {gpu_id} not available. Maximum GPU ID is {torch.cuda.device_count()-1}")
        return
        
    print(f"Starting stress test on GPU {gpu_id} for {duration} seconds")
    device = torch.device(f"cuda:{gpu_id}")
    
    # Try to allocate a large tensor (reduce size if OOM)
    try:
        a = torch.randn(tensor_size, tensor_size, device=device)
        b = torch.randn(tensor_size, tensor_size, device=device)
    except RuntimeError as e:
        # Reduce tensor size if OOM
        if "out of memory" in str(e):
            tensor_size = tensor_size // 2
            print(f"Reduced tensor size to {tensor_size} due to memory constraints")
            a = torch.randn(tensor_size, tensor_size, device=device)
            b = torch.randn(tensor_size, tensor_size, device=device)
        else:
            raise e
    
    start_time = time.time()
    while time.time() - start_time < duration:
        # Perform matrix multiplication to stress the GPU
        c = torch.matmul(a, b)
        
        # Add some variation to the load
        if (time.time() - start_time) % 5 < 2.5:
            # More intense load
            for _ in range(5):
                c = torch.matmul(c, a) / tensor_size
        
        # Ensure operation completes
        torch.cuda.synchronize()
        
        # Brief pause to allow monitoring
        time.sleep(0.05)
    
    print(f"\nCompleted stress test on GPU {gpu_id}")

def stress_all_gpus(duration=60, tensor_size=5000):
    """Stress test all available GPUs simultaneously"""
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot stress test GPUs.")
        return
        
    gpu_count = torch.cuda.device_count()
    print(f"Starting stress test on all {gpu_count} GPUs for {duration} seconds")
    
    # Create and start threads for each GPU
    threads = []
    for gpu_id in range(gpu_count):
        thread = threading.Thread(target=stress_gpu, args=(gpu_id, duration, tensor_size))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print(f"Completed stress test on all {gpu_count} GPUs")

def main():
    parser = argparse.ArgumentParser(description='GPU Monitoring and Stress Testing')
    parser.add_argument('--mode', type=str, choices=['monitor', 'stress', 'both'], default='both',
                        help='Run mode: monitor, stress, or both')
    parser.add_argument('--interval', type=float, default=0.5, help='Monitoring interval in seconds')
    parser.add_argument('--duration', type=int, default=300, help='Stress test duration in seconds')
    parser.add_argument('--size', type=int, default=5000, help='Tensor size for testing')
    parser.add_argument('--gpu', type=int, default=-1, help='Specific GPU to stress (-1 for all)')
    parser.add_argument('--project', type=str, default='gpu-monitoring', help='W&B project name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    
    args = parser.parse_args()
    
    # Print system information
    print(f"System: {os.uname()[1]}")
    print(f"PyTorch {torch.__version__} | CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()} × {torch.cuda.get_device_name(0)}")
    else:
        print("No GPUs available")
    
    if args.mode == 'monitor':
        # Just monitoring
        log_system_metrics(args.interval, not args.no_wandb, args.project)
    elif args.mode == 'stress':
        # Just stress testing
        if args.gpu >= 0:
            stress_gpu(args.gpu, args.duration, args.size)
        else:
            stress_all_gpus(args.duration, args.size)
    else:  # 'both'
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(
            target=log_system_metrics, 
            args=(args.interval, not args.no_wandb, args.project)
        )
        monitor_thread.daemon = True  # Make thread terminate when main thread ends
        monitor_thread.start()
        
        # Run stress test in main thread
        time.sleep(2)  # Allow monitoring to start
        if args.gpu >= 0:
            stress_gpu(args.gpu, args.duration, args.size)
        else:
            stress_all_gpus(args.duration, args.size)
        
        # Wait for monitoring to catch up
        time.sleep(5)
        print("\nPress Ctrl+C to stop monitoring and exit")
        try:
            monitor_thread.join()
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main()