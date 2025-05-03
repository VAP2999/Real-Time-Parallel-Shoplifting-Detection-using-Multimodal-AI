#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import time

# Ensure absolute path
def ensure_absolute_path(path):
    """Convert relative path to absolute path if needed"""
    if not os.path.isabs(path):
        return os.path.abspath(path)
    return path

# Import the model architecture
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

# Simpler feature extraction - just as placeholder
def simple_feature_extraction(image_path):
    """Extract a simple feature vector from an image as placeholder"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to a standard size
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Generate a placeholder feature vector of the correct size
    # This is NOT how your real features were extracted, just a placeholder
    features = np.random.randn(1, 2048)  # Shape expected by your model
    
    # Also return the original image for visualization
    return features, img_rgb

# Function to load the anomaly detection model
def load_model(model_path, feature_dim=2048, hidden_dims=[1024, 512, 256, 128], num_classes=6):
    """Load the trained anomaly detection model"""
    # Initialize model
    model = AnomalyDetector(
        input_dim=feature_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes
    )
    
    # Load checkpoint - set weights_only=False to fix PyTorch 2.6 security issue
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # If the checkpoint has 'model_state_dict' key, use that
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Otherwise, assume the checkpoint itself is the state dictionary
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Best validation F1: {checkpoint.get('val_anomaly_f1', 'N/A')}")
    
    return model

# Function to load class labels from HDF5 file
def load_class_labels(h5_file):
    """Load class labels from HDF5 file"""
    try:
        with h5py.File(h5_file, 'r') as h5:
            label_map = {}
            try:
                for label, idx in h5['metadata']['label_map'].attrs.items():
                    label_map[idx] = label.decode('utf-8') if isinstance(label, bytes) else label
            except Exception as e:
                print(f"Warning: Couldn't load label map properly: {e}")
                # Try a different approach
                try:
                    for key in h5['metadata']['label_map'].attrs.keys():
                        idx = h5['metadata']['label_map'].attrs[key]
                        label = key.decode('utf-8') if isinstance(key, bytes) else key
                        label_map[idx] = label
                except Exception as e2:
                    print(f"Warning: Second attempt failed: {e2}")
                    
            # If still empty, create default
            if not label_map:
                print("Creating default label mapping")
                label_map = {
                    0: "Normal",
                    1: "Abnormal"
                }
                
        return label_map
    except Exception as e:
        print(f"Error loading h5 file: {e}")
        # Create a default label map
        return {i: f"Class_{i}" for i in range(6)}

# Process video and detect anomalies
def detect_anomalies_in_video(video_path, model, output_path=None, threshold=0.5, label_map=None):
    """Process a video and detect anomalies"""
    # Check if the video file exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Initialize lists to store frames
    frames = []
    
    # Sample frames (process every 10th frame to save time)
    sample_rate = 10
    sampled_frames = []
    frame_features = []
    
    # Process frames
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every nth frame
        if frame_index % sample_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append(frame_rgb)
            
            # Generate random feature vectors (placeholder)
            features = np.random.randn(1, 2048)
            frame_features.append(features)
        
        frames.append(frame)
        frame_index += 1
        
        # Print progress
        if frame_index % 100 == 0:
            print(f"Processed {frame_index}/{frame_count} frames")
    
    cap.release()
    
    print(f"Sampled {len(sampled_frames)} frames for analysis")
    
    # Combine features
    if not frame_features:
        raise ValueError("No frames were successfully processed")
    
    # Process features with the model
    anomaly_scores = []
    action_predictions = []
    
    for features in frame_features:
        # Convert to tensor
        feature_tensor = torch.FloatTensor(features)
        
        # Get predictions
        with torch.no_grad():
            anomaly_prediction, action_prediction = model(feature_tensor)
            
            # Convert to numpy
            anomaly_score = anomaly_prediction.cpu().numpy().item() if anomaly_prediction.dim() == 0 else anomaly_prediction.cpu().numpy()[0]
            action_probs = torch.softmax(action_prediction, dim=1).cpu().numpy()[0]
            
            anomaly_scores.append(anomaly_score)
            action_predictions.append(np.argmax(action_probs))
    
    # Interpolate scores for all frames
    all_anomaly_scores = []
    all_action_predictions = []
    
    for i in range(frame_index):
        sample_idx = i // sample_rate if i // sample_rate < len(anomaly_scores) else len(anomaly_scores) - 1
        all_anomaly_scores.append(anomaly_scores[sample_idx])
        all_action_predictions.append(action_predictions[sample_idx])
    
    # Count anomalies
    anomaly_frames = [score >= threshold for score in all_anomaly_scores]
    anomaly_count = sum(anomaly_frames)
    
    print(f"Detected anomalies in {anomaly_count} out of {len(frames)} frames ({anomaly_count/len(frames)*100:.2f}%)")
    print(f"Average anomaly score: {np.mean(all_anomaly_scores):.4f}")
    
    # Create output video if requested
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(os.path.dirname(output_path), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create anomaly score plot
        plt.figure(figsize=(12, 6))
        plt.plot(all_anomaly_scores, color='blue', alpha=0.7)
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        plt.xlabel('Frame Number')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Scores Throughout Video')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Highlight anomaly regions
        for i in range(len(anomaly_frames)):
            if anomaly_frames[i]:
                plt.axvspan(i, i+1, alpha=0.2, color='red')
        
        # Save plot
        plot_path = os.path.join(plots_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_anomaly_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Anomaly score plot saved to {plot_path}")
        
        # Create action prediction plot if we have class labels
        if label_map:
            action_labels = [label_map.get(pred, f"Class_{pred}") for pred in all_action_predictions]
            
            # Count occurrences of each action
            from collections import Counter
            action_counts = Counter(action_labels)
            
            # Create bar chart
            plt.figure(figsize=(12, 6))
            actions = list(action_counts.keys())
            counts = list(action_counts.values())
            
            plt.bar(actions, counts)
            plt.xlabel('Action Class')
            plt.ylabel('Frame Count')
            plt.title('Action Classes Throughout Video')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save plot
            action_plot_path = os.path.join(plots_dir, f"{os.path.splitext(os.path.basename(output_path))[0]}_action_plot.png")
            plt.savefig(action_plot_path)
            plt.close()
            print(f"Action prediction plot saved to {action_plot_path}")
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames with annotations
        for i, frame in enumerate(frames):
            if i < len(all_anomaly_scores):  # Safety check
                # Add anomaly score
                score = all_anomaly_scores[i]
                is_anomaly = score >= threshold
                
                # Create text
                score_text = f"Anomaly: {score:.2f}" 
                
                # Add action class if available
                if i < len(all_action_predictions):
                    action_pred = all_action_predictions[i]
                    if label_map:
                        action_label = label_map.get(action_pred, f"Class_{action_pred}")
                        action_text = f"Action: {action_label}"
                    else:
                        action_text = f"Action: Class {action_pred}"
                else:
                    action_text = "Action: Unknown"
                
                # Add rectangle background for text
                cv2.rectangle(frame, (10, 10), (300, 70), (0, 0, 0), -1)
                
                # Add text
                color = (0, 0, 255) if is_anomaly else (0, 255, 0)  # Red if anomaly, green otherwise
                cv2.putText(frame, score_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, action_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Add border if anomaly
                if is_anomaly:
                    cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 5)
            
            # Write frame to output video
            out.write(frame)
            
            # Print progress
            if i % 100 == 0:
                print(f"Writing frame {i}/{len(frames)} to output video")
        
        # Release video writer
        out.release()
        print(f"Output video saved to {output_path}")
    
    return all_anomaly_scores, all_action_predictions

# Process image and detect anomaly
def detect_anomaly_in_image(image_path, model, output_path=None, threshold=0.5, label_map=None):
    """Process an image and detect anomaly"""
    # Extract placeholder features
    features, img = simple_feature_extraction(image_path)
    
    # Convert to tensor
    feature_tensor = torch.FloatTensor(features)
    
    # Process with anomaly detection model
    with torch.no_grad():
        anomaly_prediction, action_prediction = model(feature_tensor)
        
        # Convert to numpy - ensure scalar is converted properly
        anomaly_score = anomaly_prediction.cpu().numpy().item() if anomaly_prediction.dim() == 0 else anomaly_prediction.cpu().numpy()[0]
        action_probs = torch.softmax(action_prediction, dim=1).cpu().numpy()[0]
    
    # Classify anomaly based on threshold
    anomaly_detected = anomaly_score >= threshold
    
    # Get action prediction
    action_pred = np.argmax(action_probs)
    
    # Print results
    print(f"Anomaly score: {anomaly_score:.4f} ({'Anomaly' if anomaly_detected else 'Normal'})")
    if label_map:
        action_label = label_map.get(action_pred, f"Class_{action_pred}")
        print(f"Action prediction: {action_label} (confidence: {action_probs[action_pred]:.4f})")
    else:
        print(f"Action prediction: Class {action_pred} (confidence: {action_probs[action_pred]:.4f})")
    
    # Visualize result
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add annotations to image
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        
        # Add title with anomaly score
        title = f"Anomaly: {anomaly_score:.2f} ({'ANOMALY' if anomaly_detected else 'NORMAL'})"
        if label_map:
            action_label = label_map.get(action_pred, f"Class_{action_pred}")
            title += f"\nAction: {action_label} ({action_probs[action_pred]:.2f})"
        else:
            title += f"\nAction: Class {action_pred} ({action_probs[action_pred]:.2f})"
        
        plt.title(title, color='red' if anomaly_detected else 'green', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Output image saved to {output_path}")
    
    return anomaly_score, action_probs

def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection on Images/Videos')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or video file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Path to save output file')
    parser.add_argument('--h5_file', type=str, default=None, help='Path to HDF5 file with class labels')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for anomaly detection')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    args.input = ensure_absolute_path(args.input)
    args.model = ensure_absolute_path(args.model)
    if args.output:
        args.output = ensure_absolute_path(args.output)
    if args.h5_file:
        args.h5_file = ensure_absolute_path(args.h5_file)
    
    # Determine input type (image or video)
    file_ext = os.path.splitext(args.input)[1].lower()
    is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv']
    
    # Set default output path if not provided
    if not args.output:
        if is_video:
            args.output = os.path.splitext(args.input)[0] + '_output.mp4'
        else:
            args.output = os.path.splitext(args.input)[0] + '_output.png'
    
    print(f"Processing {'video' if is_video else 'image'}: {args.input}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    
    # Load anomaly detection model
    model = load_model(args.model)
    
    # Load class labels if h5_file is provided
    label_map = None
    if args.h5_file:
        label_map = load_class_labels(args.h5_file)
        print(f"Loaded {len(label_map)} class labels")
    
    # Process input
    start_time = time.time()
    
    try:
        if is_video:
            anomaly_scores, action_preds = detect_anomalies_in_video(
                args.input, model, args.output, args.threshold, label_map
            )
        else:
            anomaly_score, action_probs = detect_anomaly_in_image(
                args.input, model, args.output, args.threshold, label_map
            )
        
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error processing input: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()