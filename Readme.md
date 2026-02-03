# Robotics + AI/ML Development Docker Environment

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-green.svg)](https://docs.ros.org/en/humble/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-grade Docker environment for autonomous robotics development with integrated AI/ML capabilities. Built for professional robotics engineers, researchers, and developers working on computer vision, autonomous navigation, and intelligent systems.

**Developer:** Lasantha Kulasooriya  
**Location:** Sri Lanka 🇱🇰  
**Version:** 1.0.0  
**Last Updated:** February 2026

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Stack Details](#-stack-details)
- [Architecture](#-architecture)
- [Management Commands](#-management-commands)
- [Development Workflow](#-development-workflow)
- [Testing & Verification](#-testing--verification)
- [Hardware Support](#-hardware-support)
- [Docker Compose Setup](#-docker-compose-setup)
- [Advanced Configuration](#-advanced-configuration)
- [Troubleshooting](#-troubleshooting)
- [Performance Optimization](#-performance-optimization)
- [Best Practices](#-best-practices)
- [Examples & Tutorials](#-examples--tutorials)
- [Contributing](#-contributing)
- [FAQ](#-faq)
- [Changelog](#-changelog)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact & Support](#-contact--support)

---

## 🎯 Overview

This Docker environment provides a complete, production-ready development stack for building autonomous robotics systems. It combines the power of ROS2 Humble with modern AI/ML frameworks, enabling seamless development from simulation to deployment.

### **Why Use This Environment?**

✅ **Reproducible** - Same environment on laptop, workstation, and robot  
✅ **Isolated** - No conflicts with host system packages  
✅ **GPU-Accelerated** - Full CUDA support for AI/ML workloads  
✅ **Production-Ready** - Battle-tested in real-world robotics projects  
✅ **Well-Tested** - 700+ packages tested and verified  
✅ **Comprehensive** - Everything needed for robotics development  
✅ **Documented** - Extensive guides and examples  
✅ **Maintained** - Regular updates and community support

### **Perfect For:**

- 🤖 **Autonomous Mobile Robots** - Navigation, mapping, obstacle avoidance
- 📹 **Computer Vision** - Object detection, tracking, segmentation
- 🎯 **Deep Learning** - Model training, inference, optimization
- 🗺️ **SLAM & Mapping** - Simultaneous localization and mapping
- 🦾 **Robotic Manipulation** - Arm control, pick-and-place, grasping
- 🚗 **Autonomous Vehicles** - Self-driving systems, sensor fusion
- 📡 **IoT Robotics** - Connected robots, cloud integration
- 🎓 **Research & Education** - Academic projects, teaching, learning

---

## ✨ Features

### **🤖 ROS2 Robotics Framework**

**Core ROS2 Packages (350+)**
- ✅ ROS2 Humble Desktop Full - Complete ROS2 installation
- ✅ ros-base & ros-core - Foundation libraries
- ✅ ros-dev-tools - Development utilities
- ✅ colcon - Build system and tools
- ✅ rosdep - Dependency management

**Navigation & SLAM**
- ✅ Navigation2 - Full autonomous navigation stack
  - Path planning (NavFn, Smac, TEB)
  - Behavior trees for decision making
  - Costmap layers for obstacle handling
  - Controller plugins (DWB, RPP, MPPI)
  - Recovery behaviors
- ✅ SLAM Toolbox - Mapping and localization
  - Online and offline SLAM
  - Map serialization
  - Loop closure detection
- ✅ Robot Localization - Multi-sensor fusion
  - EKF and UKF filters
  - IMU, odometry, GPS integration

**Control & Hardware Interfaces**
- ✅ ros2_control - Hardware abstraction layer
- ✅ ros2_controllers - Ready-to-use controllers
  - Differential drive controller
  - Ackermann steering controller
  - Mecanum drive controller
  - Joint state broadcaster
  - Joint trajectory controller
  - Gripper action controller
  - And 10+ more controller types
- ✅ controller_manager - Runtime controller management

**Sensors & Perception**
- ✅ Camera Drivers
  - Intel RealSense (RGB-D cameras)
  - Stereolabs ZED (stereo cameras)
  - USB cameras via v4l2
  - Image transport plugins
  - Compressed image transport
- ✅ LiDAR Drivers
  - RPLidar (2D laser scanners)
  - Laser geometry utilities
  - Laser filters
- ✅ Transforms & Geometry
  - TF2 transform library
  - Static and dynamic transforms
  - Geometry message conversions
  - Sensor message conversions

**Visualization & Debugging**
- ✅ RViz2 - 3D visualization
  - Robot model display
  - Sensor data visualization
  - Path and trajectory display
  - Interactive markers
  - Custom plugins support
- ✅ rqt Tools Suite
  - Topic monitor (rqt_graph)
  - Image viewer (rqt_image_view)
  - Plot tool (rqt_plot)
  - TF tree viewer (rqt_tf_tree)
  - Console logs
  - Service caller
  - And 20+ more tools

**Communication & Integration**
- ✅ ROS Bridge - Web integration
  - WebSocket server
  - JSON message conversion
  - Web-based control interfaces
- ✅ Teleop - Remote control
  - Keyboard teleoperation
  - Joystick control
  - Twist message publishing

### **🧠 AI/ML & Computer Vision**

**Deep Learning Frameworks**
- ✅ PyTorch 2.5.1 - Industry-standard deep learning
  - CUDA 12.1 acceleration
  - Automatic mixed precision (AMP)
  - Distributed training support
  - TorchScript for deployment
  - C++ API (LibTorch)
- ✅ TorchVision 0.20.1 - Computer vision models
  - Pre-trained models (ResNet, EfficientNet, etc.)
  - Image transformations
  - Video processing
  - Detection and segmentation tools

**Object Detection & Tracking**
- ✅ YOLO v8/v11 (Ultralytics 8.3.227)
  - Object detection (80+ classes)
  - Instance segmentation
  - Pose estimation
  - Classification
  - Model export (ONNX, TensorRT)
  - Custom dataset training
  - Real-time inference
- ✅ Supervision 0.27.0 - Detection utilities
  - Bounding box operations
  - Annotation tools
  - Video processing
  - Tracking helpers
- ✅ DeepSORT 1.3.2 - Multi-object tracking
  - ID persistence across frames
  - Occlusion handling
  - Re-identification
  - Real-time performance

**Computer Vision Libraries**
- ✅ OpenCV 4.11.0 (with contrib modules)
  - Image processing
  - Video I/O
  - Feature detection (SIFT, SURF, ORB)
  - Camera calibration
  - ArUco marker detection
  - DNN module for inference
  - CUDA-accelerated operations
- ✅ MediaPipe 0.10.18 - ML solutions
  - Face detection & mesh
  - Hand tracking (21 keypoints)
  - Pose estimation (33 keypoints)
  - Holistic tracking
  - Object detection
  - Selfie segmentation
- ✅ Pillow 12.0.0 - Image library
  - Image I/O
  - Format conversions
  - Basic transformations

**Model Optimization**
- ✅ ONNX 1.19.1 - Model interchange format
  - PyTorch to ONNX conversion
  - Model optimization
  - Cross-framework compatibility
- ✅ ONNX Runtime GPU 1.23.2
  - Accelerated inference
  - 2-10x faster than PyTorch
  - Production deployment
- ✅ TensorRT - NVIDIA inference optimizer
  - 5-10x inference speedup
  - FP16/INT8 quantization
  - Dynamic shapes
  - Optimal for Jetson devices

**NLP & Transformers**
- ✅ HuggingFace Ecosystem
  - Transformers - Pre-trained models
  - Tokenizers - Fast text processing
  - Datasets - Easy dataset loading
  - Accelerate - Training optimization
  - safetensors - Secure model format
  - sentencepiece - Subword tokenization

### **📊 Scientific Computing**

**Core Scientific Libraries**
- ✅ NumPy 1.26.4 - Array computing
  - N-dimensional arrays
  - Linear algebra
  - Fourier transforms
  - Random number generation
- ✅ SciPy 1.15.3 - Scientific algorithms
  - Optimization
  - Integration
  - Interpolation
  - Signal processing
  - Statistics
  - Sparse matrices
- ✅ pandas 2.0.3 - Data analysis
  - DataFrame operations
  - CSV/Excel I/O
  - Time series analysis
  - Data cleaning
  - SQL-like operations
- ✅ Polars 1.35.2 - Fast DataFrames
  - 5-10x faster than pandas
  - Lazy evaluation
  - Parallel processing
  - Memory efficient

**Machine Learning**
- ✅ scikit-learn 1.7.2
  - Classification algorithms
  - Regression models
  - Clustering (K-means, DBSCAN)
  - Dimensionality reduction (PCA)
  - Model selection
  - Preprocessing utilities

**GPU-Accelerated Computing**
- ✅ CuPy 13.6.0 - NumPy on GPU
  - Drop-in NumPy replacement
  - CUDA kernels
  - 10-100x speedup for array ops
  - Custom CUDA code support
- ✅ Numba 0.62.1 - JIT compiler
  - Automatic parallelization
  - CUDA kernel generation
  - No code changes needed
  - Python to machine code

**Visualization**
- ✅ Matplotlib 3.10.7
  - 2D plotting
  - 3D plotting
  - Animation support
  - Publication-quality figures
- ✅ Seaborn 0.13.2
  - Statistical visualizations
  - Beautiful default themes
  - Built on matplotlib

### **🔌 Communication & Integration**

**Data Storage & Caching**
- ✅ Redis - In-memory data store
  - Sub-millisecond latency
  - Pub/sub messaging
  - Data structures (strings, lists, sets, hashes)
  - Persistence options
  - Cluster support
- ✅ hiredis - Fast Redis parser
  - C-based parser
  - 5-10x faster than pure Python

**IoT & Messaging**
- ✅ MQTT (paho-mqtt 2.1.0)
  - Publish/subscribe pattern
  - Quality of Service (QoS)
  - Persistent sessions
  - SSL/TLS support
  - IoT device communication

**Hardware Communication**
- ✅ CAN bus (python-can 3.3.2)
  - SocketCAN interface
  - Motor controller communication
  - Industrial protocols
  - Message filtering
- ✅ Serial (pyserial 3.5)
  - UART/RS-232/RS-485
  - Arduino/microcontroller communication
  - Sensor interfacing
  - GPS modules

**Web & HTTP**
- ✅ Flask 3.1.2 - Web framework
  - RESTful API development
  - WebSocket support
  - Template rendering
  - Session management
- ✅ flask-cors 6.0.2 - CORS handling
- ✅ Flask-Login 0.6.3 - User authentication
- ✅ requests 2.32.5 - HTTP client
  - GET/POST/PUT/DELETE
  - JSON handling
  - File uploads
  - Session persistence

### **🛠️ Development Tools**

**Interactive Development**
- ✅ Jupyter - Notebook environment
  - Interactive Python
  - Markdown documentation
  - Inline plots
  - Code cells
- ✅ IPython - Enhanced Python shell
  - Tab completion
  - Magic commands
  - History
  - Rich output

**Code Quality**
- ✅ Black 25.9.0 - Code formatter
  - PEP 8 compliant
  - Deterministic formatting
  - Fast
- ✅ Flake8 7.3.0 - Linter
  - Style checker
  - Error detection
  - Complexity analysis
- ✅ Pylint 4.0.4 - Static analyzer
  - Code analysis
  - Bug detection
  - Code smells

**Testing**
- ✅ pytest 6.2.5 - Testing framework
  - Simple test writing
  - Fixtures
  - Parametrization
  - Plugin system
- ✅ pytest-cov 3.0.0 - Coverage reporting

**Utilities**
- ✅ tqdm 4.67.1 - Progress bars
- ✅ tabulate 0.9.0 - Pretty tables
- ✅ colorama 0.4.4 - Colored terminal
- ✅ coloredlogs 15.0.1 - Colored logs

**Configuration Management**
- ✅ PyYAML 6.0.3 - YAML parser
- ✅ python-dotenv 1.2.1 - Environment variables
- ✅ omegaconf 2.3.0 - Hierarchical configs
- ✅ pydantic 2.12.4 - Data validation
- ✅ pydantic-settings 2.12.0 - Settings management

**Math & Transforms**
- ✅ transforms3d 0.4.2 - 3D transformations
  - Euler angles
  - Quaternions
  - Rotation matrices
  - Homogeneous transforms
- ✅ sympy 1.14.0 - Symbolic mathematics
  - Algebra
  - Calculus
  - Equation solving
  - LaTeX output

---

## 🚀 Quick Start

### **1. Prerequisites Check**
```bash
# Check Docker
docker --version

# Check NVIDIA GPU
nvidia-smi

# Check disk space (need 50GB+)
df -h
```

### **2. Clone Repository**
```bash
git clone https://github.com/LasaK97/robotics-ai-ml-docker-env.git
cd robotics-ai-ml-docker-env
```

### **3. Build Docker Image**
```bash
# This takes 30-60 minutes on first build
./build.sh
```

**Build process includes:**
- ⏳ Downloading base CUDA image (4 GB)
- ⏳ Installing system packages (5-10 min)
- ⏳ Installing ROS2 Humble (10-15 min)
- ⏳ Installing PyTorch & AI/ML packages (15-20 min)
- ⏳ Final configuration (5 min)

### **4. Launch Container**
```bash
./run.sh
```

You should see:
```
╔════════════════════════════════════════════════════════════╗
║     Robotics + AI/ML Development Docker Environment       ║
╚════════════════════════════════════════════════════════════╝
         Developer: Lasantha Kulasooriya                     

✓ Image found: robotics-aiml-dev:latest
✓ Workspace: /home/user/Desktop/ros2_humble_ws
✓ X11 access enabled for GUI applications

Launching container...
root@hostname:~/ros2_ws#
```

### **5. Verify Installation**
```bash
# Test ROS2
ros2 topic list

# Test GPU
nvidia-smi

# Test PyTorch
python3 << EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

# Test YOLO
python3 -c "from ultralytics import YOLO; print('YOLO ready')"

# Test OpenCV
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### **6. Run First Example**
```bash
# Test RViz visualization
rviz2

# Test a simple ROS2 node
ros2 run demo_nodes_cpp talker
```

**Success!** 🎉 You're ready to develop.

---

## 💻 System Requirements

### **Minimum Requirements**

| Component | Specification |
|-----------|---------------|
| **OS** | Ubuntu 22.04 or 24.04 |
| **CPU** | 4 cores / 8 threads |
| **RAM** | 16 GB |
| **GPU** | NVIDIA with CUDA 8.7+ |
| **Storage** | 50 GB free (SSD recommended) |
| **Network** | 20 Mbps (for initial build) |

### **Recommended Requirements**

| Component | Specification |
|-----------|---------------|
| **OS** | Ubuntu 24.04 LTS |
| **CPU** | 8+ cores (Intel i7/i9, AMD Ryzen 7/9) |
| **RAM** | 32 GB or more |
| **GPU** | NVIDIA RTX 30xx/40xx series |
| **Storage** | 100 GB NVMe SSD |
| **Network** | 50+ Mbps |

### **Tested Hardware Configurations**

| Device | CPU | GPU | RAM | Status |
|--------|-----|-----|-----|--------|
| ASUS ROG G18 | i9-14900HX | RTX 4060 Laptop | 32GB | ✅ Fully Tested |
| Jetson AGX Orin | ARM Cortex-A78AE | Ampere iGPU | 64GB | ✅ Fully Tested |
| Desktop Workstation | i9-13900K | RTX 4090 | 64GB | ✅ Verified |
| Dell Precision | Xeon W-2295 | Quadro RTX 5000 | 64GB | ✅ Verified |
| Lenovo ThinkPad P1 | i7-12800H | RTX 3070 Ti | 32GB | ✅ Verified |

### **NVIDIA GPU Compatibility**

**Supported GPUs (CUDA Compute Capability 8.7+):**
- ✅ RTX 40 Series: 4090, 4080, 4070, 4060
- ✅ RTX 30 Series: 3090 Ti, 3090, 3080 Ti, 3080, 3070, 3060
- ✅ RTX 20 Series: 2080 Ti, 2080 Super, 2070, 2060
- ✅ Jetson: AGX Orin, Orin NX, Orin Nano
- ✅ Data Center: A100, A40, A30, A10

**Not Recommended (older architectures):**
- ⚠️ GTX 16 Series (limited features)
- ❌ GTX 10 Series (too old)

---

## 📦 Installation

### **Step 1: Install Docker Engine**

#### **For Ubuntu 22.04/24.04:**
```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# Add user to docker group (no sudo needed)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker run hello-world
```

### **Step 2: Install NVIDIA Container Toolkit**

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU information displayed.

### **Step 3: Clone Repository**

```bash
# Clone to your preferred location
cd ~/Desktop
git clone https://github.com/LasaK97/robotics-ai-ml-docker-env.git
cd robotics-ai-ml-docker-env

# Make scripts executable
chmod +x *.sh

# Verify files
ls -lh
```

You should see:
```
-rw-r--r-- Dockerfile
-rw-r--r-- docker-compose.yml
-rw-r--r-- .dockerignore
-rwxr-xr-x build.sh
-rwxr-xr-x run.sh
-rwxr-xr-x manage.sh
-rwxr-xr-x test.sh
-rw-r--r-- README.md
```

### **Step 4: Configure Workspace (Optional)**

By default, the workspace is at `~/Desktop/ros2_humble_ws`. To change:

```bash
# Edit run.sh
nano run.sh

# Change this line:
WORKSPACE="$HOME/Desktop/ros2_humble_ws"

# To your preferred location:
WORKSPACE="/your/custom/path"
```

### **Step 5: Build Docker Image**

```bash
./build.sh
```

**What happens during build:**
1. Downloads CUDA 12.6 base image (~4 GB)
2. Installs system dependencies
3. Adds ROS2 Humble repository
4. Installs 350+ ROS2 packages
5. Installs PyTorch with CUDA support
6. Installs AI/ML libraries
7. Installs scientific computing packages
8. Configures environment
9. Creates workspace structure

**Build time:** 30-60 minutes (first time)

**Common build issues:**
- Network timeout → Just retry: `./build.sh`
- Disk full → Clean space: `docker system prune -a`
- Memory error → Close other applications

### **Step 6: Launch Container**

```bash
./run.sh
```

First launch creates a new container. Subsequent launches attach to existing container.

---

## 🎮 Usage

### **Basic Container Operations**

#### **Start Container**
```bash
./run.sh
# or
./manage.sh start
./manage.sh shell
```

#### **Stop Container**
```bash
# From inside container
exit

# From host
./manage.sh stop
```

#### **Check Status**
```bash
./manage.sh status
```

#### **Remove Container** (keeps image)
```bash
./manage.sh remove
```

### **Inside Container Commands**

#### **Quick Aliases**
```bash
cb                      # Build ROS2 workspace
cbs package_name        # Build specific package
cs                      # Source workspace
```

These expand to:
```bash
cb   = cd /root/ros2_ws && colcon build --symlink-install
cbs  = cd /root/ros2_ws && colcon build --symlink-install --packages-select
cs   = source /root/ros2_ws/install/setup.bash
```

#### **ROS2 Commands**
```bash
# List topics
ros2 topic list

# Echo topic
ros2 topic echo /topic_name

# Publish to topic
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "..."

# List nodes
ros2 node list

# Node info
ros2 node info /node_name

# Launch file
ros2 launch package_name launch_file.py

# Run node
ros2 run package_name node_name

# Check ROS2 installation
ros2 doctor
```

#### **Visualization**
```bash
# Launch RViz2
rviz2

# Launch rqt
rqt

# View images
rqt_image_view

# Plot topics
rqt_plot

# View TF tree
rqt_tf_tree
```

#### **GPU Monitoring**
```bash
# Check GPU status
nvidia-smi

# Watch GPU (updates every 1s)
watch -n1 nvidia-smi

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

#### **Python Development**
```bash
# Interactive Python
ipython

# Jupyter notebook
jupyter notebook --ip=0.0.0.0 --allow-root

# Run Python script
python3 script.py

# Install package
pip3 install package_name
```

---

## 📚 Stack Details

### **Docker Image Layers**

```
Layer 1: nvidia/cuda:12.6.0-devel-ubuntu22.04 (4 GB)
         ├── CUDA Toolkit 12.6
         ├── cuDNN 9.3
         └── NVIDIA drivers

Layer 2: System packages (2 GB)
         ├── build-essential, cmake, git
         ├── Python 3.10 + development headers
         └── System libraries

Layer 3: ROS2 Humble (3 GB)
         ├── 350+ ROS2 packages
         ├── Navigation2
         └── Visualization tools

Layer 4: PyTorch & AI/ML (4 GB)
         ├── PyTorch 2.5.1
         ├── TorchVision
         └── ONNX Runtime

Layer 5: Computer Vision (2 GB)
         ├── OpenCV 4.11
         ├── YOLO
         └── MediaPipe

Layer 6: Scientific Computing (1 GB)
         ├── NumPy, SciPy, pandas
         ├── scikit-learn
         └── CuPy

Layer 7: Communication (500 MB)
         ├── Redis, MQTT
         ├── Flask
         └── Serial/CAN

Layer 8: Development Tools (1 GB)
         ├── Jupyter
         ├── Testing frameworks
         └── Code formatters

Total Size: ~18 GB
```

### **Environment Variables**

```bash
# CUDA
CUDA_HOME=/usr/local/cuda-12.6
CUDA_ARCH_LIST=8.7;8.9
PATH=${CUDA_HOME}/bin:${PATH}
LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# ROS2
ROS_VERSION=2
ROS_PYTHON_VERSION=3
ROS_DISTRO=humble
ROS_DOMAIN_ID=0
ROS_LOCALHOST_ONLY=0
RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Python
PYTHONPATH=/root/ros2_ws/install/lib/python3.10/site-packages
```

### **Installed Packages by Category**

**ROS2 Core:** 50 packages  
**Navigation:** 25 packages  
**Control:** 20 packages  
**Sensors:** 30 packages  
**Visualization:** 15 packages  
**Utilities:** 210 packages  

**PyTorch Ecosystem:** 10 packages  
**Computer Vision:** 8 packages  
**Scientific Computing:** 12 packages  
**Communication:** 8 packages  
**Development Tools:** 30 packages  
**Utilities:** 182 packages  

**Total: 700+ packages**

---

## 🏗️ Architecture

### **System Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                    Host System                          │
│                  Ubuntu 22.04/24.04                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │         Docker Container                          │ │
│  │         Ubuntu 22.04 + ROS2 Humble                │ │
│  ├───────────────────────────────────────────────────┤ │
│  │                                                   │ │
│  │  ┌──────────────┐  ┌──────────────┐             │ │
│  │  │  ROS2 Humble │  │  Navigation2 │             │ │
│  │  └──────────────┘  └──────────────┘             │ │
│  │                                                   │ │
│  │  ┌──────────────┐  ┌──────────────┐             │ │
│  │  │ CUDA 12.6    │  │ PyTorch 2.5  │             │ │
│  │  └──────────────┘  └──────────────┘             │ │
│  │                                                   │ │
│  │  ┌──────────────┐  ┌──────────────┐             │ │
│  │  │  YOLO v8/11  │  │  OpenCV 4.11 │             │ │
│  │  └──────────────┘  └──────────────┘             │ │
│  │                                                   │ │
│  └───────────────────────────────────────────────────┘ │
│                         ↕                               │
│  ┌───────────────────────────────────────────────────┐ │
│  │           Shared Workspace                        │ │
│  │     /home/user/Desktop/ros2_humble_ws             │ │
│  │              (Bidirectional sync)                 │ │
│  └───────────────────────────────────────────────────┘ │
│                         ↕                               │
│  ┌───────────────────────────────────────────────────┐ │
│  │           NVIDIA GPU (RTX 4060)                   │ │
│  │         Full GPU passthrough to container         │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### **Data Flow**

```
Developer's Code (Host)
         ↓
Workspace Directory (Shared)
         ↓
Container (/root/ros2_ws)
         ↓
ROS2 Build System (colcon)
         ↓
Compiled Packages
         ↓
ROS2 Runtime (nodes, topics)
         ↓
Sensors/Actuators or Simulation
```

### **Network Architecture**

```
Container (network_mode: host)
    ├── ROS2 DDS (UDP multicast)
    │   └── Ports: 7400-7500
    │
    ├── Redis
    │   └── Port: 6379
    │
    ├── LiveKit (if using docker-compose)
    │   ├── HTTP: 7880
    │   ├── HTTPS: 7881
    │   └── WebRTC: 7882 (UDP)
    │
    └── Flask API (optional)
        └── Port: 5000
```

---

## 🎛️ Management Commands

### **Complete Command Reference**

#### **./manage.sh**

Full-featured container management script with 9 commands:

```bash
./manage.sh [COMMAND]
```

**Available Commands:**

| Command | Description | Usage Example |
|---------|-------------|---------------|
| `start` | Start stopped container | `./manage.sh start` |
| `stop` | Stop running container | `./manage.sh stop` |
| `restart` | Restart container | `./manage.sh restart` |
| `shell` | Open bash shell | `./manage.sh shell` |
| `logs` | View container logs | `./manage.sh logs` |
| `status` | Show container status | `./manage.sh status` |
| `remove` | Remove container | `./manage.sh remove` |
| `gpu` | Test GPU access | `./manage.sh gpu` |
| `build` | Build ROS2 workspace | `./manage.sh build` |
| `clean` | Clean build artifacts | `./manage.sh clean` |
| `help` | Show help message | `./manage.sh help` |

**Examples:**

```bash
# Check if container is running
./manage.sh status

# Start container and open shell
./manage.sh start
./manage.sh shell

# Test GPU inside container
./manage.sh gpu

# Build all ROS2 packages
./manage.sh build

# Clean and rebuild
./manage.sh clean
./manage.sh build

# View real-time logs
./manage.sh logs

# Stop container when done
./manage.sh stop
```

#### **./build.sh**

Builds the Docker image with comprehensive checks:

```bash
./build.sh
```

**Features:**
- ✅ Checks Docker is running
- ✅ Verifies NVIDIA GPU access
- ✅ Checks disk space (needs 30GB+)
- ✅ Saves build log to `build.log`
- ✅ Shows progress and time estimates
- ✅ Verifies successful build

**Options:**
```bash
# Standard build
./build.sh

# View build log
tail -f build.log

# Rebuild from scratch (no cache)
docker build --no-cache -t robotics-aiml-dev:latest .
```

#### **./run.sh**

Launches container with proper configuration:

```bash
./run.sh
```

**Features:**
- ✅ Checks if image exists
- ✅ Handles existing containers
- ✅ Mounts workspace
- ✅ Enables GPU access
- ✅ Configures X11 for GUI
- ✅ Sets up networking

**What it does:**
- If container doesn't exist → Creates new one
- If container exists but stopped → Starts it
- If container is running → Attaches to it

#### **./test.sh**

Comprehensive environment verification:

```bash
./test.sh
```

**Tests performed:**
1. ✅ Docker image exists
2. ✅ Container exists
3. ✅ Container is running
4. ✅ GPU accessible
5. ✅ CUDA version correct
6. ✅ ROS2 Humble working
7. ✅ PyTorch installed
8. ✅ PyTorch CUDA enabled
9. ✅ YOLO available
10. ✅ OpenCV working
11. ✅ Redis client installed
12. ✅ Flask installed

**Output example:**
```
╔════════════════════════════════════════════════════════════╗
║         Robotics + AI/ML Environment Verification         ║
╚════════════════════════════════════════════════════════════╝
         Developer: Lasantha Kulasooriya                     

1. Checking Docker image... ✓
2. Checking container... ✓
3. Checking container status... ✓ (running)
4. Testing GPU access... ✓
5. Checking CUDA version... ✓ (12.6)
6. Testing ROS2 Humble... ✓
7. Testing PyTorch... ✓ (2.5.1+cu121)
8. Testing PyTorch CUDA... ✓ (1 GPU)
9. Testing YOLO (Ultralytics)... ✓
10. Testing OpenCV... ✓ (4.11.0)
11. Testing Redis client... ✓
12. Testing Flask... ✓

╔════════════════════════════════════════════════════════════╗
║           All Tests Passed Successfully! ✓                ║
╚════════════════════════════════════════════════════════════╝

Your environment is ready for development!
```

---

## 💼 Development Workflow

### **Typical Development Session**

#### **Morning: Start Development**
```bash
# 1. Navigate to project
cd ~/Desktop/robotics-ai-ml-docker-env

# 2. Start container
./manage.sh start

# 3. Open shell
./manage.sh shell
```

#### **During Development**
```bash
# Inside container

# 4. Navigate to workspace
cd /root/ros2_ws/src

# 5. Create or edit packages
# (Files on host automatically sync)

# 6. Build packages
cb  # Build all
# or
cbs my_package  # Build specific package

# 7. Source workspace
cs

# 8. Test your code
ros2 launch my_package my_launch.py

# 9. Debug with visualization
rviz2
```

#### **Evening: Clean Up**
```bash
# 10. Exit container
exit

# 11. Stop container (optional)
./manage.sh stop
```

### **Creating a New ROS2 Package**

```bash
# Inside container
cd /root/ros2_ws/src

# Create Python package
ros2 pkg create --build-type ament_python my_robot_pkg \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs

# Create C++ package
ros2 pkg create --build-type ament_cmake my_robot_cpp \
    --dependencies rclcpp std_msgs sensor_msgs

# Build
cd /root/ros2_ws
colcon build --packages-select my_robot_pkg
source install/setup.bash

# Test
ros2 run my_robot_pkg my_node
```

### **Working with Computer Vision**

```bash
# Example: Object detection with YOLO
python3 << EOF
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolo11n.pt')

# Run inference
results = model('image.jpg')

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
        print(f"Detected: {model.names[int(cls)]} ({conf:.2f})")
EOF
```

### **Multi-Sensor Fusion Example**

```bash
# Example ROS2 node for sensor fusion
cat << 'EOF' > /root/ros2_ws/src/sensor_fusion_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10)
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        
        self.bridge = CvBridge()
        self.get_logger().info('Sensor Fusion Node Started')
    
    def camera_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # Process camera data
        
    def lidar_callback(self, msg):
        # Process LiDAR data
        ranges = msg.ranges
        
def main():
    rclpy.init()
    node = SensorFusionNode()
    rclpy.spin(node)
    
if __name__ == '__main__':
    main()
EOF

# Run the node
python3 sensor_fusion_node.py
```

### **Using Jupyter Notebooks**

```bash
# Inside container
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser

# On host, open browser to:
# http://localhost:8888
```

### **Git Workflow Inside Container**

```bash
# Inside container, workspace is shared with host
cd /root/ros2_ws/src/my_package

# Configure git (first time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Standard git workflow
git add .
git commit -m "feat: Add obstacle avoidance"
git push

# Changes are immediately visible on host
```

---

## ✅ Testing & Verification

### **Quick Health Check**

```bash
./test.sh
```

Runs 12 automated tests covering all major components.

### **Manual Testing**

#### **Test 1: ROS2 Installation**
```bash
./manage.sh shell

# Check ROS2 version
ros2 --version

# List installed packages
ros2 pkg list | wc -l  # Should show 350+

# Test publisher/subscriber
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_cpp listener

# Kill processes
killall talker listener
```

#### **Test 2: GPU & CUDA**
```bash
# Inside container

# Test 1: nvidia-smi
nvidia-smi

# Test 2: CUDA version
nvcc --version

# Test 3: CUDA sample
cat << 'EOF' > test_cuda.cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU!\n");
}

int main() {
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF

nvcc test_cuda.cu -o test_cuda
./test_cuda
```

#### **Test 3: PyTorch**
```bash
python3 << 'EOF'
import torch

print("=" * 50)
print("PyTorch Configuration")
print("=" * 50)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")
print(f"GPU Count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"\nGPU 0: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test tensor on GPU
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = x @ y
    print(f"\nMatrix multiplication successful!")
    print(f"Result shape: {z.shape}")
else:
    print("\nWARNING: CUDA not available!")
EOF
```

#### **Test 4: YOLO Detection**
```bash
python3 << 'EOF'
from ultralytics import YOLO
import torch

print("Loading YOLO model...")
model = YOLO('yolo11n.pt')  # Downloads if not present

print(f"Model loaded on: {model.device}")
print(f"Classes: {len(model.names)}")

# Test inference on dummy image
import numpy as np
dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

print("\nRunning inference...")
results = model(dummy_image)

print(f"Inference successful!")
print(f"Detected {len(results[0].boxes)} objects")
EOF
```

#### **Test 5: OpenCV**
```bash
python3 << 'EOF'
import cv2
import numpy as np

print(f"OpenCV Version: {cv2.__version__}")
print(f"Build Info:")
print(cv2.getBuildInformation())

# Test CUDA modules
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print(f"\nCUDA Devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
else:
    print("\nOpenCV built without CUDA (using CPU version)")

# Test basic operations
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

print(f"\nImage processing successful!")
print(f"Original shape: {img.shape}")
print(f"Gray shape: {gray.shape}")
EOF
```

#### **Test 6: ROS2 + Computer Vision Integration**
```bash
# Test camera simulation
ros2 run image_tools cam2image &

# View in RViz
rviz2 &

# In RViz, add Image display, set topic to /image

# Kill processes
killall cam2image rviz2
```

#### **Test 7: Navigation2**
```bash
# Test navigation stack components
ros2 launch nav2_bringup tb3_simulation_launch.py headless:=False

# This launches full navigation stack with Gazebo simulation
# You should see TurtleBot3 in Gazebo and RViz
```

### **Performance Benchmarks**

#### **GPU Performance**
```bash
python3 << 'EOF'
import torch
import time

# Matrix multiplication benchmark
sizes = [1000, 2000, 4000, 8000]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Running on: {device}")
print("\nMatrix Multiplication Benchmark:")
print("Size\t\tTime (ms)")
print("-" * 30)

for size in sizes:
    x = torch.rand(size, size, device=device)
    y = torch.rand(size, size, device=device)
    
    # Warm up
    _ = x @ y
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    z = x @ y
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    
    print(f"{size}x{size}\t\t{(end-start)*1000:.2f}")
EOF
```

#### **YOLO Inference Speed**
```bash
python3 << 'EOF'
from ultralytics import YOLO
import numpy as np
import time

model = YOLO('yolo11n.pt')
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warm up
for _ in range(10):
    _ = model(dummy_image, verbose=False)

# Benchmark
iterations = 100
start = time.time()
for _ in range(iterations):
    _ = model(dummy_image, verbose=False)
end = time.time()

fps = iterations / (end - start)
print(f"YOLO Inference Speed: {fps:.1f} FPS")
print(f"Latency: {1000/fps:.1f} ms")
EOF
```

---

## 🔧 Hardware Support

### **Cameras**

#### **Intel RealSense**
```bash
# Check connected cameras
rs-enumerate-devices

# View camera stream
realsense-viewer

# ROS2 launch
ros2 launch realsense2_camera rs_launch.py

# Access in Python
python3 << 'EOF'
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

print(f"Captured depth: {depth_frame.get_width()}x{depth_frame.get_height()}")
print(f"Captured color: {color_frame.get_width()}x{color_frame.get_height()}")

pipeline.stop()
EOF
```

#### **Stereolabs ZED**
```bash
# Check ZED camera
/usr/local/zed/tools/ZED_Explorer

# ROS2 launch
ros2 launch zed_wrapper zed_camera.launch.py

# Access via ROS2 topics
ros2 topic list | grep zed
```

#### **USB Cameras**
```bash
# List cameras
v4l2-ctl --list-devices

# Test camera
v4l2-ctl --device=/dev/video0 --all

# Capture with OpenCV
python3 << 'EOF'
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    print(f"Captured: {frame.shape}")
    cv2.imwrite('test.jpg', frame)
else:
    print("Failed to capture")

cap.release()
EOF
```

### **LiDAR**

#### **RPLidar**
```bash
# Give permissions (on host)
sudo chmod 666 /dev/ttyUSB0

# ROS2 launch
ros2 launch rplidar_ros rplidar_a1_launch.py

# View in RViz
ros2 run rviz2 rviz2
# Add LaserScan display, topic: /scan
```

### **CAN Bus**

```bash
# Setup CAN interface (on host)
sudo ip link set can0 type can bitrate 500000
sudo ip link set up can0

# Inside container
python3 << 'EOF'
import can

bus = can.interface.Bus(channel='can0', bustype='socketcan')

# Send message
msg = can.Message(
    arbitration_id=0x123,
    data=[0x11, 0x22, 0x33, 0x44],
    is_extended_id=False
)
bus.send(msg)

# Receive messages
for msg in bus:
    print(f"ID: 0x{msg.arbitration_id:X}, Data: {msg.data.hex()}")
    break

bus.shutdown()
EOF
```

### **Serial Devices**

```bash
# Give permissions (on host)
sudo chmod 666 /dev/ttyUSB0

# Inside container
python3 << 'EOF'
import serial

ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

# Write
ser.write(b'Hello\n')

# Read
data = ser.readline()
print(f"Received: {data}")

ser.close()
EOF
```

---

## 🐳 Docker Compose Setup

For multi-container applications with Redis and LiveKit:

### **Launch with Docker Compose**

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Restart specific service
docker-compose restart redis
```

### **docker-compose.yml Structure**

```yaml
services:
  robotics-dev:
    # Main development container
    
  redis:
    # State management & caching
    
  livekit:
    # Voice assistant server
```

### **Access Services**

```bash
# Inside robotics-dev container
python3 << 'EOF'
import redis

# Connect to Redis (container name as hostname)
r = redis.Redis(host='redis', port=6379, decode_responses=True)
r.set('test_key', 'Hello from container!')
value = r.get('test_key')
print(value)
EOF
```

---

## ⚙️ Advanced Configuration

### **Custom Environment Variables**

Create `.env` file:
```bash
cat << 'EOF' > .env
# ROS2 Configuration
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Custom paths
MY_ROBOT_CONFIG=/root/config
MY_DATA_PATH=/root/data

# API Keys (never commit to git!)
OPENAI_API_KEY=your_key_here
EOF

# Load in container
source .env
```

### **Persistent Data Volumes**

```bash
# Create named volume
docker volume create robot_data

# Use in run.sh
docker run ... \
    -v robot_data:/root/persistent_data \
    ...
```

### **Custom CUDA Memory Allocation**

```bash
# Set memory fraction (75%)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TF_FORCE_GPU_ALLOW_GROWTH=true

# In Python
import torch
torch.cuda.set_per_process_memory_fraction(0.75, 0)
```

### **ROS2 QoS Settings**

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Custom QoS for sensors
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# Use in subscription
self.subscription = self.create_subscription(
    Image,
    '/camera/image_raw',
    self.callback,
    sensor_qos
)
```

### **Multi-GPU Support**

```bash
# Use specific GPU
docker run --gpus '"device=0"' ...

# Use multiple GPUs
docker run --gpus '"device=0,1"' ...

# In Python
import torch

# Set default GPU
torch.cuda.set_device(0)

# Use specific GPU
x = torch.rand(100, 100).cuda(0)  # GPU 0
y = torch.rand(100, 100).cuda(1)  # GPU 1
```

---

## 🐛 Troubleshooting

### **Common Issues & Solutions**

#### **Issue 1: Docker daemon not running**
```
Error: Cannot connect to the Docker daemon
```

**Solution:**
```bash
# Start Docker
sudo systemctl start docker

# Enable at boot
sudo systemctl enable docker

# Check status
sudo systemctl status docker
```

#### **Issue 2: Permission denied**
```
Error: permission denied while trying to connect to Docker daemon
```

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Re-login or use
newgrp docker

# Verify
docker run hello-world
```

#### **Issue 3: NVIDIA runtime not found**
```
Error: could not select device driver "" with capabilities: [[gpu]]
```

**Solution:**
```bash
# Reinstall NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

#### **Issue 4: Build fails - disk space**
```
Error: no space left on device
```

**Solution:**
```bash
# Check space
df -h

# Clean Docker
docker system prune -a --volumes

# Remove unused images
docker image prune -a

# Check again
df -h
```

#### **Issue 5: Build fails - memory**
```
Error: Killed (exit code 137)
```

**Solution:**
```bash
# Increase Docker memory limit
# Edit /etc/docker/daemon.json
{
  "default-ulimits": {
    "memlock": {
      "Hard": -1,
      "Soft": -1
    }
  }
}

# Restart Docker
sudo systemctl restart docker

# Or close other applications and retry
```

#### **Issue 6: GPU not visible in container**
```
CUDA error: no CUDA-capable device
```

**Solution:**
```bash
# Check on host
nvidia-smi

# Check Docker can see GPU
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Verify runtime
docker info | grep -i runtime

# Should show: nvidia

# If not, reconfigure
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### **Issue 7: ROS2 nodes can't communicate**
```
No topics found
```

**Solution:**
```bash
# Check ROS_DOMAIN_ID matches
echo $ROS_DOMAIN_ID

# Check DDS discovery
ros2 daemon stop
ros2 daemon start

# Check firewall
sudo ufw status

# If active, allow ROS2 ports
sudo ufw allow 7400:7500/udp

# Test with simple nodes
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_cpp listener
```

#### **Issue 8: GUI applications don't work**
```
Error: cannot open display
```

**Solution:**
```bash
# On host, allow X11
xhost +local:docker

# Verify DISPLAY variable
echo $DISPLAY

# Check in container
echo $DISPLAY
glxinfo | grep "OpenGL"

# If still fails, use
docker run -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    ...
```

#### **Issue 9: Camera not detected**
```
No device found
```

**Solution:**
```bash
# Check on host
ls -l /dev/video*

# Give permissions
sudo chmod 666 /dev/video0

# Verify in container
ls -l /dev/video*
v4l2-ctl --list-devices

# For RealSense
rs-enumerate-devices
```

#### **Issue 10: Build is very slow**
```
Taking hours to build
```

**Solution:**
```bash
# Use local mirror (if available)
# Edit Dockerfile, change:
# FROM nvidia/cuda:12.6.0-devel-ubuntu22.04
# Add after FROM:
RUN sed -i 's/archive.ubuntu.com/YOUR_LOCAL_MIRROR/g' /etc/apt/sources.list

# Or use --network=host for faster downloads
docker build --network=host -t robotics-aiml-dev:latest .

# Or download base image first
docker pull nvidia/cuda:12.6.0-devel-ubuntu22.04
```

### **Performance Issues**

#### **Slow inference**
```bash
# Check GPU utilization
nvidia-smi dmon

# Enable TensorRT optimization
# In Python:
model = YOLO('yolo11n.pt')
model.export(format='engine')  # Creates TensorRT engine
model = YOLO('yolo11n.engine')  # Use optimized model
```

#### **Out of memory**
```bash
# Check memory usage
nvidia-smi

# Reduce batch size
# In code, use smaller batches

# Clear cache
import torch
torch.cuda.empty_cache()

# Monitor memory
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

---

## 🚀 Performance Optimization

### **Docker Build Optimization**

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Build with multiple jobs
docker build --build-arg MAKEFLAGS="-j$(nproc)" -t robotics-aiml-dev:latest .

# Use layer caching
docker build --cache-from robotics-aiml-dev:latest -t robotics-aiml-dev:latest .
```

### **ROS2 Build Optimization**

```bash
# Parallel build
colcon build --parallel-workers $(nproc)

# Build with compiler optimization
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Build specific packages only
colcon build --packages-select my_package

# Skip tests
colcon build --cmake-args -DBUILD_TESTING=OFF
```

### **GPU Utilization**

```bash
# Monitor GPU
watch -n1 nvidia-smi

# Enable TF32 for Ampere GPUs
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### **Memory Optimization**

```bash
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Clear unused variables
del variable
torch.cuda.empty_cache()

# Use lighter data types
# fp16 instead of fp32
model.half()
```

---

## 📖 Best Practices

### **Development Best Practices**

1. **Always source workspace** after building:
   ```bash
   colcon build
   source install/setup.bash
   ```

2. **Use symlink install** for faster Python iteration:
   ```bash
   colcon build --symlink-install
   ```

3. **Test in simulation** before deploying to robot

4. **Use version control** (git) for all code

5. **Write launch files** instead of manual commands

6. **Document your packages** with README files

7. **Use rqt tools** for debugging

8. **Profile your code** before optimizing

### **Docker Best Practices**

1. **Don't run as root** in production (use USER in Dockerfile)

2. **Use .dockerignore** to reduce build context

3. **Minimize layer count** by combining RUN commands

4. **Clean up in same layer** (apt clean, rm cache)

5. **Pin versions** for reproducibility

6. **Use multi-stage builds** for smaller images (if needed)

7. **Tag images** properly (v1.0.0, not just latest)

8. **Document Dockerfile** with comments

### **ROS2 Best Practices**

1. **Use composition** instead of nodelets

2. **Implement lifecycle nodes** for production

3. **Use QoS profiles** appropriately

4. **Handle parameters** properly

5. **Write integration tests**

6. **Profile with ros2 tools**:
   ```bash
   ros2 run rqt_graph rqt_graph
   ros2 topic hz /topic
   ros2 topic bw /topic
   ```

7. **Use rosbag2** for recording:
   ```bash
   ros2 bag record -a
   ```

---

## 📚 Examples & Tutorials

### **Example 1: Simple Object Detection Node**

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Initialize YOLO
        self.model = YOLO('yolo11n.pt')
        self.bridge = CvBridge()
        
        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        
        # Publisher
        self.publisher = self.create_publisher(
            Image,
            '/detections/image',
            10)
        
        self.get_logger().info('Object Detection Node Started')
    
    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Run YOLO detection
        results = self.model(cv_image, verbose=False)
        
        # Draw bounding boxes
        annotated = results[0].plot()
        
        # Convert back to ROS Image
        output_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        output_msg.header = msg.header
        
        # Publish
        self.publisher.publish(output_msg)
        
        # Log detections
        boxes = results[0].boxes
        if len(boxes) > 0:
            self.get_logger().info(f'Detected {len(boxes)} objects')

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Usage:**
```bash
# Copy to workspace
cp object_detection_node.py /root/ros2_ws/src/my_package/my_package/

# Make executable
chmod +x /root/ros2_ws/src/my_package/my_package/object_detection_node.py

# Update setup.py to include the script

# Build
colcon build --packages-select my_package

# Run
ros2 run my_package object_detection_node
```

### **Example 2: Navigation with Obstacle Avoidance**

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav2_simple_commander.robot_navigator import BasicNavigator
import numpy as np

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        
        # Initialize navigator
        self.navigator = BasicNavigator()
        
        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        
        # Subscriber for laser scan
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        
        self.obstacle_detected = False
        self.min_distance = 0.5  # meters
        
        self.get_logger().info('Navigation Node Started')
    
    def scan_callback(self, msg):
        # Check for obstacles
        ranges = np.array(msg.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]
        
        if len(valid_ranges) > 0:
            min_range = np.min(valid_ranges)
            self.obstacle_detected = min_range < self.min_distance
            
            if self.obstacle_detected:
                self.stop_robot()
    
    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().warn('Obstacle detected! Stopping.')
    
    def navigate_to_pose(self, x, y, theta):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.navigator.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = np.sin(theta / 2.0)
        goal.pose.orientation.w = np.cos(theta / 2.0)
        
        self.navigator.goToPose(goal)
        
        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                self.get_logger().info(f'Distance remaining: {feedback.distance_remaining:.2f}')
            rclpy.spin_once(self, timeout_sec=0.1)

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    
    # Navigate to goal (x=2.0, y=1.0, theta=0.0)
    node.navigate_to_pose(2.0, 1.0, 0.0)
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### **Example 3: Multi-Camera Fusion**

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraFusionNode(Node):
    def __init__(self):
        super().__init__('camera_fusion_node')
        
        self.bridge = CvBridge()
        
        # Multiple camera subscribers
        self.front_sub = self.create_subscription(
            Image, '/camera/front/image_raw',
            lambda msg: self.camera_callback(msg, 'front'), 10)
        
        self.left_sub = self.create_subscription(
            Image, '/camera/left/image_raw',
            lambda msg: self.camera_callback(msg, 'left'), 10)
        
        self.right_sub = self.create_subscription(
            Image, '/camera/right/image_raw',
            lambda msg: self.camera_callback(msg, 'right'), 10)
        
        # Store latest images
        self.images = {
            'front': None,
            'left': None,
            'right': None
        }
        
        # Publisher for fused view
        self.fused_pub = self.create_publisher(
            Image, '/camera/fused', 10)
        
        # Timer for fusion
        self.timer = self.create_timer(0.1, self.fuse_images)
        
        self.get_logger().info('Camera Fusion Node Started')
    
    def camera_callback(self, msg, camera_name):
        self.images[camera_name] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    
    def fuse_images(self):
        if all(img is not None for img in self.images.values()):
            # Resize all images to same height
            h = 480
            resized = {}
            for name, img in self.images.items():
                aspect = img.shape[1] / img.shape[0]
                w = int(h * aspect)
                resized[name] = cv2.resize(img, (w, h))
            
            # Concatenate horizontally
            fused = np.hstack([
                resized['left'],
                resized['front'],
                resized['right']
            ])
            
            # Add labels
            cv2.putText(fused, 'LEFT', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(fused, 'FRONT', (w + 50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(fused, 'RIGHT', (2*w + 50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Publish
            msg = self.bridge.cv2_to_imgmsg(fused, 'bgr8')
            self.fused_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### **How to Contribute**

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Test thoroughly**
5. **Commit**: `git commit -m 'feat: Add amazing feature'`
6. **Push**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### **Contribution Guidelines**

- Follow existing code style
- Add tests for new features
- Update documentation
- Keep commits atomic and well-described
- Use conventional commit messages

### **Reporting Issues**

When reporting issues, please include:
- Docker version
- Host OS and version
- GPU model
- Error messages
- Steps to reproduce
- Expected vs actual behavior

---

## ❓ FAQ

### **General Questions**

**Q: Can I use this on Windows?**  
A: Yes, with WSL2 + Docker Desktop + NVIDIA Container Toolkit for Windows.

**Q: Do I need an NVIDIA GPU?**  
A: For AI/ML features, yes. ROS2 works without GPU but you'll miss CUDA acceleration.

**Q: Can I use AMD GPUs?**  
A: Not currently. This environment is specifically built for NVIDIA CUDA.

**Q: How much disk space do I need?**  
A: Minimum 50 GB, recommended 100 GB.

**Q: Can I deploy this to Jetson?**  
A: Yes! Build on Jetson using the same Dockerfile. It's tested on Jetson AGX Orin.

### **Technical Questions**

**Q: How do I add new packages?**  
A: 
```bash
# Temporary (lost on container removal)
pip3 install package_name

# Permanent (add to Dockerfile)
RUN pip3 install --no-cache-dir --ignore-installed package_name
```

**Q: Can I use ROS1?**  
A: This environment is ROS2 Humble. For ROS1, you'd need a different base image.

**Q: How do I update packages?**  
A: Rebuild the image:
```bash
docker build --no-cache -t robotics-aiml-dev:latest .
```

**Q: Can I run multiple containers?**  
A: Yes! Use docker-compose or run with different names:
```bash
docker run --name robot1 ...
docker run --name robot2 ...
```

**Q: How do I backup my work?**  
A: Your workspace is on the host at `~/Desktop/ros2_humble_ws`. Just backup that directory.

**Q: Can I use this for production?**  
A: Yes, but consider:
- Remove development tools
- Use specific image tags (not :latest)
- Implement proper logging
- Add health checks
- Use docker-compose for orchestration

### **Troubleshooting FAQs**

**Q: Build fails with "no space left"**  
A: Clean Docker: `docker system prune -a`

**Q: GPU not detected**  
A: Verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`

**Q: ROS2 nodes can't see each other**  
A: Check `ROS_DOMAIN_ID` matches on all systems

**Q: GUI apps don't work**  
A: Run on host: `xhost +local:docker`

---

## 📝 Changelog

### **v1.0.0** (February 2026)
- Initial release
- ROS2 Humble Desktop Full
- CUDA 12.6 support
- PyTorch 2.5.1
- YOLO v8/v11
- 700+ packages
- Complete documentation
- Management scripts
- Docker Compose configuration

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Lasantha Kulasooriya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

This project builds upon the incredible work of many open-source communities:

- **ROS2** by Open Robotics
- **PyTorch** by Meta AI Research
- **CUDA** by NVIDIA
- **Ultralytics YOLO** by Glenn Jocher
- **OpenCV** by Intel & community
- **Docker** by Docker Inc.
- **Navigation2** by Steve Macenski
- **And hundreds of other contributors**

Special thanks to the robotics and AI/ML communities for their continuous contributions to open-source software.

---

## 📞 Contact & Support

### **Developer**

**Lasantha Kulasooriya**  
Robotics & AI Engineer   
Sri Lanka 🇱🇰

### **Project Links**

- **GitHub Repository**: [github.com/LasaK97/robotics-ai-ml-docker-env](https://github.com/LasaK97/robotics-ai-ml-docker-env)
- **Documentation**: See this README
- **Issues**: [GitHub Issues](https://github.com/LasaK97/robotics-ai-ml-docker-env/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LasaK97/robotics-ai-ml-docker-env/discussions)

### **Get Help**

1. **Check documentation** - This README covers most use cases
2. **Search existing issues** - Someone may have solved your problem
3. **Open an issue** - Provide details, logs, and steps to reproduce
4. **Join discussions** - Ask questions, share experiences

### **Support the Project**

If this project helps you:
- ⭐ **Star** the repository on GitHub
- 🐛 **Report bugs** and suggest features
- 📝 **Improve documentation**
- 🤝 **Contribute code**
- 💬 **Share** with others

---

## 🌟 Star History

If you find this project useful, please give it a star! ⭐

---

**Built with ❤️ for the robotics community**

*Last updated: February 2026*