
# ROS2 Humble Docker Image
# Robotics + AI/ML Development 

# Base Image: Ubuntu 22.04 with CUDA 12.6 pre-installed
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Colombo

# Set CUDA env variables
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

# Support both laptop (RTX 4060 = 8.9) and Jetson (AGX Orin = 8.7)
ENV CUDA_ARCH_LIST="8.7;8.9"
ENV TORCH_CUDA_ARCH_LIST="8.7;8.9"

# ROS2 env variables
ENV ROS_VERSION=2
ENV ROS_PYTHON_VERSION=3
ENV ROS_DISTRO=humble
ENV ROS_LOCALHOST_ONLY=0


# System dependencies and Tools
RUN apt-get update && apt-get install -y \
    locales \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    software-properties-common \
    build-essential \
    cmake \
    git \
    vim \
    nano \
    python3-pip \
    python3-dev \
    python3-venv \
    ca-certificates \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Add ROS2 Humble repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble Desktop Full + Robotics packages
RUN apt-get update && apt-get install -y \
    # Core ROS2 packages
    ros-humble-desktop \
    ros-humble-ros-base \
    ros-humble-ros-core \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep \
    \
    # CycloneDDS for real-time performance
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-cyclonedds \
    \
    # Navigation2 complete stack
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-nav2-msgs \
    ros-humble-nav2-simple-commander \
    \
    # SLAM and Localization
    ros-humble-slam-toolbox \
    ros-humble-robot-localization \
    \
    # Control packages
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-diff-drive-controller \
    ros-humble-ackermann-steering-controller \
    ros-humble-mecanum-drive-controller \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    \
    # Camera and sensor drivers
    ros-humble-realsense2-camera \
    ros-humble-realsense2-camera-msgs \
    ros-humble-librealsense2 \
    ros-humble-zed-msgs \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-compressed-image-transport \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    \
    # LiDAR
    ros-humble-rplidar-ros \
    ros-humble-laser-geometry \
    ros-humble-laser-filters \
    \
    # TF and transforms
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-tf2-sensor-msgs \
    ros-humble-tf-transformations \
    \
    # Visualization
    ros-humble-rviz2 \
    ros-humble-rviz-common \
    ros-humble-rviz-default-plugins \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-rqt-graph \
    ros-humble-rqt-image-view \
    ros-humble-rqt-plot \
    ros-humble-rqt-tf-tree \
    \
    # ROS Bridge for web integration
    ros-humble-rosbridge-server \
    ros-humble-rosbridge-library \
    ros-humble-rosapi \
    \
    # Teleop
    ros-humble-teleop-twist-keyboard \
    ros-humble-teleop-twist-joy \
    ros-humble-joy \
    \
    # Additional utilities
    ros-humble-diagnostic-updater \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    && rm -rf /var/lib/apt/lists/*

# Install additional system libraries
RUN apt-get update && apt-get install -y \
    libudev-dev \
    libusb-1.0-0-dev \
    libv4l-dev \
    v4l-utils \
    udev \
    can-utils \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep (ROS dependency manager)
RUN rosdep init || true \
    && rosdep update

# Upgrade pip and install Python build tools
RUN python3 -m pip install --upgrade pip setuptools\<80 wheel

# Install PyTorch
RUN pip3 install --no-cache-dir --ignore-installed \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Core ML/AI frameworks
RUN pip3 install --no-cache-dir \
    # Deep Learning frameworks
    onnx==1.19.1 \
    onnxruntime-gpu==1.23.2 \
    onnxslim==0.1.82 \
    \
    # HuggingFace ecosystem
    transformers \
    accelerate \
    datasets \
    tokenizers \
    huggingface-hub==0.36.0 \
    safetensors==0.7.0 \
    sentencepiece==0.2.1 \
    \
    # TensorRT Python bindings
    nvidia-tensorrt

# Computer Vision packages
RUN pip3 install --no-cache-dir numpy==1.26.4

RUN pip3 install --no-cache-dir \
    opencv-python \
    opencv-contrib-python \
    pillow==12.0.0

RUN pip3 install --no-cache-dir \
    ultralytics==8.3.227 \
    ultralytics-thop==2.0.18 \
    supervision==0.27.0


RUN pip3 install --no-cache-dir \
    deep-sort-realtime==1.3.2 \
    mediapipe==0.10.18

# Scientific Computing packages
RUN pip3 install --no-cache-dir \
    # Core scientific libraries
    numpy==1.26.4 \
    scipy==1.15.3 \
    pandas==2.0.3 \
    polars==1.35.2 \
    \
    # Machine Learning
    scikit-learn==1.7.2 \
    \
    # GPU acceleration
    cupy-cuda12x==13.6.0 \
    numba==0.62.1 \
    \
    # Visualization
    matplotlib==3.10.7 \
    seaborn==0.13.2

# Redis and communication libraries
RUN pip3 install --no-cache-dir \
    # Redis
    redis \
    hiredis \
    \
    # MQTT
    paho-mqtt==2.1.0 \
    \
    # CAN bus
    python-can==3.3.2 \
    \
    # Serial
    pyserial==3.5 \
    \
    # HTTP
    requests==2.32.5 \
    requests-toolbelt==0.9.1

# Web frameworks 
RUN pip3 install --no-cache-dir --ignore-installed \
    Flask==3.1.2 \
    flask-cors==6.0.2 \
    Flask-Login==0.6.3 \
    Werkzeug==3.1.3

# Development and utility tools
RUN pip3 install --no-cache-dir --ignore-installed \
    # Development
    jupyter \
    ipython \
    black==25.9.0 \
    flake8==7.3.0 \
    pylint==4.0.4 \
    pytest==6.2.5 \
    pytest-cov==3.0.0 \
    \
    # Data processing
    tqdm==4.67.1 \
    tabulate==0.9.0 \
    colorama==0.4.4 \
    coloredlogs==15.0.1 \
    \
    # Configuration
    PyYAML==6.0.3 \
    python-dotenv==1.2.1 \
    omegaconf==2.3.0 \
    pydantic==2.12.4 \
    pydantic-settings==2.12.0 \
    \
    # Math and transforms
    transforms3d==0.4.2 \
    sympy==1.14.0

# Workspace directory structure
RUN mkdir -p /root/ros2_ws/src
WORKDIR /root/ros2_ws

# Setup ROS2 environment in bashrc
RUN echo "# ROS2 Humble Setup" >> /root/.bashrc \
    && echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "if [ -f /root/ros2_ws/install/setup.bash ]; then" >> /root/.bashrc \
    && echo "    source /root/ros2_ws/install/setup.bash" >> /root/.bashrc \
    && echo "fi" >> /root/.bashrc \
    && echo "" >> /root/.bashrc \
    && echo "# CUDA Environment" >> /root/.bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda-12.6" >> /root/.bashrc \
    && echo "export PATH=\${CUDA_HOME}/bin:\${PATH}" >> /root/.bashrc \
    && echo "export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:/usr/local/lib:\${LD_LIBRARY_PATH}" >> /root/.bashrc \
    && echo "" >> /root/.bashrc \
    && echo "# ROS2 Environment Variables" >> /root/.bashrc \
    && echo "export ROS_DOMAIN_ID=0" >> /root/.bashrc \
    && echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc \
    && echo "" >> /root/.bashrc \
    && echo "# Convenience aliases" >> /root/.bashrc \
    && echo "alias cb='cd /root/ros2_ws && colcon build --symlink-install'" >> /root/.bashrc \
    && echo "alias cbs='cd /root/ros2_ws && colcon build --symlink-install --packages-select'" >> /root/.bashrc \
    && echo "alias cs='source /root/ros2_ws/install/setup.bash'" >> /root/.bashrc

# Create a startup script for easy container launching
RUN echo '#!/bin/bash' > /root/startup.sh \
    && echo 'source /opt/ros/humble/setup.bash' >> /root/startup.sh \
    && echo 'if [ -f /root/ros2_ws/install/setup.bash ]; then' >> /root/startup.sh \
    && echo '    source /root/ros2_ws/install/setup.bash' >> /root/startup.sh \
    && echo 'fi' >> /root/startup.sh \
    && echo 'exec "$@"' >> /root/startup.sh \
    && chmod +x /root/startup.sh

# Set default command
CMD ["/bin/bash"]

# Build information
LABEL maintainer="Hype Invention - Lasantha Kulsaooriya"
LABEL description="Manriix Autonomous Photography Robot"
LABEL version="1.0"
LABEL cuda.version="12.6"
LABEL ros.distro="humble"