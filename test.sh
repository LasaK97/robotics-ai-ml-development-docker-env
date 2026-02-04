#!/bin/bash

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

IMAGE_NAME="manriix-ros2-humble:latest"
CONTAINER_NAME="manriix-dev"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Manriix Environment Verification Test            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo -e "${BLUE}         Developer: Lasantha Kulasooriya                     ${NC}"
echo ""

# Test 1: Image exists
echo -n "1. Checking Docker image... "
if docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}   Image not found. Run: ./build.sh${NC}"
    exit 1
fi

# Test 2: Container exists
echo -n "2. Checking container... "
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}   Container not found. Run: ./run.sh${NC}"
    exit 1
fi

# Test 3: Container running
echo -n "3. Checking container status... "
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${GREEN}✓ (running)${NC}"
else
    echo -e "${YELLOW}✗ (stopped, starting...)${NC}"
    docker start "$CONTAINER_NAME" > /dev/null 2>&1
    sleep 2
fi

# Test 4: GPU access
echo -n "4. Testing GPU access... "
if docker exec "$CONTAINER_NAME" nvidia-smi > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}   GPU not accessible${NC}"
    exit 1
fi

# Test 5: CUDA version
echo -n "5. Checking CUDA version... "
CUDA_VERSION=$(docker exec "$CONTAINER_NAME" nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -d',' -f1)
if [ -n "$CUDA_VERSION" ]; then
    echo -e "${GREEN}✓ ($CUDA_VERSION)${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 6: ROS2
echo -n "6. Testing ROS2 Humble... "
if docker exec "$CONTAINER_NAME" bash -c "source /opt/ros/humble/setup.bash && ros2 topic list" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 7: PyTorch
echo -n "7. Testing PyTorch... "
if docker exec "$CONTAINER_NAME" python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(docker exec "$CONTAINER_NAME" python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo -e "${GREEN}✓ ($TORCH_VERSION)${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 8: PyTorch CUDA
echo -n "8. Testing PyTorch CUDA... "
if docker exec "$CONTAINER_NAME" python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_COUNT=$(docker exec "$CONTAINER_NAME" python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    echo -e "${GREEN}✓ ($GPU_COUNT GPU)${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 9: YOLO/Ultralytics
echo -n "9. Testing YOLO (Ultralytics)... "
if docker exec "$CONTAINER_NAME" python3 -c "from ultralytics import YOLO" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 10: OpenCV
echo -n "10. Testing OpenCV... "
if docker exec "$CONTAINER_NAME" python3 -c "import cv2" 2>/dev/null; then
    CV_VERSION=$(docker exec "$CONTAINER_NAME" python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null)
    echo -e "${GREEN}✓ ($CV_VERSION)${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 11: Redis client
echo -n "11. Testing Redis client... "
if docker exec "$CONTAINER_NAME" python3 -c "import redis" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Test 12: Flask
echo -n "12. Testing Flask... "
if docker exec "$CONTAINER_NAME" python3 -c "import flask" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

exit 1
fi

# Test 13: CycloneDDS
echo -n "13. Testing CycloneDDS... "
if docker exec "$CONTAINER_NAME" bash -c "source /opt/ros/humble/setup.bash && ros2 pkg list | grep -q cyclonedds" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

echo ""

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           All Tests Passed Successfully! ✓                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Your Manriix development environment is ready!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Open shell: ${GREEN}./manage.sh shell${NC}"
echo -e "  2. Test RViz: ${GREEN}rviz2${NC} (inside container)"
echo -e "  3. Check GPU: ${GREEN}./manage.sh gpu${NC}"
echo ""