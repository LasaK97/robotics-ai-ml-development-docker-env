#!/bin/bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Manriix  Development Container Launcher           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo -e "${BLUE}           Developer: Lasantha Kulasooriya                     ${NC}"
echo ""

# Configuration
IMAGE_NAME="manriix-ros2-humble:latest"
CONTAINER_NAME="manriix-dev"
WORKSPACE="$HOME/Desktop/ros2_humble_ws"

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo -e "${RED}✗ Image not found: ${IMAGE_NAME}${NC}"
    echo -e "${YELLOW}Please build the image first: ./build.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Image found: ${IMAGE_NAME}${NC}"

# Check workspace directory
if [ ! -d "$WORKSPACE" ]; then
    echo -e "${YELLOW}! Workspace directory not found: ${WORKSPACE}${NC}"
    echo -e "${YELLOW}  Creating workspace...${NC}"
    mkdir -p "$WORKSPACE/src"
fi

echo -e "${GREEN}✓ Workspace: ${WORKSPACE}${NC}"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}! Container '${CONTAINER_NAME}' already exists${NC}"
    
    # Check if running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${GREEN}✓ Container is already running. Attaching...${NC}"
        echo ""
        docker exec -it "$CONTAINER_NAME" /bin/bash
        exit 0
    else
        echo -e "${YELLOW}  Starting existing container...${NC}"
        docker start "$CONTAINER_NAME"
        echo -e "${GREEN}✓ Container started. Attaching...${NC}"
        echo ""
        docker exec -it "$CONTAINER_NAME" /bin/bash
        exit 0
    fi
fi

# Allow X11 connections for GUI apps (RViz, rqt)
xhost +local:docker > /dev/null 2>&1

echo -e "${GREEN}✓ X11 access enabled for GUI applications${NC}"
echo ""
echo -e "${BLUE}Launching new container '${CONTAINER_NAME}'...${NC}"
echo ""

# Run new container
docker run -it \
    --name "$CONTAINER_NAME" \
    --runtime=nvidia \
    --gpus all \
    --privileged \
    --network host \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e ROS_DOMAIN_ID=0 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -v "$WORKSPACE:/root/ros2_ws:rw" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev:/dev \
    -v /dev/shm:/dev/shm \
    "$IMAGE_NAME" \
    /bin/bash

echo ""
echo -e "${YELLOW}Container exited.${NC}"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo -e "  Restart container: ${GREEN}docker start ${CONTAINER_NAME}${NC}"
echo -e "  Attach to container: ${GREEN}docker exec -it ${CONTAINER_NAME} bash${NC}"
echo -e "  Or use helper: ${GREEN}./manage.sh shell${NC}"
echo -e "  Remove container: ${GREEN}docker rm ${CONTAINER_NAME}${NC}"
echo ""