#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║               Manriix Robot Docker Builder                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo -e "${BLUE}         Developer: Lasantha Kulasooriya                     ${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check NVIDIA Container Toolkit
if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo -e "${RED}✗ NVIDIA Container Toolkit not working.${NC}"
    echo -e "${YELLOW}Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"
echo -e "${GREEN}✓ NVIDIA Container Toolkit is working${NC}"
echo ""

# Build parameters
IMAGE_NAME="manriix-ros2-humble"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${YELLOW}Build Configuration:${NC}"
echo -e "  Image Name: ${FULL_IMAGE_NAME}"
echo -e "  Dockerfile: $(pwd)/Dockerfile"
echo -e "  Context: $(pwd)"
echo ""

# Check disk space
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 30 ]; then
    echo -e "${RED}✗ Insufficient disk space. Need at least 30GB free.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Disk space available: ${AVAILABLE_SPACE}GB${NC}"
echo ""

# Ask for confirmation
read -p "$(echo -e ${YELLOW}Do you want to proceed with the build? [y/N]: ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Build cancelled.${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Starting Docker image build...${NC}"
echo -e "${YELLOW}This will take 30-60 minutes depending on internet speed.${NC}"
echo -e "${YELLOW}Downloading packages...${NC}"
echo ""

# Build the image with progress
docker build \
    -t "$FULL_IMAGE_NAME" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -f Dockerfile \
    . 2>&1 | tee build.log

# Check if build was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              Build Completed Successfully!                ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}✓ Image built: ${FULL_IMAGE_NAME}${NC}"
    
    # Get image size
    IMAGE_SIZE=$(docker images "$FULL_IMAGE_NAME" --format "{{.Size}}")
    echo -e "${GREEN}✓ Image Size: ${IMAGE_SIZE}${NC}"
    echo -e "${GREEN}✓ Build log saved: build.log${NC}"
    
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo -e "  1. Test GPU access:"
    echo -e "     ${GREEN}docker run --rm --gpus all ${FULL_IMAGE_NAME} nvidia-smi${NC}"
    echo ""
    echo -e "  2. Run the container:"
    echo -e "     ${GREEN}./run.sh${NC}"
    echo ""
    echo -e "  3. Verify everything:"
    echo -e "     ${GREEN}./test.sh${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                  Build Failed!                            ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}Check build.log for details${NC}"
    exit 1
fi