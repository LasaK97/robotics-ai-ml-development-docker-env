#!/bin/bash

set -e

CONTAINER_NAME="manriix-dev"
IMAGE_NAME="manriix-ros2-humble:latest"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_help() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║          Manriix Container Management Helper              ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${BLUE}           Developer: Lasantha Kulasooriya                     ${NC}"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start      - Start the container"
    echo "  stop       - Stop the container"
    echo "  restart    - Restart the container"
    echo "  shell      - Open a bash shell in the container"
    echo "  logs       - Show container logs (follow mode)"
    echo "  status     - Show container status"
    echo "  remove     - Remove the container (keeps image)"
    echo "  gpu        - Test GPU access"
    echo "  build      - Build ROS2 workspace inside container"
    echo "  clean      - Clean ROS2 build artifacts"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 status     # Check if container is running"
    echo "  $0 shell      # Open shell in container"
    echo "  $0 gpu        # Test GPU and PyTorch"
    echo "  $0 build      # Build ROS2 workspace"
    echo ""
}

check_exists() {
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${RED}✗ Container '${CONTAINER_NAME}' not found${NC}"
        echo -e "${YELLOW}Create it first: ./run.sh${NC}"
        exit 1
    fi
}

is_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

case "$1" in
    start)
        check_exists
        if is_running; then
            echo -e "${GREEN}✓ Container is already running${NC}"
        else
            echo -e "${YELLOW}Starting container...${NC}"
            docker start "$CONTAINER_NAME"
            echo -e "${GREEN}✓ Container started${NC}"
        fi
        ;;
        
    stop)
        check_exists
        if is_running; then
            echo -e "${YELLOW}Stopping container...${NC}"
            docker stop "$CONTAINER_NAME"
            echo -e "${GREEN}✓ Container stopped${NC}"
        else
            echo -e "${YELLOW}Container is not running${NC}"
        fi
        ;;
        
    restart)
        check_exists
        echo -e "${YELLOW}Restarting container...${NC}"
        docker restart "$CONTAINER_NAME"
        echo -e "${GREEN}✓ Container restarted${NC}"
        ;;
        
    shell)
        check_exists
        if ! is_running; then
            echo -e "${YELLOW}Starting container...${NC}"
            docker start "$CONTAINER_NAME"
        fi
        echo -e "${GREEN}Opening shell in container...${NC}"
        echo ""
        docker exec -it "$CONTAINER_NAME" /bin/bash
        ;;
        
    logs)
        check_exists
        echo -e "${BLUE}Showing container logs (Ctrl+C to exit)...${NC}"
        docker logs -f "$CONTAINER_NAME"
        ;;
        
    status)
        if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${BLUE}║              Manriix Container Status                     ║${NC}"
            echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            docker ps -a --filter "name=^${CONTAINER_NAME}$" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
            echo ""
            
            if is_running; then
                echo -e "${GREEN}✓ Container is running${NC}"
                echo ""
                echo "Available commands:"
                echo "  Open shell: ${GREEN}$0 shell${NC}"
                echo "  View logs: ${GREEN}$0 logs${NC}"
                echo "  Test GPU: ${GREEN}$0 gpu${NC}"
                echo "  Stop: ${GREEN}$0 stop${NC}"
            else
                echo -e "${YELLOW}○ Container is stopped${NC}"
                echo ""
                echo "Start it: ${GREEN}$0 start${NC}"
            fi
        else
            echo -e "${RED}✗ Container not found${NC}"
            echo -e "${YELLOW}Create it: ./run.sh${NC}"
        fi
        ;;
        
    remove)
        check_exists
        echo ""
        echo -e "${YELLOW}This will remove the container but keep the image.${NC}"
        echo -e "${YELLOW}Your workspace files will NOT be affected.${NC}"
        echo ""
        read -p "Are you sure you want to remove the container? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if is_running; then
                echo -e "${YELLOW}Stopping container...${NC}"
                docker stop "$CONTAINER_NAME"
            fi
            docker rm "$CONTAINER_NAME"
            echo -e "${GREEN}✓ Container removed${NC}"
            echo -e "${YELLOW}Image '${IMAGE_NAME}' is still available${NC}"
            echo -e "${YELLOW}Run './run.sh' to create a new container${NC}"
        else
            echo -e "${YELLOW}Cancelled${NC}"
        fi
        ;;
        
    gpu)
        check_exists
        if ! is_running; then
            echo -e "${YELLOW}Starting container...${NC}"
            docker start "$CONTAINER_NAME"
        fi
        echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║                    GPU Test Report                        ║${NC}"
        echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${BLUE}1. NVIDIA GPU Status:${NC}"
        docker exec "$CONTAINER_NAME" nvidia-smi
        echo ""
        echo -e "${BLUE}2. PyTorch CUDA Test:${NC}"
        docker exec "$CONTAINER_NAME" python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
        echo ""
        ;;
        
    build)
        check_exists
        if ! is_running; then
            echo -e "${YELLOW}Starting container...${NC}"
            docker start "$CONTAINER_NAME"
        fi
        echo -e "${BLUE}Building ROS2 workspace...${NC}"
        docker exec -it "$CONTAINER_NAME" /bin/bash -c "cd /root/ros2_ws && colcon build --symlink-install"
        echo -e "${GREEN}✓ Build complete${NC}"
        echo -e "${YELLOW}Don't forget to source: ${GREEN}source install/setup.bash${NC}"
        ;;
        
    clean)
        check_exists
        if ! is_running; then
            echo -e "${YELLOW}Starting container...${NC}"
            docker start "$CONTAINER_NAME"
        fi
        echo -e "${YELLOW}Cleaning build artifacts...${NC}"
        docker exec "$CONTAINER_NAME" /bin/bash -c "cd /root/ros2_ws && rm -rf build install log"
        echo -e "${GREEN}✓ Build artifacts cleaned${NC}"
        ;;
        
    help|--help|-h|"")
        show_help
        ;;
        
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac