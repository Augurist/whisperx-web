#!/bin/bash

# Load environment variables from .env if it exists
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a  # turn off automatic export
    echo -e "${BLUE}ℹ️  Environment loaded from .env${NC}"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================${NC}"
echo -e "${BLUE}   WhisperX Docker Manager${NC}"
echo -e "${BLUE}==================================${NC}"

# Function to check for PyTorch updates
check_pytorch_updates() {
    echo -e "\n${CYAN}🔍 Checking for PyTorch image updates...${NC}"
    
    # Get current image from Dockerfile
    CURRENT_IMAGE=$(grep "^FROM pytorch/pytorch:" Dockerfile 2>/dev/null | cut -d' ' -f2)
    if [ -z "$CURRENT_IMAGE" ]; then
        echo -e "${YELLOW}⚠️  Could not determine current image${NC}"
        return
    fi
    
    echo -e "Current image: ${YELLOW}$CURRENT_IMAGE${NC}"
    
    # Check latest available (requires curl and python3)
    if command -v curl &> /dev/null && command -v python3 &> /dev/null; then
        LATEST=$(curl -s "https://hub.docker.com/v2/repositories/pytorch/pytorch/tags/?page_size=10&ordering=-last_updated" 2>/dev/null | \
          python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for tag in data['results']:
        name = tag['name']
        if 'cuda12' in name and 'runtime' in name and 'rc' not in name:
            print(f\"pytorch/pytorch:{name}\")
            break
except:
    pass
" 2>/dev/null)
        
        if [ -n "$LATEST" ] && [ "$LATEST" != "$CURRENT_IMAGE" ]; then
            echo -e "${YELLOW}⚠️  Update available: ${GREEN}$LATEST${NC}"
            echo -e "   To update: Edit Dockerfile and change FROM line"
        elif [ -n "$LATEST" ]; then
            echo -e "${GREEN}✅ You're using the latest image!${NC}"
        fi
    fi
}

# Function to check GPU availability
check_gpu() {
    echo -e "\n${CYAN}🖥️  Checking GPU availability...${NC}"
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$GPU_INFO" ]; then
            echo -e "${GREEN}✅ GPU detected: $GPU_INFO${NC}"
        else
            echo -e "${RED}❌ No GPU detected${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  nvidia-smi not found${NC}"
    fi
}

# Function to check Docker status
check_docker_status() {
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ Docker is not running!${NC}"
        echo -e "Please start Docker and try again."
        exit 1
    fi
}

# Function to stop all containers properly
stop_containers() {
    echo -e "\n${YELLOW}🛑 Stopping existing containers...${NC}"
    
    # Check if any containers are running
    if docker compose ps --services --filter "status=running" 2>/dev/null | grep -q .; then
        docker compose down --remove-orphans
        echo -e "${GREEN}✅ Containers stopped and removed${NC}"
        
        # Wait a moment for ports to be released
        sleep 2
    else
        echo -e "${BLUE}ℹ️  No containers were running${NC}"
    fi
}

# Function to check for port conflicts
check_ports() {
    echo -e "\n${CYAN}🔌 Checking for port conflicts...${NC}"
    
    # Check if port 5000 is in use
    if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Port 5000 is already in use!${NC}"
        echo -n "Do you want to stop the process using it? [y/N]: "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            # Try to stop the process
            PID=$(lsof -Pi :5000 -sTCP:LISTEN -t)
            kill $PID 2>/dev/null && echo -e "${GREEN}✅ Process stopped${NC}"
        fi
    else
        echo -e "${GREEN}✅ Port 5000 is available${NC}"
    fi
}

# Function to clean up old images
cleanup_images() {
    echo -e "\n${CYAN}🧹 Cleaning up Docker resources...${NC}"
    
    # Remove dangling images
    DANGLING=$(docker images -f "dangling=true" -q | wc -l)
    if [ "$DANGLING" -gt 0 ]; then
        docker image prune -f > /dev/null 2>&1
        echo -e "${GREEN}✅ Removed $DANGLING dangling images${NC}"
    fi
    
    # Show disk usage
    DISK_USAGE=$(docker system df --format "table {{.Type}}\t{{.Size}}" | grep Images | awk '{print $2}')
    echo -e "${BLUE}ℹ️  Docker images using: $DISK_USAGE${NC}"
}

# Function to check environment variables
check_env() {
    echo -e "\n${CYAN}🔐 Checking environment variables...${NC}"
    
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${YELLOW}⚠️  HF_TOKEN not set - speaker diarization will be disabled${NC}"
        echo -e "   To enable: export HF_TOKEN='your_token_here'"
    else
        echo -e "${GREEN}✅ HF_TOKEN is configured${NC}"
    fi
}

# Main script starts here
echo ""

# Run all checks
check_docker_status
check_gpu
check_pytorch_updates
check_env

# Check if containers are currently running
RUNNING_CONTAINERS=$(docker compose ps --services --filter "status=running" 2>/dev/null)
if [ -n "$RUNNING_CONTAINERS" ]; then
    echo -e "\n${YELLOW}⚠️  The following containers are currently running:${NC}"
    echo "$RUNNING_CONTAINERS" | sed 's/^/   - /'
fi

# Ask about rebuilding
echo -e "\n${PURPLE}What would you like to do?${NC}"
echo -e "  ${GREEN}1)${NC} Rebuild everything (after code/Dockerfile changes)"
echo -e "  ${GREEN}2)${NC} Quick restart (no code changes)"
echo -e "  ${GREEN}3)${NC} Stop all containers"
echo -e "  ${GREEN}4)${NC} View logs only"
echo -e "  ${GREEN}5)${NC} Clean up and rebuild (remove old images)"
echo -e "  ${GREEN}6)${NC} Cancel"
echo -n "Your choice [1-6]: "
read -r choice

case $choice in
    1)
        echo -e "\n${GREEN}🔨 Rebuilding containers with latest code...${NC}"
        stop_containers
        check_ports
        docker compose build --no-cache
        docker compose up -d
        echo -e "${GREEN}✅ Containers rebuilt and started!${NC}"
        ;;
    2)
        echo -e "\n${BLUE}♻️  Quick restarting containers...${NC}"
        stop_containers
        check_ports
        docker compose up -d
        echo -e "${GREEN}✅ Containers restarted!${NC}"
        ;;
    3)
        echo -e "\n${YELLOW}🛑 Stopping all containers...${NC}"
        stop_containers
        echo -e "${GREEN}✅ All containers stopped${NC}"
        exit 0
        ;;
    4)
        echo -e "\n${BLUE}📋 Viewing logs only...${NC}"
        ;;
    5)
        echo -e "\n${PURPLE}🧹 Deep clean and rebuild...${NC}"
        stop_containers
        cleanup_images
        check_ports
        docker compose build --no-cache --pull
        docker compose up -d
        echo -e "${GREEN}✅ Clean rebuild complete!${NC}"
        ;;
    6)
        echo -e "\n${RED}❌ Cancelled${NC}"
        exit 0
        ;;
    *)
        echo -e "\n${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac

# Show status and logs (except for stop option)
if [ "$choice" != "3" ] && [ "$choice" != "6" ]; then
    echo -e "\n${CYAN}📊 Container Status:${NC}"
    docker compose ps
    
    echo -e "\n${BLUE}📋 Showing logs (Ctrl+C to exit)...${NC}"
    echo -e "${BLUE}==================================${NC}\n"
    
    # Show logs with timestamps
    docker compose logs -f --timestamps
fi
