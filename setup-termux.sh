#!/data/data/com.termux/files/usr/bin/bash
# g4f-Bridge Setup Script for Termux
# Run: bash setup-termux.sh

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            g4f-Bridge Setup for Termux                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Detect environment
if [ -d "/data/data/com.termux" ]; then
    echo "✓ Termux environment detected"
    IS_TERMUX=1
else
    echo "⚠ Not running in Termux, using standard Linux setup"
    IS_TERMUX=0
fi

# Update packages
echo ""
echo "▶ Updating packages..."
if [ "$IS_TERMUX" = "1" ]; then
    pkg update -y && pkg upgrade -y
else
    sudo apt update -y && sudo apt upgrade -y 2>/dev/null || true
fi

# Install dependencies
echo ""
echo "▶ Installing system dependencies..."
if [ "$IS_TERMUX" = "1" ]; then
    pkg install -y python rust binutils build-essential libffi openssl git
else
    sudo apt install -y python3 python3-pip python3-venv git 2>/dev/null || true
fi

# Create project directory
echo ""
echo "▶ Setting up project directory..."
PROJECT_DIR="$HOME/g4f-bridge"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create virtual environment
echo ""
echo "▶ Creating Python virtual environment..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

$PYTHON_CMD -m venv venv 2>/dev/null || {
    echo "⚠ venv module not available, using global Python"
    mkdir -p venv/bin
    ln -sf $(which $PYTHON_CMD) venv/bin/python
    ln -sf $(which pip) venv/bin/pip 2>/dev/null || ln -sf $(which pip3) venv/bin/pip
}

# Activate virtual environment
source venv/bin/activate 2>/dev/null || true

# Upgrade pip
echo ""
echo "▶ Upgrading pip..."
pip install --upgrade pip setuptools wheel 2>/dev/null || true

# Install dependencies one by one
echo ""
echo "▶ Installing Python dependencies..."
echo ""

install_package() {
    local package=$1
    local required=$2
    
    echo -n "  Installing $package... "
    if pip install "$package" 2>/dev/null; then
        echo "✓"
        return 0
    else
        if [ "$required" = "required" ]; then
            echo "✗ FAILED (REQUIRED)"
            return 1
        else
            echo "✗ FAILED (optional)"
            return 0
        fi
    fi
}

# Required packages
install_package "fastapi" "required"
install_package "g4f" "required"
install_package "aiohttp" "required"
install_package "requests" "required"

# Optional packages (may fail on ARM)
install_package "uvicorn[standard]" "optional" || install_package "uvicorn" "optional"
install_package "flask" "optional"
install_package "aiosqlite" "optional"
install_package "apscheduler" "optional"
install_package "tiktoken" "optional"
install_package "orjson" "optional"

# Download project files if not present
echo ""
echo "▶ Checking project files..."

# Create files if they don't exist (copy from current directory or download)
FILES=(
    "environment.py"
    "config.py"
    "logger_setup.py"
    "storage.py"
    "resilience.py"
    "scanner.py"
    "token_manager.py"
    "router.py"
    "updater.py"
    "main.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ⚠ $file not found - please copy it manually"
    fi
done

# Create startup script
echo ""
echo "▶ Creating startup script..."
cat > "$PROJECT_DIR/start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
python main.py
EOF
chmod +x "$PROJECT_DIR/start.sh"
echo "  ✓ start.sh created"

# Create Termux boot script (auto-start)
if [ "$IS_TERMUX" = "1" ]; then
    echo ""
    echo "▶ Setting up Termux auto-start..."
    mkdir -p ~/.termux/boot
    cat > ~/.termux/boot/start-g4f-bridge.sh << EOF
#!/data/data/com.termux/files/usr/bin/bash
termux-wake-lock
cd $PROJECT_DIR
source venv/bin/activate 2>/dev/null || true
python main.py
EOF
    chmod +x ~/.termux/boot/start-g4f-bridge.sh
    echo "  ✓ Auto-start configured"
    echo "  ℹ Install Termux:Boot from F-Droid for auto-start on boot"
fi

# Print completion message
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "To start the bridge:"
echo ""
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Or simply:"
echo ""
echo "  $PROJECT_DIR/start.sh"
echo ""
echo "The API key will be displayed when the server starts."
echo ""

# Offer to start now
read -p "Start g4f-Bridge now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$PROJECT_DIR"
    source venv/bin/activate 2>/dev/null || true
    python main.py
fi