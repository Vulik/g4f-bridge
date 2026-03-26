#!/bin/bash
# g4f-Bridge Setup Script for Linux
# Run: bash setup.sh

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║               g4f-Bridge Setup for Linux                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check for Python
echo "▶ Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "  ✓ Found $(python3 --version)"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "  ✓ Found $(python --version)"
else
    echo "  ✗ Python not found! Please install Python 3.8+"
    exit 1
fi

# Create project directory
echo ""
echo "▶ Setting up project directory..."
PROJECT_DIR="$HOME/g4f-bridge"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create virtual environment
echo ""
echo "▶ Creating virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "▶ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "▶ Installing dependencies..."
pip install fastapi uvicorn[standard] g4f aiohttp requests aiosqlite apscheduler flask

# Try optional packages
echo ""
echo "▶ Installing optional dependencies..."
pip install tiktoken 2>/dev/null && echo "  ✓ tiktoken" || echo "  ✗ tiktoken (using fallback)"
pip install orjson 2>/dev/null && echo "  ✓ orjson" || echo "  ✗ orjson"

# Create startup script
echo ""
echo "▶ Creating startup script..."
cat > "$PROJECT_DIR/start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python main.py
EOF
chmod +x "$PROJECT_DIR/start.sh"

# Create systemd service (optional)
echo ""
echo "▶ Creating systemd service file..."
cat > "$PROJECT_DIR/g4f-bridge.service" << EOF
[Unit]
Description=g4f-Bridge OpenAI API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "  ✓ Service file created at $PROJECT_DIR/g4f-bridge.service"
echo "  ℹ To install as system service:"
echo "    sudo cp $PROJECT_DIR/g4f-bridge.service /etc/systemd/system/"
echo "    sudo systemctl enable g4f-bridge"
echo "    sudo systemctl start g4f-bridge"

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

# Offer to start now
read -p "Start g4f-Bridge now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$PROJECT_DIR"
    source venv/bin/activate
    python main.py
fi