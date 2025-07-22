#!/bin/bash
# Test GitHub Actions workflow locally
set -e

echo "=== Testing GitHub Actions Workflow Locally ==="
echo "This simulates the GitHub Actions environment"
echo

# Create a temporary directory for testing
TEST_DIR="/home/thierrygc/test_1/github_ready/test_workflow_env"
rm -rf $TEST_DIR
mkdir -p $TEST_DIR
cd $TEST_DIR

# Copy required files
cp ../requirements_pattern_scanner.txt .
cp ../combined_pattern_scanner_gh.py .
cp ../pattern_evaluator.py .
cp -r ../config .

# Create a virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

echo
echo "=== Step 1: Installing system dependencies ==="
# Note: We'll test without sudo for local testing
echo "Checking for TA-Lib..."

# Check if TA-Lib is already installed
if [ -f "/usr/local/lib/libta_lib.so" ]; then
    echo "TA-Lib already installed at /usr/local/lib"
else
    echo "TA-Lib not found. To install:"
    echo "  wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
    echo "  tar -xzf ta-lib-0.4.0-src.tar.gz"
    echo "  cd ta-lib/"
    echo "  ./configure --prefix=/usr/local"
    echo "  make"
    echo "  sudo make install"
    echo "  sudo ldconfig"
fi

echo
echo "=== Step 2: Installing Python dependencies ==="
python -m pip install --upgrade pip

# Set environment variables for TA-Lib
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=/usr/local/include:$C_INCLUDE_PATH

echo "Environment variables set:"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  LIBRARY_PATH=$LIBRARY_PATH"
echo "  C_INCLUDE_PATH=$C_INCLUDE_PATH"

echo
echo "Installing requirements (except TA-Lib)..."
# Install everything except TA-Lib first
grep -v "TA-Lib" requirements_pattern_scanner.txt > requirements_no_talib.txt
pip install -r requirements_no_talib.txt

echo
echo "Building TA-Lib Python wrapper from source..."
# Set proper paths for system TA-Lib
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=/usr/include:$C_INCLUDE_PATH

# Install TA-Lib Python wrapper
pip install --no-cache-dir TA-Lib

echo
echo "=== Step 3: Verifying installations ==="
echo "Testing TA-Lib import..."
python -c "import talib; print(f'✓ TA-Lib version: {talib.__version__}')" || echo "✗ TA-Lib import failed"

echo
echo "Testing TradingPatternScanner import..."
python -c "from tradingpatterns.tradingpatterns import detect_head_shoulder; print('✓ TradingPatternScanner imported successfully')" || echo "✗ TradingPatternScanner import failed"

echo
echo "=== Step 4: Testing scanner imports ==="
python -c "
import sys
sys.path.insert(0, '.')
try:
    import combined_pattern_scanner_gh
    print('✓ Scanner module imported successfully')
except Exception as e:
    print(f'✗ Scanner import failed: {e}')
"

echo
echo "=== Test complete ==="
echo "To clean up test environment: rm -rf $TEST_DIR"