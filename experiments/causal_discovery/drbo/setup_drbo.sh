#!/bin/bash
# Run once from experiments/causal_discovery/drbo/
#
# Usage:
#   chmod +x setup_drbo.sh
#   ./setup_drbo.sh

set -e

echo "======================================"
echo "  DrBO + Causal SHAP Setup"
echo "======================================"

# 1. Clone DrBO
if [ ! -d "DrBO" ]; then
    echo "[1/4] Cloning DrBO repository..."
    git clone https://github.com/baosws/DrBO.git
else
    echo "[1/4] DrBO directory already exists - skipping clone."
fi

# 2. Install DrBO requirements
echo "[2/4] Installing DrBO requirements..."
pip install -r DrBO/requirements.txt --quiet

# 3. Install additional dependencies
echo "[3/4] Installing additional packages..."
pip install shap scikit-learn pandas numpy scipy openpyxl networkx matplotlib --quiet

# 4. Install DrBO as editable package
echo "[4/4] Making DrBO importable..."
# DrBO doesn't have a setup.py; we add it to path via .pth file
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$(pwd)/DrBO" > "${SITE_PKG}/drbo_path.pth"
echo "Added DrBO to Python path via ${SITE_PKG}/drbo_path.pth"

echo ""
echo "======================================"
echo "  Setup complete!"
echo "  Open drbo_causal_shap.ipynb"
echo "======================================"