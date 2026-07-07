# Setup script for 3D Reconstruction Pipeline
# Run this script to set up the environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  3D Reconstruction Pipeline Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is available
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Error: git is not installed" -ForegroundColor Red
    exit 1
}

# Check if python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: python is not installed" -ForegroundColor Red
    exit 1
}

# Step 1: Clone MASt3R repository
Write-Host "[Step 1] Cloning MASt3R repository..." -ForegroundColor Yellow
if (Test-Path "repos\mast3r") {
    Write-Host "  MASt3R already cloned, skipping..." -ForegroundColor Gray
} else {
    git clone --recursive https://github.com/naver/mast3r.git repos/mast3r
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Error cloning MASt3R" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✓ MASt3R cloned successfully" -ForegroundColor Green
}

# Step 2: Install MASt3R dependencies
Write-Host "[Step 2] Installing MASt3R dependencies..." -ForegroundColor Yellow
Set-Location repos\mast3r

# Install main requirements
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Error installing MASt3R requirements" -ForegroundColor Red
    exit 1
}

# Install DUSt3R requirements
pip install -r dust3r\requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Error installing DUSt3R requirements" -ForegroundColor Red
    exit 1
}

Set-Location ..\..

# Step 3: Install additional dependencies
Write-Host "[Step 3] Installing additional dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Error installing additional requirements" -ForegroundColor Red
    exit 1
}

# Step 4: Create sample input images
Write-Host "[Step 4] Creating sample input..." -ForegroundColor Yellow
if (-not (Test-Path "input\*.jpg")) {
    Write-Host "  Please add your images to the input/ directory" -ForegroundColor Gray
    Write-Host "  Supported formats: .jpg, .png" -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Add your images (2-5) to the input/ directory" -ForegroundColor White
Write-Host "  2. Run the pipeline:" -ForegroundColor White
Write-Host "     python main.py --input ./input/ --output ./output/" -ForegroundColor White
Write-Host ""
