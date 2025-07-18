#!/usr/bin/env pwsh
# PowerShell script to generate DDE datasets only for families that failed in the previous run

# Set root directory
$rootDir = $PSScriptRoot
$dataDir = Join-Path $rootDir "data"
$failed_families = @("neutral_dde", "reaction_diffusion")

# Print start message
$timestamp = Get-Date -Format "ddd MMM dd HH:mm:ss yyyy"
Write-Host "Starting DDE dataset generation for failed families - $timestamp"
Write-Host "Generating datasets for previously failed DDE families"
Write-Host "----------------------------------------"

# Create base data directory if it doesn't exist
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}

# Loop through each failed DDE family
$i = 1
foreach ($family_name in $failed_families) {
    Write-Host "$i/$($failed_families.Count): Generating $family_name datasets"
    
    # Create family-specific directory
    $family_dir = Join-Path $dataDir $family_name
    if (-not (Test-Path $family_dir)) {
        New-Item -ItemType Directory -Path $family_dir | Out-Null
    }
    
    # Generate dataset (10K samples)
    Write-Host "Generating dataset for $family_name (10K samples)..."
    python -m data_pipeline.generate_dataset --family $family_name --N 10000 --output-dir $family_dir --plot-examples
    
    Write-Host "----------------------------------------"
    $i++
}

# Generate combined dataset with train/test split
$combinedDir = Join-Path $dataDir "combined"
if (-not (Test-Path $combinedDir)) {
    New-Item -ItemType Directory -Path $combinedDir | Out-Null
}

Write-Host "Generating combined dataset with train/test split (hold out τ ∈ [2.0, 3.0] for testing)..."

foreach ($family_name in $failed_families) {
    python -m data_pipeline.generate_dataset --family $family_name --N 10000 --output-dir $combinedDir --tau-split 2.0 3.0 --plot-examples
}

# Print completion message
$timestamp_end = Get-Date -Format "ddd MMM dd HH:mm:ss yyyy"
Write-Host "----------------------------------------"
Write-Host "Dataset generation completed - $timestamp_end"
Write-Host "Total samples: $($failed_families.Count * 10000) (10K per family)"
Write-Host "Data saved to: $dataDir/"
Write-Host "----------------------------------------"
