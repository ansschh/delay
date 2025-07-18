# generate_all_datasets.ps1
# Script to generate 10K samples for all DDE families

# Create output directories
New-Item -Path "data/mackey_glass" -ItemType Directory -Force | Out-Null
New-Item -Path "data/delayed_logistic" -ItemType Directory -Force | Out-Null
New-Item -Path "data/neutral_dde" -ItemType Directory -Force | Out-Null
New-Item -Path "data/reaction_diffusion" -ItemType Directory -Force | Out-Null
New-Item -Path "data/combined" -ItemType Directory -Force | Out-Null

# Print start message
Write-Host "Starting DDE dataset generation - $(Get-Date)"
Write-Host "Generating 10,000 samples for each DDE family"
Write-Host "----------------------------------------"

# Set common parameters
$N = 10000          # Number of samples per family
$TAU_MIN = 0.5      # Minimum delay value
$TAU_MAX = 5.0      # Maximum delay value
$T = 50.0           # Maximum integration time
$DT = 0.1           # Step size
$OUTPUT_DIR = "data" # Output directory
$HIST_TYPE = "cubic_spline" # History type

# Function to generate dataset for a specific family
function New-Dataset {
    param (
        [string]$family,
        [string]$output_subdir
    )
    
    $datasetFile = "$OUTPUT_DIR/$output_subdir/$family.pkl"
    
    # Check if dataset already exists
    if (Test-Path $datasetFile) {
        Write-Host "Dataset for $family already exists at $datasetFile" -ForegroundColor Cyan
        Write-Host "Skipping generation for this family."
    } else {
        Write-Host "Generating dataset for $family (10K samples)..."
        
        # Run the Python script
        python data_pipeline/generate_dataset.py `
            --families "$family" `
            --N $N `
            --tau_min $TAU_MIN `
            --tau_max $TAU_MAX `
            --T $T `
            --dt $DT `
            --history_type $HIST_TYPE `
            --output_dir "$OUTPUT_DIR/$output_subdir" `
            --plot_examples
        
        # Check if generation was successful
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Successfully generated dataset for $family" -ForegroundColor Green
        } else {
            Write-Host "✗ Failed to generate dataset for $family" -ForegroundColor Red
        }
    }
    
    Write-Host "----------------------------------------"
}

# Generate datasets for each family with appropriate subfolders
Write-Host "1/4: Generating Mackey-Glass datasets"
New-Dataset -family "mackey_glass" -output_subdir "mackey_glass"

Write-Host "2/4: Generating Delayed Logistic datasets"
New-Dataset -family "delayed_logistic" -output_subdir "delayed_logistic"

Write-Host "3/4: Generating Neutral DDE datasets"
New-Dataset -family "neutral_dde" -output_subdir "neutral_dde"

Write-Host "4/4: Generating Reaction-Diffusion datasets"
New-Dataset -family "reaction_diffusion" -output_subdir "reaction_diffusion"

# Function to check if combined train/test files exist for a family
function Test-CombinedDataset {
    param (
        [string]$family
    )
    
    $trainFile = "$OUTPUT_DIR/combined/${family}_train.pkl"
    $testFile = "$OUTPUT_DIR/combined/${family}_test.pkl"
    
    if ((Test-Path $trainFile) -and (Test-Path $testFile)) {
        Write-Host "- ${family}: Train/test datasets already exist" -ForegroundColor Cyan
        return $true  # Files exist
    } else {
        return $false  # Files don't exist
    }
}

# Generate a combined dataset with train/test split based on delay values
Write-Host "Checking for combined datasets with train/test split..."

# Create combined directory if it doesn't exist
New-Item -Path "$OUTPUT_DIR/combined" -ItemType Directory -Force | Out-Null

# Check which families need combined dataset generation
$missingFamilies = @()
$allFamilies = @("mackey_glass", "delayed_logistic", "neutral_dde", "reaction_diffusion")

foreach ($family in $allFamilies) {
    if (-not (Test-CombinedDataset -family $family)) {
        $missingFamilies += $family
    }
}

# Only generate if there are missing families
if ($missingFamilies.Count -gt 0) {
    Write-Host "Generating combined dataset with train/test split (hold out τ ∈ [2.0, 3.0] for testing)..."
    Write-Host "Generating for families: $($missingFamilies -join ", ")"
    
    python data_pipeline/generate_dataset.py `
        --families $($missingFamilies -join " ") `
        --N $N `
        --tau_min $TAU_MIN `
        --tau_max $TAU_MAX `
        --T $T `
        --dt $DT `
        --history_type $HIST_TYPE `
        --output_dir "$OUTPUT_DIR/combined" `
        --tau_split_min 2.0 `
        --tau_split_max 3.0 `
        --plot_examples
} else {
    Write-Host "All combined train/test datasets already exist. Skipping combined generation." -ForegroundColor Green
}

Write-Host "----------------------------------------"
Write-Host "Dataset generation completed - $(Get-Date)"
Write-Host "Total samples: 40,000 (10K per family)"
Write-Host "Data saved to: $OUTPUT_DIR/"
Write-Host "----------------------------------------"
