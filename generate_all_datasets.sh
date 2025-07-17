#!/bin/bash

# generate_all_datasets.sh
# Script to generate 10K samples for all DDE families

# Create output directories
mkdir -p data/mackey_glass
mkdir -p data/delayed_logistic
mkdir -p data/neutral_dde
mkdir -p data/reaction_diffusion

# Print start message
echo "Starting DDE dataset generation - $(date)"
echo "Generating 10,000 samples for each DDE family"
echo "----------------------------------------"

# Set common parameters
N=10000          # Number of samples per family
TAU_MIN=0.5      # Minimum delay value
TAU_MAX=5.0      # Maximum delay value
T=50.0           # Maximum integration time
DT=0.1           # Step size
OUTPUT_DIR="data" # Output directory
HIST_TYPE="cubic_spline" # History type

# Function to generate dataset for a specific family
generate_dataset() {
    family=$1
    output_subdir=$2
    
    echo "Generating dataset for $family (10K samples)..."
    
    # Run the Python script
    python data_pipeline/generate_dataset.py \
        --families "$family" \
        --N $N \
        --tau_min $TAU_MIN \
        --tau_max $TAU_MAX \
        --T $T \
        --dt $DT \
        --history_type $HIST_TYPE \
        --output_dir "$OUTPUT_DIR/$output_subdir" \
        --plot_examples
    
    # Check if generation was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully generated dataset for $family"
    else
        echo "✗ Failed to generate dataset for $family"
    fi
    
    echo "----------------------------------------"
}

# Generate datasets for each family with appropriate subfolders
echo "1/4: Generating Mackey-Glass datasets"
generate_dataset "mackey_glass" "mackey_glass"

echo "2/4: Generating Delayed Logistic datasets"
generate_dataset "delayed_logistic" "delayed_logistic"

echo "3/4: Generating Neutral DDE datasets"
generate_dataset "neutral_dde" "neutral_dde"

echo "4/4: Generating Reaction-Diffusion datasets"
generate_dataset "reaction_diffusion" "reaction_diffusion"

# Generate a combined dataset with train/test split based on delay values
echo "Generating combined dataset with train/test split (hold out τ ∈ [2.0, 3.0] for testing)..."
python data_pipeline/generate_dataset.py \
    --families "mackey_glass" "delayed_logistic" "neutral_dde" "reaction_diffusion" \
    --N $N \
    --tau_min $TAU_MIN \
    --tau_max $TAU_MAX \
    --T $T \
    --dt $DT \
    --history_type $HIST_TYPE \
    --output_dir "$OUTPUT_DIR/combined" \
    --tau_split_min 2.0 \
    --tau_split_max 3.0 \
    --plot_examples

echo "----------------------------------------"
echo "Dataset generation completed - $(date)"
echo "Total samples: 40,000 (10K per family)"
echo "Data saved to: $OUTPUT_DIR/"
echo "----------------------------------------"
