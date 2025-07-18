#!/bin/bash
# Bash script to generate DDE datasets only for families that failed in the previous run

# Set root directory
rootDir=$(dirname $(realpath $0))
dataDir="${rootDir}/data"
failed_families=("neutral_dde" "reaction_diffusion")

# Set Python path to include project root directory
export PYTHONPATH="$rootDir:$PYTHONPATH"

# Print start message
timestamp=$(date)
echo "Starting DDE dataset generation for failed families - $timestamp"
echo "Generating datasets for previously failed DDE families"
echo "----------------------------------------"

# Create base data directory if it doesn't exist
mkdir -p $dataDir

# Loop through each failed DDE family
i=1
total=${#failed_families[@]}
for family_name in "${failed_families[@]}"; do
    echo "$i/$total: Generating $family_name datasets"
    
    # Create family-specific directory
    family_dir="${dataDir}/${family_name}"
    mkdir -p $family_dir
    
    # Generate dataset (10K samples)
    echo "Generating dataset for $family_name (10K samples)..."
    python -m data_pipeline.generate_dataset --family $family_name --N 10000 --output-dir $family_dir --plot-examples
    
    echo "----------------------------------------"
    i=$((i+1))
done

# Generate combined dataset with train/test split
combinedDir="${dataDir}/combined"
mkdir -p $combinedDir

echo "Generating combined dataset with train/test split (hold out τ ∈ [2.0, 3.0] for testing)..."

for family_name in "${failed_families[@]}"; do
    python -m data_pipeline.generate_dataset --family $family_name --N 10000 --output-dir $combinedDir --tau-split 2.0 3.0 --plot-examples
done

# Print completion message
timestamp_end=$(date)
echo "----------------------------------------"
echo "Dataset generation completed - $timestamp_end"
echo "Total samples: $((${#failed_families[@]} * 10000)) (10K per family)"
echo "Data saved to: $dataDir/"
echo "----------------------------------------"
