#!/bin/bash
# Script to run all experiments for Delay Neural Operator benchmarking
# This script trains and evaluates all models on all datasets

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
RESULTS_DIR="./benchmark_results"
mkdir -p $RESULTS_DIR

# Set up logging
LOGFILE="$RESULTS_DIR/benchmark_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Starting Delay Neural Operator benchmark experiments at $(date)"
echo "Results will be saved to $RESULTS_DIR"
echo "Full logs saved to $LOGFILE"

# Models and datasets
MODELS=("stacked" "steps" "kernel")
DATASETS=("mackey_glass" "delayed_logistic" "neutral_dde" "reaction_diffusion")

# Function to run a single experiment
run_experiment() {
    MODEL=$1
    DATASET=$2
    
    echo "================================================="
    echo "Training $MODEL on $DATASET dataset"
    echo "================================================="
    
    # Create experiment directory
    EXP_DIR="$RESULTS_DIR/${MODEL}_${DATASET}"
    mkdir -p $EXP_DIR
    
    # Run training
    PYTHONPATH=. python scripts/train.py \
        model=$MODEL \
        data=$DATASET \
        logging.use_wandb=true \
        hydra.run.dir=$EXP_DIR \
        train.max_epochs=100
    
    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo "Training completed successfully"
        
        # Run evaluation
        echo "Evaluating $MODEL on $DATASET dataset"
        PYTHONPATH=. python scripts/evaluate.py \
            model=$MODEL \
            data=$DATASET \
            hydra.run.dir=$EXP_DIR
        
        # Check if evaluation succeeded
        if [ $? -eq 0 ]; then
            echo "Evaluation completed successfully"
        else
            echo "ERROR: Evaluation failed for $MODEL on $DATASET"
        fi
    else
        echo "ERROR: Training failed for $MODEL on $DATASET"
    fi
}

# Function to run all experiments
run_all_experiments() {
    # Create summary file
    SUMMARY_FILE="$RESULTS_DIR/benchmark_summary.csv"
    echo "model,dataset,avg_l2_error,avg_spectral_error,stability_metric,throughput,model_parameters" > $SUMMARY_FILE
    
    # Run all combinations
    for MODEL in "${MODELS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            run_experiment $MODEL $DATASET
            
            # Append results to summary if available
            RESULT_FILE="$RESULTS_DIR/${MODEL}_${DATASET}/evaluation_results/${MODEL}_${DATASET}_results.csv"
            if [ -f "$RESULT_FILE" ]; then
                tail -n +2 "$RESULT_FILE" >> $SUMMARY_FILE
            fi
        done
    done
    
    echo "All experiments completed. Summary saved to $SUMMARY_FILE"
}

# Function to generate comparison plots
generate_comparison_plots() {
    echo "Generating comparison plots..."
    
    # Run Python script to generate comparison plots
    PYTHONPATH=. python scripts/generate_comparison_plots.py \
        --results_file $RESULTS_DIR/benchmark_summary.csv \
        --output_dir $RESULTS_DIR/plots
        
    echo "Comparison plots saved to $RESULTS_DIR/plots"
}

# Main execution
run_all_experiments
generate_comparison_plots

echo "Benchmark completed at $(date)"
echo "Full results available in $RESULTS_DIR"
