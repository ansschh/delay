# Delay Differential Equation (DDE) Data Pipeline

This project implements a modular Python-based pipeline for generating datasets from various families of Delay Differential Equations. The pipeline is designed to create (history, τ, solution) triples that can be used to train Delay Neural Operators (Delay-NO).

## Project Structure

```
data_pipeline/
├─ families/              # DDE family implementations
│  ├─ __init__.py
│  ├─ mackey_glass.py     # Mackey-Glass equation
│  ├─ delayed_logistic.py # Delayed logistic equation
│  ├─ neutral_dde.py      # Neutral DDE
│  └─ reaction_diffusion.py # Reaction-diffusion with delay
├─ solvers/               # DDE solvers
│  ├─ __init__.py
│  ├─ method_of_steps.py  # Method of steps with continuous RK
│  └─ radar5_wrapper.py   # Wrapper for stiff/neutral DDEs
├─ utils/                 # Utility functions
│  ├─ history_generators.py # Functions to generate history data
│  └─ io.py               # Input/output utilities
└─ generate_dataset.py    # Main script to generate datasets
```

## Requirements

```
numpy
scipy
matplotlib
pydelay  # For stiff DDE solving (similar to RADAR5)
```

## Usage

### Basic Usage

To generate a dataset for the Mackey-Glass equation with default parameters:

```powershell
python data_pipeline/generate_dataset.py --families mackey_glass --N 100 --plot_examples
```

### Advanced Usage

Generate multiple families with custom parameters:

```powershell
python data_pipeline/generate_dataset.py --families mackey_glass delayed_logistic --N 1000 --tau_min 0.5 --tau_max 10.0 --T 100.0 --dt 0.05 --history_type fourier --output_dir data --plot_examples
```

### Train/Test Split

To hold out certain delay values for testing (e.g., τ ∈ [2, 3]):

```powershell
python data_pipeline/generate_dataset.py --families mackey_glass --N 1000 --tau_split_min 2.0 --tau_split_max 3.0
```

## Key Features

1. **Modular Design**: Each DDE family is implemented as a separate module.
2. **Multiple History Types**: Support for cubic spline, Fourier series, and filtered Brownian motion history functions.
3. **Method of Steps**: Implementation follows Bellen & Zennaro's approach with continuous RK.
4. **Stiff Solver**: Option to use a RADAR5-style solver for stiff or neutral DDEs.
5. **Spatial DDEs**: Includes reaction-diffusion equations with spatial discretization.
6. **Delay Extrapolation Testing**: Option to hold out certain delay values for testing.

## Adding New DDE Families

To add a new DDE family:

1. Create a new Python file in the `families` directory.
2. Implement the `f(t, u, u_tau, ...)` function for the right-hand side.
3. Implement a `random_history(τ, ...)` function to generate initial histories.
4. Optionally, add a `stiff = True` flag if needed.

## Next Enhancements

- Add more diverse history generators.
- Implement adaptive sampling for chaotic regimes.
- Add more visualization tools.
- Support for distributed computation for large datasets.
