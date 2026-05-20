# PINN for Zimm-Bragg / Ising helix-coil kinetics

Code, data-generation pipeline, and paper source for a physics-informed
neural network that learns the closed-moment Glauber dynamics of an
Ising-like Zimm-Bragg chain.

## Layout

```
.
├── code/                          shared library + PINN variants
│   ├── shared.py                  data loading, Glauber forward map,
│   │                              MLP factory, Adam→L-BFGS trainer
│   ├── variants.py                five PINN variants:
│   │                              three-separate, inverse, analytical
│   │                              (chosen), three-channel, per-run
│   └── observable_variants.py     "start-from-m / ε / ε₂" experiment
├── grid_simulator.py              Glauber heat-bath Monte Carlo
│                                  simulator for the (β, h, J) grid
├── data_grid/                     per-condition .npz trajectories
│                                  (output of grid_simulator)
├── data_grid_2/                   second grid (more negative J, h)
├── checkpoints_grid/              saved PINN model states
├── paper/                         IEEE-format paper source + figures
│   ├── main.tex                   paper
│   ├── references.bib             bibliography
│   ├── figures/                   figures referenced from main.tex
│   └── README.md                  build instructions for the paper
├── PINN_PerODE.ipynb              chosen-pipeline training and
│                                  in-domain / extrapolation plots
├── benchmarks.ipynb               head-to-head benchmark of the five
│                                  PINN variants on one condition
├── benchmarks_grid2.ipynb         same benchmark on the second grid
├── alternative_observables.ipynb  start-from-m / ε / ε₂ comparison
├── generate_chunk_1.ipynb         driver: simulate first half of grid
└── generate_chunk_2.ipynb         driver: simulate second half of grid
```

## Reproducing the results

1. Generate the dataset (skip if `data_grid/` is already present):

   ```
   jupyter nbconvert --to notebook --execute generate_chunk_1.ipynb
   jupyter nbconvert --to notebook --execute generate_chunk_2.ipynb
   ```

2. Train and produce the in-domain / extrapolation figures:

   ```
   jupyter nbconvert --to notebook --execute PINN_PerODE.ipynb
   ```

3. Run the variant benchmark:

   ```
   jupyter nbconvert --to notebook --execute benchmarks.ipynb
   jupyter nbconvert --to notebook --execute benchmarks_grid2.ipynb
   ```

4. Build the paper from `paper/` (see `paper/README.md`).

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Numba (for `grid_simulator.py` JIT)
- Joblib (optional, for parallel simulation)
- Matplotlib

## Authors

- Meruzhan Khachatryan, BS in Data Science, American University of Armenia
- Supervisor: Varazdat Stepanyan
