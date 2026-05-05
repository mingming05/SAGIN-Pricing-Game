# Decentralized Dynamic Bandwidth Pricing in SAGIN: A Stackelberg Game Approach

This repository contains the simulation code for the paper: **"Dynamic Bandwidth Pricing in Space-Air-Ground Integrated Networks: A Decentralized Stackelberg Game Approach"**.

## Overview
This project models the bandwidth trading process in a 6G Space-Air-Ground Integrated Network (SAGIN) as a single-leader, multi-follower Stackelberg game. It includes:
1.  **Closed-form analytical solutions** for the Stackelberg Equilibrium using backward induction under a full participation regime.
2.  A **fully decentralized finite-difference gradient ascent algorithm** with numerical regularization that preserves UAV data privacy and prevents gradient explosion.
3.  **Monte Carlo simulations** with randomized density parameters and lower-bound physical constraints to validate empirical convergence and robustness under varying traffic densities.

## Requirements
To run the simulations, you need Python 3.x and the following libraries:
- `numpy`
- `matplotlib`
- `scipy`

You can install them using:
```bash
pip install -r requirements.txt
```

## How to Run
Simply execute the main Python script (or `simulation.py` if renamed):
```bash
python main.py
```
The script will automatically generate and save 5 high-quality figures (`.png` and `.eps` formats) used in the paper, illustrating utility curves, demand dynamics, algorithmic convergence, density impacts, and Monte Carlo evaluations.

## Figures Generated
1. **`Fig1_LEO_Utility`**: Shows the strictly concave utility function of the LEO satellite, confirming the unique global optimal price ($p^*$).
2. **`Fig2_UAV_Demand`**: Illustrates the monotonically decreasing bandwidth demand of heterogeneous UAVs with respect to price.
3. **`Fig3_Convergence`**: Demonstrates the rapid and stable empirical convergence of the decentralized finite-difference algorithm, utilizing the $\delta$ regularization parameter.
4. **`Fig4_Density_Impact`**: Analyzes the macroscopic impact of escalating ground user density, showing a sub-linear price increase and a sharp utility rise.
5. **`Fig5_Monte_Carlo`**: Validates the expected LEO utility and statistical reliability via 500 independent randomized trials with lower-bounded uniform distributions.