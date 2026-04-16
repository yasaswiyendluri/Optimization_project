# Selective Maintenance Optimization using ASA-PSO

## Overview

This project focuses on optimizing maintenance decisions for a multi-stage machine system. The objective is to determine suitable maintenance levels for each machine such that overall system reliability is maximized while satisfying cost and time constraints.

The approach integrates:

* Reliability modeling using the Weibull distribution
* Quality degradation modeling
* Maintenance cost and time constraints
* Optimization using Particle Swarm Optimization (PSO) and ASA-PSO

**ASA-PSO (Adaptive Simulated Annealing – Particle Swarm Optimization)** enhances standard PSO by improving exploration through adaptive parameter adjustment.

---

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

* **Source:** UCI Machine Learning Repository / Kaggle
* **File used:** `ai4i2020.csv`
* **Location:** `/data/ai4i2020.csv`

### Dataset Description

The dataset contains simulated machine operation data with key attributes such as:

* Machine Type (L, M, H)
* Tool Wear (minutes)
* Torque (Nm)
* Machine Failure (0 = No Failure, 1 = Failure)
* Failure modes (TWF, HDF, PWF, OSF, RNF)

For this project:

* Machines are grouped into three stages based on type for modeling purposes:

  * L → Stage 1
  * M → Stage 2
  * H → Stage 3
* This grouping represents a simplified multi-stage production system.
* Time-to-Failure (TTF) is approximated using failure occurrence patterns and tool wear progression, and is used for reliability modeling.

---

## Project Structure

```
Optimization_project/
├── data/
│   └── ai4i2020.csv
├── src/
│   ├── models.py
│   ├── optimizer.py
├── main.py
├── config.py
├── requirements.txt
├── README.md
```

---

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
pandas  
numpy  
matplotlib  
```

---

## How to Run

From the project root directory:

```
python main.py
```

```
To run individual modes:
python main.py --mode eda
python main.py --mode quality
python main.py --mode states
python main.py --mode optimize
python main.py --mode compare

```
---

## Methodology

1. Load and preprocess the dataset
2. Perform exploratory analysis of machine behavior and failure modes
3. Develop:

   * Reliability model using Weibull distribution
   * Quality degradation model
4. Incorporate maintenance cost and time constraints
5. Apply optimization techniques (PSO and ASA-PSO)
6. Compare convergence and performance of both methods

---

## Objective Function

The overall system reliability is defined as:

```
R_sys = R1 × R2 × R3
```

where each stage reliability is modeled using the Weibull distribution based on estimated failure characteristics.

### Constraints

* Total maintenance cost ≤ specified budget
* Total maintenance time ≤ allowed limit

---

## Output

All outputs are stored in the `results/` folder:

* Failure mode distribution plots
* Reliability probability plots
* Quality degradation curves
* Optimal maintenance strategy
* Convergence comparison (PSO vs ASA-PSO)

---

## Notes

* The dataset is included in the repository for convenience.
* All configurable parameters (budget, machine settings, optimizer settings) are defined in `config.py`.
* Random seeds are fixed to ensure reproducibility of results.

---

