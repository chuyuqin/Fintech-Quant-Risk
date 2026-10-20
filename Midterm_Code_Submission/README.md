# Midterm calculations

## Run

Python 3.12 was used. Open a terminal in this folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python solve_midterm.py
```

The script reads the four CSV files in `data/` and writes numerical results to
`results/`. Calculations and checks are labeled by problem and part.

## Contents

- `solve_midterm.py`: analysis, numerical checks, and result exports.
- `quant_risk/`: reusable numerical functions.
- `data/`: input prices, returns, and holdings.
- `results/`: numerical tables and calculation metadata; see its README for an index.
- `requirements.txt`: package versions.

The simulation uses 100,000 draws with seed 5452025. Problem 1(c) uses
absolute tolerance 1e-10 and zero relative tolerance. The script checks matrix
properties, regression identities, parameter validity, inverse-CDF accuracy,
position valuations, portfolio aggregation, and empirical risk calculations.
