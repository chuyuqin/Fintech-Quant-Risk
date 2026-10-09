# Fintech Quant Risk

Coursework for FinTech 545.

- `Assignment1/`: original Assignment 1 submission.
- `Assignment2/`: written PDF, data, reproducible Python analysis, and tests.
- `functional-tests-1/`: earlier risk-management package and functional tests.

## Reusable Python library

Assignment 2 includes the installable `quant_risk` package. From the repository
root, install it with:

```bash
python -m pip install -e ./Assignment2
```

Then import its methods from any Python script:

```python
from quant_risk import historical_risk, fit_margin, near_psd, fit_copulas
```

See `Assignment2/README.md` for a clean-environment setup and commands to reproduce
all results and rebuild the report.
