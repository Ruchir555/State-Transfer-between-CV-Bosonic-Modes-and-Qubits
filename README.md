# State Transfer between CV Bosonic Modes and Qubits

This repository contains an exploratory research notebook for constructing
sequences of T-transforms between probability vectors related by majorization.
The calculations were developed in the context of transforming truncated
two-mode squeezed-state Schmidt coefficients toward a Bell-state distribution.

[Open the notebook in Google Colab](https://colab.research.google.com/github/Ruchir555/State-Transfer-between-CV-Bosonic-Modes-and-Qubits/blob/main/T_parameter_finder.ipynb)

## Core calculation

For descending vectors `x` and `y` satisfying `x ≺ y`, the notebook searches
for an index `k` such that

```text
y[k] <= x[0] <= y[k - 1]
```

and computes

```text
t = (x[0] - y[k]) / (y[0] - y[k]).
```

It then applies the corresponding pairwise T-transform, reduces the vectors,
and repeats. The main function is:

```python
t_values, index_choices = t_parameters(x, y, checkMajorization=True)
```

`t_values` contains the mixing parameter at each step. `index_choices` reports
the selected index using one-based indexing, matching the mathematical notes.

## Worked example

```python
x = [5, 3, 2]
y = [6, 3, 1]

t_values, index_choices = t_parameters(
    x,
    y,
    checkMajorization=True,
)

print(t_values)       # [2/3, 2/3] numerically
print(index_choices)  # indices used by the successive transforms
```

The notebook also explores truncated two-mode squeezed states, Bell-state
targets, entanglement entropy, nearest-neighbour versus hopping transforms and
the resulting measurement parameters.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab T_parameter_finder.ipynb
```

The notebook records an evolving research calculation rather than a packaged
library. Run its sections in order within the part being studied; later cells
reuse variables defined by earlier exploratory cells. The sorting helpers also
modify list inputs in place, so pass copies if the original ordering matters.

## Tested module and continuous integration

The reusable functions in `t_transforms.py` provide non-mutating
majorization checks, T-transform parameters and truncated TMSS probability
vectors. Run their analytical tests with:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs these tests on Python 3.11 and 3.13 for every push and
pull request. The exploratory notebook is retained as the research record.
