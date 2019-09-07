# pi0_reco

Pi0 reconstruction for LArTPC detectors

The [pi0](./pi0) folder contains a python module that can be invoked as:
```python
import pi0
```

Put functions in either [pi0](./pi0) or [pi0/utils](./pi0/utils) folders, and load them in your own jupyter analysis notebooks.  We can add additional directories if necessary

General Contribution Rules:
1. Fork your own copy from github, and create a pull request to contribute
2. Generic LArTPC reconstruction code should go into [lartpc-mlreco3d](https://github.com/DeepLearnPhysics/lartpc_mlreco3d)
3. __No Jupyter Notebooks!__
4. Make sure temporary/generated files are in `.gitignore` and not committed.
5. Please comment your code and put a docstring at the head of functions. 