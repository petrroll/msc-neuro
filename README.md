# msc-neuro
Deep-learning architectures for analysing population neural data:
- Research into DL architectures targeting pop. data recorded from mammalian primary visual cortex.
- Development of new parametrized kernels for the NDN3 library.
- Experiments testing novel combinations of already available/newly created methods.
- Master thesis (text eventually [here](https://github.com/petrroll/msc-thesis)).


### Setup:
0. Hydrate submodules: `git submodule update --init --recursive`
1. Build virtenv + deps: `make env-dev` (interactive dev) | `make env` (just script deps)
2. Activate virtenv: `source ./activate`
3. Run scripts, start `jupyter notebook`, ...

> Note: If `tensorflow-gpu` is preferred, please update `requirements.txt`'s first line from `-r NDN3/requirements.txt` to `-r NDN3/requirements-gpu.txt`

### Data
- Uses data from [Model Constrained by Visual Hierarchy Improves Prediction of Neural Responses to Natural Scenes](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927) paper.
