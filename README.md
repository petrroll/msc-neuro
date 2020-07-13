# msc-neuro

### Setup:
0. Hydrate submodules: `git submodule update --init --recursive`
1. Build virtenv + deps: `make env-dev` (interactive dev) | `make env` (just script deps)
2. Activate virtenv: `source ./activate`
3. Run scripts, start `jupyter notebook`, ...

> Note: If `tensorflow-gpu` is preferred, please update `requirements.txt`'s first line from `-r NDN3/requirements.txt` to `-r NDN3/requirements-gpu.txt`

### Data
- Uses data from [Model Constrained by Visual Hierarchy Improves Prediction of Neural Responses to Natural Scenes](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927) paper.
