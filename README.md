# msc-neuro
Set of scripts to facilitate experiments done as part of an [AI/ML Msc. thesis](https://github.com/petrroll/msc-thesis) at the Math and Physics Faculty of Charles University in Prague.

### Abstract
Accurate models of visual processing are key tools for sensory neuroscience. In this thesis, we explore novel deep learning architectures targeting population data recorded from mammalian primary visual cortex when presented with natural images as stimuli. 

We reimplement a prior model, assessing it in terms of stability and sensitivity to hyperparameters and architecture fine-tuning in an existing neuroscience focused deep learning framework NDN3. We proceed to extend the model with components of various DNN models, analyzing novel combinations and ideas from classical computer vision deep learning. 

We were able to identify modifications that secure greater stability of the model. Furthermore, we document the importance of small hyperparameters adjustments versus architectural advantages that could facilitate further experiments with the examined architectures. All new model componentes were contributed to the open-source NDN3 package, thus enabling rapid experimentation with new techniques in future. 

### Setup:
0. Hydrate submodules: `make NDN3`
1. Build virtenv + deps: `make env-dev` (analysis dev) | `make env` (just script deps)
2. Activate virtenv: `source ./activate`
3. Work:
    - Run an experiment: e.g.: `python3 experiments/experiments_3/bs4_exp3_runner.py qsub-cpu`
    - Run a single experiment instance directly: e.g: `python3 ./experiments/baseline/bl1.py`
    - Use jupyter notebook for results analysis: `jupyter notebook` and navigate to `./playgrounds/`

> Note: If `tensorflow-gpu` is preferred, please update `requirements.txt`'s first line from `-r NDN3/requirements.txt` to `-r NDN3/requirements-gpu.txt`

### Data
- Uses data from [Model Constrained by Visual Hierarchy Improves Prediction of Neural Responses to Natural Scenes](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927) paper.
    - Navigate to Supporting Information.
    - Download the first supplement.
    - Unzip it to `./`.

### Experiments pipeline: 
An experiment is a set of tested architectures + hyperparameters.
- Identified by `{exp}` name and `{exp_folder}` group. 
    - Both are used throughout the whole system for scripts discovery, logs naming, ... .
    - `{exp_folder}` is used just for namespacing experiments, having two `{exp}`s together in one `{exp_folder}` versus in separate ones has no impact on their function.
    - `{exp}` usually in the form of `bsX_expY.py`: An experiment `Y` based on baseline `X` model.
- Single experiment is defined by `{exp}.py` script and optionally a runner `{exp}_runner.py` (naming can be arbitrary for runner).
- `{exp}.py` 
    - Defines the architecture and includes all code to instantiate and train the model.
    - Usually trains the same model multiple times (see the loop) to control for random init (`repetitions`).
    - Sometimes accepts hyperparameters as command line arguments to support testing multiple variations as part of one experiment.
    - To test multiple variations it should be called multiple times with different command line arguments. Each execution with a particular set is called a `run`.
    - Also defines `{name}` that shows on logs file paths, etc. Should serve as an human interpretable identifier of both the `exp` and also the particular `run` (i.e. encode passed args).
- `{exp}_runner.py`
    - Runs an experiment through executing the `{exp}.py` script with all args combinations we want to test. I.e. executes all runs of an experiment.
    - Executes individual runs through environment specific runners. Runs usually work in parallel (e.g. as individual jobs).

> In the thesis text `repetitions` are called runs, and `runs` are simply experiment instances. The difference in naming is an unfortunate relict of an attempt to have understandable text but also backwards compatible scripts.

### Repo structure: 
- `./Data/region{x}`: Used dataset.
- `./experiments/`: Experiments scripts.
    - `{exp_folder}/{exp}.py`: A script defining the architecture of an experiment.
    - `{exp_folder}/{exp}_runner.py`: A runner script defining which runs (with what parameters) of respective experiment are to be tested.
- `./NDN3/`: A git submodule with the used neuroscience focused ML framework.
- `./playgrounds/`: Jupyter notebooks containing analysis.
- `./training_data/`: Data produced by running the experiments.
    - `job_logs/{exp_folder}/{exp}/(e|o)_{repetition}.log`: Standard output and standard error of experiment runs invoked through runners.
    - `logs/{exp_folder}x{run}/{name}__{repetition}/`: Tensorflow summaries gathered during training.
    - `models/{exp_folder}x{run}/{name}__{repetition}.ndnmod`: NDN3 models saved at the end of training.
    - `experiments.txt`: A list of finished `{exp_folder}/{exp}/{name}`s.
- `./utils/`: Various helper utilities.
    - `analysis_*`: Analysis utilities.
    - `runners.py`: Experiments execution pipeline.

> Note: This is not a software engineering thesis and so the focus was not on creating a properly extensible and maintainable experiments pipeline.