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
3. Work
    - Run experiments: `python3 experiments/experiments_3/bs4_exp3_runner.py qsub-cpu`
    - Run single experiment instance: `python3 ./experiments/baseline/bl1.py`
    - Start jupyter notebook with analysis: `jupyter notebook` and navigate to `./playgrounds/`

> Note: If `tensorflow-gpu` is preferred, please update `requirements.txt`'s first line from `-r NDN3/requirements.txt` to `-r NDN3/requirements-gpu.txt`

### Data
- Uses data from [Model Constrained by Visual Hierarchy Improves Prediction of Neural Responses to Natural Scenes](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927) paper.

### Structure: 
- `./Data/`: Used dataset.
- `./experiments/`: Experiments scripts.
    - `bsX_expY.py`: A script defining the architecture of experiment `Y` that's based on baseline `X` model.
    - `bsX_expY_runner.py`: A runner script defining what instances (with different parameters) of respective experiment are to be tested.
- `./NDN3/`: A git submodule with the used neuroscience focused ML framework.
- `./playgrounds/`: Jupyter notebooks containing analysis.
- `./training_data/`: Data produced by running the experiments.
    - `job_logs/`: Standard output and standard error of experiment runs invoked through runners.
    - `logs/`: Tensorflow summaries gathered during experiments.
    - `models/`: Saved NDN3 models.
    - `experiments.txt`: A list of experiment instances names.
- `./utils/`: Various helper utilities.
    - `analysis_*`: Analysis utilities.
    - `runners.py`: Experiments execution pipeline.

> Note: This is not a software engineering thesis and so the focus was not on creating a properly extensible and maintainable experiments pipeline.