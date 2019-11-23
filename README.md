# msc-neuro

### Setup:
0. Hydrate submodules: `git submodule update --init --recursive-`
1. Build virtenv + deps: `make req-dev` (interactive dev) | `make env` (just script deps)
2. Activate virtenv: `source env/bib/activate`
3. Run scripts, start `jupyter notebook`, ...

> Note: If `tensorflow-gpu` is preffered, please update `requirements.txt`'s first line from `-r NDN3/requirements.txt
` to `-r NDN3/requirements-gpu.txt
`

### Data
- Data are currently not publicly available.
