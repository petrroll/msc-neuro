# Based on: https://blog.horejsek.com/makefile-with-python/
VENV_NAME?=env
PYTHON=$(VENV_NAME)/bin/python
PYTHON_GLOBAL=python3.7

# ENV & deps
.PHONY: $(VENV_NAME)-dev
$(VENV_NAME)-dev: requirements-dev.txt $(VENV_NAME)
	$(PYTHON) -m pip install -r requirements-dev.txt

$(VENV_NAME): requirements.txt activate
	@test -d $(VENV_NAME) || $(PYTHON_GLOBAL) -m venv $(VENV_NAME)
	$(PYTHON) -m pip install -r requirements.txt

activate:
	echo "source "$(VENV_NAME)"/bin/activate" > activate

.PHONY: Dockerfile
Dockerfile:
	docker build -t houska/mscneuro .

.PHONY: NDN3
NDN3:
	git submodule update --init --recursive

# Cleanup

.PHONY: clean_$(VENV_NAME)
clean_$(VENV_NAME): clean_activate
	rm -rf $(VENV_NAME)

.PHONY: clean_tflogs
clean_tflogs:
	rm -rf ./logs/

.PHONY: clean_activate
clean_activate:
	rm -f activate

# Common:
.PHONY: clean
clean: clean_$(VENV_NAME) clean_tflogs clean_activate
