# Based on: https://blog.horejsek.com/makefile-with-python/
VENV_NAME?=env
PYTHON=$(VENV_NAME)/bin/python
PYTHON_GLOBAL=python3

# ENV & deps
$(VENV_NAME): requirements.txt
	@test -d $(VENV_NAME) || $(PYTHON_GLOBAL) -m venv $(VENV_NAME)
	$(PYTHON) -m pip install -r requirements.txt

.PHONY: req-dev
req-dev: requirements-dev.txt $(VENV_NAME)
	$(PYTHON) -m pip install -r requirements-dev.txt

.PHONY: clean_$(VENV_NAME)
clean_$(VENV_NAME):
	rm -rf $(VENV_NAME)

# Common:
.PHONY: clean
clean: clean_$(VENV_NAME)
