.PHONY: install test lint figures clean smoke

install:
	pip install -e ".[dev]"

test:
	pytest -q

smoke:
	python -c "import fireanalogs; print('fireanalogs', fireanalogs.__version__, 'imports cleanly')"

lint:
	ruff check src tests

figures:
	jupyter nbconvert --to notebook --execute notebooks/05_figures.ipynb --output 05_figures.executed.ipynb

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
