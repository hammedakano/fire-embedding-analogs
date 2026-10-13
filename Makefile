.PHONY: install data reproduce test notebooks clean

RECORD ?=

install:
	pip install -e ".[dev]"

data:
	python scripts/download_data.py $(if $(RECORD),--record $(RECORD),)

reproduce:
	python scripts/reproduce.py

test:
	pytest -q

notebooks:
	cd notebooks && for nb in 0*.ipynb; do \
		jupyter nbconvert --to notebook --execute "$$nb" --output "$${nb%.ipynb}.executed.ipynb" || exit 1; \
	done

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info src/*.egg-info notebooks/*.executed.ipynb
