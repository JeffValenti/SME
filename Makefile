.PHONY: test docs

test:
	pytest -v --cov=sme --cov-report html
	open htmlcov/index.html

docs:
	$(MAKE) -C docs html
	$(MAKE) -C docs html
	open docs/_build/html/index.html

