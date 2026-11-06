# Makefile — Master Orchestration & Reproducibility Commands
# Run: make all to reproduce the complete pipeline and verification suite

PYTHON := python
VENV_PYTHON := ./venv/Scripts/python.exe
STREAMLIT := streamlit

.PHONY: all pipeline extended figures tables test app clean help

help:
	@echo "Available commands:"
	@echo "  make all        - Run complete end-to-end pipeline, extension, and tests"
	@echo "  make pipeline   - Run original 5-phase research pipeline (run_all.py)"
	@echo "  make extended   - Run new audit extension modules (run_extended.py)"
	@echo "  make figures    - Regenerate all 8 publication figures (PDF + PNG)"
	@echo "  make tables     - Export publication tables to results/tables/ (.tex & .csv)"
	@echo "  make test       - Run full 32-test pytest verification suite"
	@echo "  make app        - Launch Streamlit research platform demo (localhost:8501)"
	@echo "  make clean      - Clean python caches and temporary logs"

all: pipeline extended tables figures test
	@echo "All research benchmarks, tables, figures, and tests successfully completed."

pipeline:
	$(PYTHON) run_all.py

extended:
	$(PYTHON) run_extended.py

figures:
	$(PYTHON) src/viz/generate_figures.py

tables:
	$(PYTHON) scripts/export_tables.py

test:
	$(PYTHON) -m pytest tests/ -v

app:
	$(STREAMLIT) run app/Home.py

clean:
	rm -rf __pycache__ .pytest_cache src/__pycache__ tests/__pycache__
