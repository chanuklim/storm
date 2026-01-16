# Repository Guidelines

## Project Structure & Module Organization
- Core Python package lives in `knowledge_storm/`: retrieval and LM adapters in `rm.py` and `lm.py`, shared interfaces in `interface.py`, wiki pipeline in `storm_wiki/`, and collaborative flow in `collaborative_storm/`.
- Example scripts are under `examples/`: `storm_examples/` for STORMWiki runners (per-provider scripts like `run_storm_wiki_gpt.py`, `..._ollama.py`), and `costorm_examples/` for collaborative runs.
- Frontend prototype is `frontend/demo_light/` (Streamlit UI) with its own `assets/`; repository-level `assets/` contains diagrams.
- Packaging files: `requirements.txt`, `setup.py`, `MANIFEST.in`; see `README.md` and `CONTRIBUTING.md` for usage and contribution scope.

## Build, Test, and Development Commands
- Create env & install deps:  
  ```bash
  conda create -n storm python=3.11
  conda activate storm
  pip install -r requirements.txt
  pip install -e .
  ```
- Run a STORMWiki example (requires provider keys, see `README.md`):  
  ```bash
  python examples/storm_examples/run_storm_wiki_gpt.py
  ```
- Run Co-STORM demo: `python examples/costorm_examples/run_costorm_gpt.py`.
- Launch Streamlit UI:  
  ```bash
  cd frontend/demo_light
  pip install -r requirements.txt
  streamlit run storm.py
  ```

## Coding Style & Naming Conventions
- Python code is formatted with `black`; install the pre-commit hook (`pip install pre-commit && pre-commit install`) to auto-format `knowledge_storm/` before commits.
- Prefer 4-space indentation, snake_case for functions/variables, CapWords for classes, and module names mirroring functionality (`*_rm.py`, `*_lm.py`).
- Keep example scripts self-contained and provider-specific; mirror existing filename patterns when adding providers or retrievers.

## Testing Guidelines
- No central automated test suite exists yet; validate changes by running the relevant example scripts and, for UI work, `streamlit run storm.py` with sample topics.
- When adding modules, include lightweight checks (e.g., `python -m knowledge_storm.rm` smoke snippets) or add `pytest` cases under a new `tests/` directory following the feature’s module path.
- Document any required API keys or mock strategies in the PR description so reviewers can reproduce.

## Commit & Pull Request Guidelines
- Follow the formats in `CONTRIBUTING.md`; suggested PR titles: `[New LM/New RM/Demo Enhancement] …`.
- Commits should be scoped and descriptive (e.g., `add serper retriever wiring`), with `black` run beforehand.
- PRs should describe usage, required keys, and include example inputs/outputs; attach screenshots for frontend changes and note any new environment variables (`secrets.toml` for Streamlit).
- The project currently prioritizes new LMs/RMs and demo_light improvements; avoid broad refactors unless pre-discussed.

## Security & Configuration Tips
- Keep API keys in environment variables or `secrets.toml` (ignored by Git); never commit credentials.
- Expose configurable parameters via runner arguments rather than hard-coding secrets.
