# TCR-Embeddings

We evaluate the difference in expressivity of physico-chemical properties embeddings and large language model embeddings in the context of T cell receptors.

## Installation Instructions

1. conda create --name tcr_embeddings python=3.12
2. conda activate tcr_embeddings
3. python -m pip install poetry
4. poetry install
5. python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

## CI/CD

1. black (code formatting)
2. isort (import sorting)
3. flake8 (linting)
4. mypy (type checking)
5. unittests