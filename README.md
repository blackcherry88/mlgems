# ML Gems

A personal collection of small, readable machine-learning implementations.

The goal is to understand the essential idea behind an algorithm without
hiding it behind a large framework. Once the basic implementation is clear,
the same pattern should be easy to adapt to different models, tensor shapes,
and use cases.

## Learning approach

Each topic should:

1. Start with the simplest correct implementation.
2. Add alternative or optimized implementations one step at a time.
3. Test every variant against the reference implementation.
4. Stay small enough to read, experiment with, and port elsewhere.

These examples favor clarity and learning over production-level performance.

## Repository structure

```text
mlgems/
├── src/
│   └── flashattention/
│       ├── attention.py
│       ├── attention_nb.ipynb
│       └── test_attention.py
├── .python-version
├── requirements.txt
├── LICENSE
└── README.md
```

Each directory under `src/` is a self-contained learning topic:

- `attention.py` contains the implementations.
- `attention_nb.ipynb` is a workspace for interactive exploration.
- `test_attention.py` checks correctness and equivalence between variants.

This lightweight layout is appropriate for a snippet collection. If the
repository later becomes an installable Python library, packaging metadata
and package-level `__init__.py` files can be added then.

## Current gems

### Attention

The `src/flashattention/` example builds scaled dot-product attention in three
stages:

- `basic_attention`: a direct and readable reference implementation.
- `einsum_attention`: the same computation expressed with Einstein notation.
- `stream_attention`: a block-wise, numerically stable implementation that
  avoids materializing the complete attention matrix.

The streaming version is educational Python/PyTorch code, not a replacement
for a production FlashAttention kernel.

## Setup

This repository uses
[pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv). The checked-in
`.python-version` selects a virtual environment named `mlgems`, so pyenv will
activate it automatically when you enter the repository.

Create the environment once:

```bash
pyenv install 3.12.13
pyenv virtualenv 3.12.13 mlgems
```

If that Python version or virtual environment already exists, pyenv will say
so and you can continue. From the repository root, confirm that the
`.python-version` file selected the correct environment:

```bash
pyenv version
# mlgems (set by .../mlgems/.python-version)
```

Then install the common dependencies:

```bash
python -m pip install -r requirements.txt
```

After this one-time setup, entering the repository activates `mlgems` and the
examples, tests, and notebooks can be run as-is.

## Run the tests

From the repository root:

```bash
python -m unittest discover -s src/flashattention -p "test_*.py" -v
```

## Adding a new gem

Use one directory per concept:

```text
src/<topic>/
├── <topic>.py
├── <topic>_nb.ipynb
└── test_<topic>.py
```

Begin with a transparent baseline, add more sophisticated variants, and use
tests to keep every implementation anchored to the same behavior.
