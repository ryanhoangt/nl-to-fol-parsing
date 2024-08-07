# NL to FOL Parsing

## Setup

1. Create a new conda environment and install the required packages.

```bash
conda create -n nl2fol-local python=3.8
conda activate nl2fol-local
pip install -r requirements.txt
```

2. Setup the environment variables.

Create a `.env` file in the root directory based on the `.env.template` file.

## Usage

Use the notebook `translate.ipynb` to translate natural language to first-order logic.

## Metrics

<!-- TODO: -->

## Tests

```bash
python3 -m unittest tests/*.py
```
