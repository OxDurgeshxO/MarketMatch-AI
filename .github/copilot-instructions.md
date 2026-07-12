# Copilot Cloud Agent Onboarding Instructions

## Repository snapshot
- This repository is a single-script Python project for customer segmentation and similarity-based recommendations.
- Main executable file: `/home/runner/work/MarketMatch-AI/MarketMatch-AI/Mall Customers Project` (no `.py` extension).
- Key documentation: `/home/runner/work/MarketMatch-AI/MarketMatch-AI/README.md`.
- The repository currently includes generated plot images in the root.

## How to work efficiently here
- Treat this as a script-first repo, not a package/module-based codebase.
- Keep changes focused and minimal; most edits will be in the main script and README.
- Always quote the main script filename in shell commands because it contains spaces.
- Use absolute paths when referencing files in automation and agent outputs.

## Environment setup
From repository root (`/home/runner/work/MarketMatch-AI/MarketMatch-AI`):

1. Install dependencies (no requirements file exists yet):
   - `pip install pandas numpy matplotlib seaborn scikit-learn plotly`

2. Provide dataset file in repo root:
   - Required filename: `Mall_Customers.csv`
   - Source referenced by repo README: Kaggle Mall Customers CSV dataset

## Run commands
- Execute script:
  - `python "Mall Customers Project"`

- Optional syntax-only check:
  - `python -m py_compile "Mall Customers Project"`

## Validation expectations
- There is no dedicated test suite, linter config, or build pipeline in this repository.
- For code changes, validation is typically:
  1. syntax check (`py_compile`)
  2. script execution with dataset available
  3. quick sanity check that expected CSV/image outputs are produced

## Known errors encountered during onboarding and workarounds
1. Error:
   - `ModuleNotFoundError: No module named 'pandas'`
   Workaround:
   - Install required Python libraries with:
     `pip install pandas numpy matplotlib seaborn scikit-learn plotly`

2. Error:
   - `FileNotFoundError: [Errno 2] No such file or directory: 'Mall_Customers.csv'`
   Workaround:
   - Download/add `Mall_Customers.csv` to repository root before running the script.

## Agent tips for future tasks
- If asked to improve reproducibility, first add `requirements.txt` and document dataset placement.
- If asked for modularization, split the monolithic script into functions/modules while preserving current output behavior.
- Avoid renaming `Mall Customers Project` unless the task explicitly requests file renaming, since existing docs reference it.
