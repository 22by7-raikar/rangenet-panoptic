# Development

## Environment

Python 3.10 or newer and PyTorch 2.1 or newer. CPU is enough for tests; train
on a CUDA GPU.

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install ruff                    # lint
sudo apt install doxygen graphviz   # docs (macOS: brew install doxygen graphviz)
```

## Tests

```sh
cd train/tasks/panoptic && python -m pytest tests
```

| File | Covers |
|---|---|
| `test_dataset.py` | projection keeps the closest point, instance targets, item contract, collate |
| `test_model.py` | point pyramid, trilinear weights (normalised, exact matches, wrap-around, fallbacks, gradient), forward/backward, transpose and lower output stride, RangeNet++ checkpoint loading |
| `test_losses_cluster_metrics.py` | zero losses for perfect predictions, repel behaviour, clustering separates objects and classes, radius modes, PQ on hand-computed cases |
| `test_tools.py` | Euclidean clustering, scan and label selection when fetching, every tool end to end (`baseline`, `upper_bounds`, `benchmark`, `visualize`, `report`) |
| `test_pipeline.py` | train, checkpoint, resume, `infer.py`, `evaluate.py` end to end; a one-batch overfit |

All tests use `tests/synthetic.py`, which writes small SemanticKITTI-format
scans (road, wall, cars, persons) into a temporary directory. No dataset or GPU
needed; the suite takes about 2 minutes on a laptop CPU.

The semantic task has a smoke test, `train/tasks/semantic/tests/test_smoke.py`.
It trains one epoch, infers and evaluates with the task's own scripts on the
same synthetic scans, and needs the full `requirements.txt`:

```sh
python -m pytest train/tasks/semantic/tests      # about 30 s on CPU
```

## Lint

```sh
ruff check train/tasks/panoptic train/tasks/semantic/tests train/common/module_loader.py
```

Configuration is in `pyproject.toml`. Lint covers new code only. The upstream
semantic task keeps its own style (see
[design decision 3](design-decisions.md#3-keep-the-upstream-layout-and-code-style-of-the-semantic-task)).

## Documentation

```sh
make -C docs html          # diagrams + API reference -> docs/_build/html/index.html
```

- Markdown pages: `README.md` (also the Doxygen main page) and `docs/*.md`.
  They're written for GitHub first; Doxygen renders the same files.
- API reference: Doxygen reads the Python docstrings (`PYTHON_DOCSTRING = NO`,
  so Markdown and Doxygen commands in docstrings are rendered).
  - Every module, class and function in linted code has a docstring; ruff's
    `D1xx` rules enforce it. `__init__` arguments are documented on the class.
  - Start with a one-line summary. Use `@param` and `@return` where arguments
    aren't obvious, and give tensor shapes, e.g. `(B, 3, H, W)`.
  - Each module docstring ends with `@ingroup <subsystem>`. The groups (data,
    model, losses, inference, evaluation, training, tools, cli) are defined
    in `docs/groups.dox` and appear as *Topics* in the HTML, with diagrams.
- Code walkthrough: `docs/code-walkthrough.md`. Update it when a step of the
  pipeline changes.
- Figures: `docs/figures/README.md` lists the command for every image; rerun
  them after changes that affect the figures.
- Warnings go to `docs/_build/doxygen-warnings.log`; CI fails if it isn't
  empty.

### Diagrams

Diagrams are Graphviz sources in `docs/diagrams/*.dot`. After editing one:

```sh
make -C docs diagrams      # re-renders the changed *.svg; commit both files
```

Keep the shared style: header comment naming the code it describes,
`Helvetica`, rounded filled boxes, one colour per subsystem (blue: shared
encoder and geometry, green: semantic, red: instance/panoptic, amber: files and
outputs, purple: analysis tools).

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

1. **test**: panoptic unit and end-to-end tests on CPU (PyTorch CPU wheels).
2. **lint**: `ruff check`.
3. **semantic-smoke**: the upstream semantic task trains one epoch, infers and
   evaluates on synthetic scans.
4. **docs**: render diagrams from source, build Doxygen, fail on any
   warning, upload the HTML as an artifact.
5. **deploy-docs** (pushes to `master` only): publishes the docs to GitHub
   Pages. To switch it on: *Settings → Pages → Source: GitHub Actions*, then
   add the repository variable `DEPLOY_DOCS` = `true` (*Settings → Secrets
   and variables → Actions → Variables*).

## Conventions

- **Pull requests** are complete units of work: code, tests and documentation
  for one coherent change, with the testing that was done described in the
  PR. Avoid small follow-up PRs; collect related changes on one branch.
- **Branches** start from the latest `master`. A merged branch isn't reused
  for new work.
- **Configuration**: new behaviour is a config key with a default matching
  the paper where one exists. Deviations from the paper are documented in
  [design-decisions.md](design-decisions.md) and in the panoptic README.
- **Checkpoints** embed the resolved config, so inference must not depend on
  the config file on disk.
- **Experiments**: record the config overrides, commit hash and validation
  metrics. The results table in the panoptic README lists reproduced numbers.

## Adding a model or head

The data pipeline (`dataset.py`), evaluation (`metrics.py`) and training loop
(`trainer.py`) don't depend on the head. A new panoptic head, such as a
query-based mask head, needs:

1. a module returning the outputs the loss and inference need;
2. its loss, alongside `LPSADLoss`;
3. an inference function producing per-point `(class, instance)` like
   `panoptic_inference`;
4. a config switch and tests at the level of `test_model.py` and
   `test_pipeline.py`.
