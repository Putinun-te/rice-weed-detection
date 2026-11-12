# Rice-Weed Detection: instructions for Claude

Read these first: [docs/HANDOFF.md](docs/HANDOFF.md) (current state and next steps),
[docs/PLAN.md](docs/PLAN.md), [DATASET.md](DATASET.md), [README.md](README.md).

## Working with the owner

- Explain decisions in Thai, in short paragraphs. The owner reads code better than they write it.
- Keep commits small. Stop at each phase checkpoint and wait for an explicit OK.
- Ask before anything hard to undo or outward-facing.

## Hard rules

- No secrets in code, git, notebooks or logs. Keys (Roboflow, Kaggle) come from environment
  variables only; see `.env.example`. Never open `.env` files.
- No `git push`, PR, GitHub Release or cloud resource without the owner's explicit OK.
  Commit locally.
- Never commit `data/`, `runs/`, images, `*.pt`, `*.onnx`, `*.zip` (covered by `.gitignore`).
- Do not publish or quote any accuracy number unless it comes from the held-out split by
  site/date (not random images), and always report recall with it.
- Detection only. Never connect the model to a sprayer, solenoid or any actuator.
  The old course scripts treated any COCO detection as "weed"; that is unsafe and not reused.
- Record source, license and attribution of every external dataset in `DATASET.md`.
  Do not commit third-party images. Never describe an external dataset as the owner's own data.
- The old course model and its numbers are not reused.

## Environment

- Use the project venv only: `bash scripts/setup_env.sh`, then `.venv/bin/python`.
  Never install into a system or shared env (on Studio Lab that broke `amzn-sagemaker-scheduler`).
- Code must run on Python 3.9+ (Studio Lab is 3.9): keep `from __future__ import annotations`.
- Run tests with `.venv/bin/python -m pytest -q` (synthetic data only).
- On WSL keep everything under `~/`, never under `/mnt/c` (slow file access).
