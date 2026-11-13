# Edge checklist: Raspberry Pi 5 (UNTESTED on hardware)

Nothing in this file has been run on a Raspberry Pi or any other board. No speed or accuracy on a
Pi is claimed anywhere in this project. The only speed numbers are from a desktop CPU
(`runs/<name>/benchmark_cpu.json`), which is a rough proxy.

Detection only. The output is a CSV log. Never connect it to a sprayer, solenoid or any actuator.

## Before going to the board

- [ ] Model exported on the PC: `python -m rwd.export --weights runs/<name>/weights/best.pt`
- [ ] Export checked against the `.pt` model on val (notebook `05_export_infer.ipynb`, section 3)
- [ ] Copy `best_ncnn_model/` (or `best.onnx`) and this repo to the board. No dataset images are needed.

## On the board (to be done, untested)

- [ ] 64-bit Raspberry Pi OS, Python 3.9 or newer, a project venv (`bash scripts/setup_env.sh`)
- [ ] `pip install -r requirements-export.txt` plus `ultralytics` inside the venv. Check that `ncnn` has an aarch64 wheel.
- [ ] `python -m rwd.benchmark --weights best_ncnn_model best.onnx --images <folder> --n 20`
      and record: board model, RAM, cooling, power supply, OS version, median ms, peak memory.
- [ ] Watch for thermal throttling over a 10 minute run (`vcgencmd measure_temp`, `vcgencmd get_throttled`).
- [ ] `python -m rwd.infer --weights best_ncnn_model --source <folder> --out detections.csv` and compare
      the CSV with the one from the PC for the same images.
- [ ] If the full 960x1280 input is too slow: export a smaller size (`--imgsz 480 640`) and measure accuracy
      again on val before using it. A smaller input is a different model as far as accuracy goes.

## What a result from the board may and may not say

- May say: latency and memory of this model on that exact board and setup.
- May not say: anything about accuracy in the field, on a drone, or in Thai rice fields.
