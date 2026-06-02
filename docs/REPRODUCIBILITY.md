# Reproducibility

Recommended reporting checklist for experiments:

1. Record the git commit or release tag.
2. Record the Python, PyTorch, CUDA and GPU versions.
3. Use the scene-level split in `Config` unless explicitly running ablations.
4. Report event-level precision, recall, F1, mean IoU and mean latency.
5. Report whether incomplete lane-change events are included.
6. Archive the exact `configs/default.yaml` snapshot used by the paper run.

## Verification Commands

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe -m compileall src run.py train.py correct_dets.py
D:\pycharmproj\roslearn\.venv\Scripts\python.exe run.py --help
```

## Evaluation

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe run.py --split test
```
