# Notebooks

This directory is for interactive exploration. The pipeline itself lives in the
`swinvvc` package and the `scripts/` entry points.

Import the library rather than copying code out of it:

```python
import sys; sys.path.insert(0, "..")   # or pip install -e ..

from swinvvc.data.dtrace import build_boundary_maps
from swinvvc.data.visualize import render_label_map, render_wireframe
from swinvvc.data.yuv import get_padded_size

width, height = get_padded_size(416, 240, 32, 32)
maps = build_boundary_maps("traces/BlowingBubbles_qp22.csv", width, height)

render_wireframe(maps[0], width, height)
render_label_map(maps[0], width, height)
```

Useful entry points:

| Task | Module |
|---|---|
| inspect a CU boundary map or label map | `swinvvc.data.visualize` |
| check a class distribution | `swinvvc.data.balance`, `swinvvc.utils.plots` |
| evaluate a checkpoint | `swinvvc.training.evaluate` |
| confusion matrix and ROC figures | `swinvvc.utils.plots` |
| BD-BR, BD-PSNR, time saving, RD curves | `swinvvc.utils.bjontegaard`, `swinvvc.utils.plots` |

Clear outputs before committing a notebook:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```
