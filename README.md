# Intraoperative Hemodynamics Figure

Reproducible Python/Matplotlib figure of intraoperative hemodynamics (v17 spec).

## What it plots

| Panel | Signals |
|---|---|
| A | MAP (mmHg) and HR (bpm) |
| B | SpO₂ (%) and SvO₂ (%) |
| C | mPAP (mmHg) and CVP (mmHg) |
| D | CO (L/min) and CI (L/min/m²) |

## Design features

- Surgery phase bands (Ligation, Resection, Closure, Emergence) shaded with pastel fills
- Subtle **darker vertical lines** at each phase boundary, color-matched to the phase
- Phase labels positioned just above the x-axis in Panel D (via `ax.get_xaxis_transform()`)
- **Bold** event text annotations (Induction/Intubation, Tumor removed, Extubation)
- Dashed vertical event lines in event colors

## Usage

```bash
pip install matplotlib numpy
python figure1_hemodynamics.py
# Output saved to output/figure1_hemodynamics.png
```

## Requirements

- Python 3.8+
- matplotlib
- numpy
