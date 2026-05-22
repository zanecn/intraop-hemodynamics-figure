import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

np.random.seed(42)

t = np.linspace(-70, 255, 1200)


def smooth(x, w=15):
    return np.convolve(x, np.ones(w) / w, mode='same')


def make_map(t):
    base = 82 + 6 * np.sin(t / 30) + np.random.randn(len(t)) * 4
    base[t > 85] -= 5
    base[(t > 125) & (t < 155)] += 20
    base[t > 155] -= 15
    base[t > 215] -= 8
    return smooth(np.clip(base, 45, 145), 8)


def make_hr(t):
    base = 88 + 5 * np.sin(t / 25) + np.random.randn(len(t)) * 5
    base[(t > -10) & (t < 10)] += 18
    base[(t > 40) & (t < 60)] += 12
    base[t > 150] -= 15
    return smooth(np.clip(base, 55, 125), 8)


def make_spo2(t):
    s = 98 + np.random.randn(len(t)) * 0.3
    s[(t > 30) & (t < 55)] -= 18
    s[(t > 55) & (t < 80)] -= 8
    s[(t > 150) & (t < 165)] -= 7
    s[(t > 215) & (t < 235)] -= 5
    return smooth(np.clip(s, 82, 100), 6)


def make_svo2(t):
    s = np.full(len(t), np.nan)
    mask = t > 60
    base = 72 + np.random.randn(mask.sum()) * 4
    base[t[mask] > 130] -= 12
    base[t[mask] > 150] -= 6
    s[mask] = smooth(np.clip(base, 55, 85), 10)
    return s


def make_mpap(t):
    base = 27 + np.random.randn(len(t)) * 2
    base[(t > -5) & (t < 5)] += 2
    base[(t > 85) & (t < 135)] += 3
    base[(t > 130) & (t < 155)] += 4
    base[t > 155] -= 10
    return smooth(np.clip(base, 10, 38), 10)


def make_cvp(t):
    base = np.full(len(t), np.nan)
    mask = t > 5
    b = 13 + np.random.randn(mask.sum()) * 2
    b[t[mask] > 85] += 5
    b[t[mask] > 155] -= 8
    b[t[mask] > 215] -= 4
    base[mask] = smooth(np.clip(b, -2, 32), 10)
    return base


def make_co(t):
    c = np.full(len(t), np.nan)
    mask = t > 80
    b = 6.5 + np.random.randn(mask.sum()) * 0.2
    b[t[mask] > 130] -= 2.2
    b[(t[mask] > 138) & (t[mask] < 145)] += 1.5
    b[t[mask] > 145] -= 0.3
    c[mask] = smooth(np.clip(b, 2, 7.2), 12)
    return c


MAP  = make_map(t)
HR   = make_hr(t)
SPO2 = make_spo2(t)
SVO2 = make_svo2(t)
MPAP = make_mpap(t)
CVP  = make_cvp(t)
CO   = make_co(t)
CI   = np.where(~np.isnan(CO), CO / 1.7, np.nan)

# Surgery phase definitions: name -> (x_start, x_end, fill_color)
phases = {
    'Ligation':  (85,  130, '#f5e6c8'),
    'Resection': (130, 155, '#f5cccc'),
    'Closure':   (155, 215, '#d6ecd6'),
    'Emergence': (215, 260, '#e8d6f0'),
}
phase_border = {
    'Ligation':  '#c8a96e',
    'Resection': '#c87878',
    'Closure':   '#7ab87a',
    'Emergence': '#a87ac8',
}

# Event annotations: label -> (x_position, color)
events = {
    'Induction /\nIntubation': (0,   '#4a90d9'),
    'Tumor\nremoved':          (145, '#c0392b'),
    'Extubation':              (222, '#8e44ad'),
}

# Font sizes
AXIS_FONTSIZE   = 11   # axis labels & x-label
TICK_FONTSIZE   = 10   # tick labels
LEGEND_FONTSIZE = 8.5
PHASE_LABEL_FS  = 9    # italic phase labels just above x-axis
EVENT_LABEL_FS  = 9    # bold event-box labels (panel A only)

fig, axes = plt.subplots(4, 1, figsize=(11, 10.5), sharex=True)
fig.subplots_adjust(hspace=0.08, top=0.97, bottom=0.07, left=0.1, right=0.97)


def shade_phases(ax, show_labels=False):
    """Fill surgery-phase bands; draw subtle vertical boundary lines.
    If show_labels=True, place italic phase names just above the x-axis."""
    for name, (x0, x1, col) in phases.items():
        ax.axvspan(x0, x1, color=col, alpha=0.55, zorder=0, linewidth=0)
        ax.axvline(x=x0, color=phase_border[name], linewidth=1.1,
                   linestyle='-', alpha=0.85, zorder=1)
    if show_labels:
        trans = ax.get_xaxis_transform()   # x = data coords, y = axes fraction
        for name, (x0, x1, col) in phases.items():
            xc = (x0 + x1) / 2
            ax.text(xc, 0.04, name,
                    transform=trans, ha='center', va='bottom',
                    fontsize=PHASE_LABEL_FS, fontstyle='italic',
                    color='#555555', zorder=5)


def add_event_lines(ax):
    """Dashed vertical lines for each surgical event — all panels."""
    for label, (x, col) in events.items():
        ax.axvline(x=x, color=col, linewidth=1.1, linestyle='--',
                   alpha=0.85, zorder=2)


def add_event_annotations(ax):
    """Bold boxed labels for surgical events — panel A only."""
    y_top   = ax.get_ylim()[1]
    yrange  = ax.get_ylim()[1] - ax.get_ylim()[0]
    for label, (x, col) in events.items():
        ax.annotate(
            label,
            xy=(x, y_top - 0.01 * yrange),
            xytext=(x + 2, y_top - 0.01 * yrange),
            ha='left', va='top',
            fontsize=EVENT_LABEL_FS,
            fontweight='bold',
            color=col,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=col,
                      alpha=0.90, linewidth=1.2),
            zorder=6,
        )


def style_ax(ax):
    """Common axis styling: no top/right spine, horizontal gridlines only,
    larger tick labels."""
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.yaxis.grid(True, color='#cccccc', linewidth=0.6, linestyle='-', alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


# ── Panel A: MAP / HR ──────────────────────────────────────────────────────
ax = axes[0]
shade_phases(ax)
add_event_lines(ax)
ax.plot(t, MAP, color='#b03030', linewidth=1.4, label='MAP (mmHg)', zorder=3)
ax.plot(t, HR,  color='#d4860a', linewidth=1.2, linestyle='--', label='HR (bpm)', zorder=3)
ax.set_ylabel('MAP / HR\n(mmHg / bpm)', fontsize=AXIS_FONTSIZE)
ax.set_ylim(40, 150)
ax.legend(loc='lower left', fontsize=LEGEND_FONTSIZE, framealpha=0.9,
          handlelength=1.5, borderpad=0.4)
style_ax(ax)
add_event_annotations(ax)   # boxed bold labels in panel A only

# ── Panel B: SpO2 / SvO2 ──────────────────────────────────────────────────
ax = axes[1]
shade_phases(ax)
add_event_lines(ax)
ax.plot(t, SPO2, color='#2980b9', linewidth=1.4, label='SpO\u2082 (%)', zorder=3)
ax.plot(t, SVO2, color='#d4860a', linewidth=1.2, linestyle='-.', label='SvO\u2082 (%)', zorder=3)
ax.set_ylabel('O\u2082 sat (%)', fontsize=AXIS_FONTSIZE)
ax.set_ylim(50, 104)
ax.legend(loc='lower left', fontsize=LEGEND_FONTSIZE, framealpha=0.9,
          handlelength=1.5, borderpad=0.4)
style_ax(ax)

# ── Panel C: mPAP / CVP ───────────────────────────────────────────────────
ax = axes[2]
shade_phases(ax)
add_event_lines(ax)
ax.plot(t, MPAP, color='#7b3fa0', linewidth=1.4, label='mPAP (mmHg)', zorder=3)
ax.plot(t, CVP,  color='#555555', linewidth=1.2, linestyle='--', label='CVP (mmHg)', zorder=3)
ax.set_ylabel('Pressure\n(mmHg)', fontsize=AXIS_FONTSIZE)
ax.set_ylim(-3, 44)
ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE, framealpha=0.9,
          handlelength=1.5, borderpad=0.4)
style_ax(ax)

# ── Panel D: CO / CI  (phase labels shown just above x-axis) ──────────────
ax = axes[3]
shade_phases(ax, show_labels=True)
add_event_lines(ax)
ax.plot(t, CO, color='#1a7a3a', linewidth=1.6, label='CO (L/min)', zorder=3)
ax.plot(t, CI, color='#1a7a3a', linewidth=1.2, linestyle='--',
        label='CI (L/min/m\u00b2)', zorder=3)
ax.set_ylabel('CO / CI', fontsize=AXIS_FONTSIZE)
ax.set_ylim(1.5, 7.5)
ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE, framealpha=0.9,
          handlelength=1.5, borderpad=0.4)
style_ax(ax)

axes[3].set_xlabel('Time from induction (min)', fontsize=AXIS_FONTSIZE)
axes[3].set_xlim(-70, 260)

fig.text(0.01, 0.99, 'Figure 1 \u2014 Intraoperative Hemodynamics  (v17 spec)',
         va='top', ha='left', fontsize=10, fontweight='bold')

os.makedirs('output', exist_ok=True)
fig.savefig('output/figure1_hemodynamics.png', dpi=180, bbox_inches='tight',
            facecolor='white')
plt.close()
print('Saved to output/figure1_hemodynamics.png')
