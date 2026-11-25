"""Runs needed by the random fraction (sweep_runs.jsonl): M_q interpolation, fit, figure, LaTeX table.
Usage: python3 sims/fig_sweep.py results/sweep_runs.jsonl figures"""
import json, sys, collections, numpy as np, matplotlib
from scipy.optimize import least_squares; from scipy.stats import norm
matplotlib.use('Agg'); import matplotlib.pyplot as plt
L = [json.loads(l) for l in open(sys.argv[1])]; OUT = sys.argv[2]
ch = collections.defaultdict(list)
for r in L: ch[(r['p'], r['s'], r['theta'])].append(r)
def m_at(rows, q):  # smallest M reaching recall q, interpolated linearly in log M between the bracketing points
    rows = sorted(rows, key=lambda r: r['M']); prev = None
    for r in rows:
        if r['recall'] >= q:
            if prev is None: return float(r['M'])
            x0, x1 = np.log(prev['M']), np.log(r['M']); y0, y1 = prev['recall'], r['recall']
            return float(np.exp(x0 + (q - y0) / (y1 - y0) * (x1 - x0)))
        prev = r
    return np.nan
Q = [.5, .9, .99]
D = np.array([(p, s, th, ch[(p, s, th)][0]['N']) + tuple(m_at(ch[(p, s, th)], q) for q in Q) for (p, s, th) in sorted(ch)])
P, S, TH, N = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
def model(a, q, s, N, th): return a * s * np.log(N / s) + (np.sqrt(2 * np.log(N)) + norm.ppf(q)) ** 2 / th ** 2
fit = {}
for j, q in enumerate(Q):
    y = D[:, 4 + j]; r = least_squares(lambda c: np.log(model(c[0], q, S, N, TH)) - np.log(y), [2.])
    e = model(r.x[0], q, S, N, TH) / y - 1
    fit[q] = (r.x[0], np.median(np.abs(e)), np.abs(e).max())
    print(f'q={q}: a={r.x[0]:.2f} z={norm.ppf(q):.2f} median |rel err|={np.median(np.abs(e)):.3f} max={np.abs(e).max():.3f}')
json.dump({str(q): dict(a=v[0], med=v[1], max=v[2]) for q, v in fit.items()}, open(f'{OUT}/sweep_fit.json', 'w'), indent=1)
# figure: M99 vs s, one panel per p, one colour per theta, measured points + fitted curve
C = {.25: '#2a78d6', .5: '#eb6834', 1.0: '#1baf7a'}; MK = {.25: 'o', .5: 's', 1.0: '^'}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8a85',
                     'xtick.color': '#5a5a56', 'ytick.color': '#5a5a56', 'grid.color': '#e4e3df', 'grid.linewidth': .6})
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
a99 = fit[.99][0]; ss = np.linspace(4, 64, 100)
for ax, p in zip(axs, [100, 300, 1000]):
    Np = p * (p + 1) // 2; ax.grid(True, axis='y')
    for th in [.25, .5, 1.0]:
        m = (P == p) & (TH == th)
        ax.plot(ss, model(a99, .99, ss, Np, th), color=C[th], lw=1.2, ls='--', zorder=2)
        ax.plot(S[m], D[m, 6], ls='none', marker=MK[th], ms=6, color=C[th], mec='white', mew=1, zorder=3,
                label=f'θ = {th:g}σ')
        ax.annotate(f'θ={th:g}σ', (64, D[m, 6][-1]), xytext=(4, 0), textcoords='offset points', va='center',
                    fontsize=7, color='#3a3a38', annotation_clip=False)
    ax.set_title(f'p = {p}  (N = {Np:,})'.replace(',', ' '), fontsize=8, color='#2a2a28')
    ax.set_xlabel('number of active effects s'); ax.set_xticks([4, 16, 32, 48, 64]); ax.set_xlim(0, 72)
axs[0].set_ylabel('runs for recall 0.99 in top s')
h, l = axs[0].get_legend_handles_labels()
h.append(plt.Line2D([], [], color='#5a5a56', ls='--', lw=1.2)); l.append('fitted formula')
fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.02))
fig.tight_layout(rect=(0, 0, .97, .92)); fig.savefig(f'{OUT}/sweep_m99.pdf'); fig.savefig(f'{OUT}/sweep_m99.png', dpi=110)
# LaTeX table rows
with open(f'{OUT}/sweep_table.tex', 'w') as f:
    for row in D:
        p, s, th = int(row[0]), int(row[1]), row[2]
        pred = model(a99, .99, s, row[3], th)
        f.write(f'{p} & {s} & {th:g} & {row[4]:.0f} & {row[5]:.0f} & {row[6]:.0f} & {pred:.0f}\\\\\n')
