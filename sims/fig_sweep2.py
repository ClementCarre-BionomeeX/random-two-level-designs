"""Runs needed by the random fraction, second sweep (sweep_runs2.jsonl): m_q for four criteria, fit of the run formula
(fitted on p <= 1000, tested out of sample on p = 3000), figures and LaTeX table.
Usage: python3 sims/fig_sweep2.py results/sweep_runs2.jsonl figures"""
import json, sys, collections, numpy as np, matplotlib
from scipy.optimize import least_squares; from scipy.stats import norm
matplotlib.use('Agg'); import matplotlib.pyplot as plt
L = [json.loads(l) for l in open(sys.argv[1])]; OUT = sys.argv[2]
ch = collections.defaultdict(list)
for r in L: ch[(r['p'], r['sizes'], r['s'], r['theta'])].append(r)
CRIT = [('recall', .99), ('all_found', .95), ('bh_recall', .99), ('bh_all_found', .95)]
RECALL_KEYS = ('recall', 'bh_recall')          # fraction of effects found: margin fixed at the normal quantile
MIN_SIZE = {'equal': 1.0, 'spread': 0.5}        # smallest effect / theta
def m_at(rows, key, q):  # smallest M reaching q, interpolated linearly in log M between the bracketing points
    rows = sorted(rows, key=lambda r: r['M']); prev = None
    for r in rows:
        if r[key] >= q:
            if prev is None: return float(r['M'])
            x0, x1 = np.log(prev['M']), np.log(r['M']); y0, y1 = prev[key], r[key]
            return float(np.exp(x0 + (q - y0) / (y1 - y0) * (x1 - x0)))
        prev = r
    return np.nan
rows = []
for k in sorted(ch):
    d = dict(p=k[0], sizes=k[1], s=k[2], th=k[3], N=ch[k][0]['N'])
    for key, q in CRIT: d[key] = m_at(ch[k], key, q)
    near = [r for r in ch[k] if r['bh_ndisc'] > 0 and r['reps'] >= 100]
    d['fdr_max'] = max([r['bh_fdr'] for r in near], default=np.nan); rows.append(d)
def model(a, z, s, N, thmin): return a * s * np.log(N / s) + (np.sqrt(2 * np.log(N)) + z) ** 2 / thmin ** 2
def col(name, sel): return np.array([r[name] for r in sel], float)
FIT = {}
for sizes in ['equal', 'spread']:
    for key, q in CRIT:
        tr = [r for r in rows if r['sizes'] == sizes and r['p'] <= 1000]; te = [r for r in rows if r['sizes'] == sizes and r['p'] > 1000]
        S, N, T, y = col('s', tr), col('N', tr), col('th', tr) * MIN_SIZE[sizes], col(key, tr)
        if key in RECALL_KEYS and sizes == 'equal':
            zf = norm.ppf(q); r = least_squares(lambda c: np.log(model(c[0], zf, S, N, T)) - np.log(y), [2.]); a, z = r.x[0], zf
        else:
            r = least_squares(lambda c: np.log(model(c[0], c[1], S, N, T)) - np.log(y), [2., 2.]); a, z = r.x
        e = model(a, z, S, N, T) / y - 1
        f = dict(a=a, z=z, z_free=not (key in RECALL_KEYS and sizes == 'equal'), n=len(tr), med=float(np.median(abs(e))), max=float(abs(e).max()))
        if te:
            et = model(a, z, col('s', te), col('N', te), col('th', te) * MIN_SIZE[sizes]) / col(key, te) - 1
            f.update(n_test=len(te), test_mean=float(et.mean()), test_min=float(et.min()), test_max=float(et.max()))
        FIT[f'{sizes}/{key}'] = f
        print(sizes, key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()})
fd = [r['fdr_max'] for r in rows if not np.isnan(r['fdr_max'])]
print('max realised BH FDR near threshold: median %.3f max %.3f' % (np.median(fd), max(fd)))
ratio = collections.defaultdict(list)
eq = {(r['p'], r['s'], r['th']): r for r in rows if r['sizes'] == 'equal'}
for r in rows:
    if r['sizes'] == 'spread': ratio[r['th']].append(r['recall'] / eq[(r['p'], r['s'], r['th'])]['recall'])
for th, v in sorted(ratio.items()): print(f'spread/equal recall .99 theta={th}: median {np.median(v):.2f} range {min(v):.2f}-{max(v):.2f}')
bh = [r['bh_recall'] / r['recall'] for r in rows]; print('BH/top-s recall .99: median %.2f range %.2f-%.2f' % (np.median(bh), min(bh), max(bh)))
af = [r['all_found'] / r['recall'] for r in rows]; print('all-found .95/recall .99: median %.2f range %.2f-%.2f' % (np.median(af), min(af), max(af)))
json.dump(FIT, open(f'{OUT}/sweep2_fit.json', 'w'), indent=1)
# figures: runs for recall 0.99 in the top s against s, one panel per p, one colour per theta
C = {.25: '#2a78d6', .5: '#eb6834', 1.0: '#1baf7a'}; MK = {.25: 'o', .5: 's', 1.0: '^'}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8a85',
                     'xtick.color': '#5a5a56', 'ytick.color': '#5a5a56', 'grid.color': '#e4e3df', 'grid.linewidth': .6})
def figure(sizes, ps, name):
    f = FIT[f'{sizes}/recall']; ss = np.linspace(4, 64, 100)
    fig, axs = plt.subplots(1, len(ps), figsize=(1.9 * len(ps) + .8, 2.9), sharey=True)
    for ax, p in zip(axs, ps):
        Np = p * (p + 1) // 2; ax.grid(True, axis='y')
        for th in [.25, .5, 1.0]:
            sel = sorted([r for r in rows if r['sizes'] == sizes and r['p'] == p and r['th'] == th], key=lambda r: r['s'])
            if not sel: continue
            ax.plot(ss, model(f['a'], f['z'], ss, Np, th * MIN_SIZE[sizes]), color=C[th], lw=1.2, ls='--', zorder=2)
            ax.plot(col('s', sel), col('recall', sel), ls='none', marker=MK[th], ms=6, color=C[th], mec='white', mew=1, zorder=3, label=f'θ = {th:g}σ')
        t = f'p = {p}  (N = {Np:,})'.replace(',', ' ')
        ax.set_title(t + ('\nnot used in the fit' if p > 1000 else ''), fontsize=8, color='#2a2a28')
        ax.set_xlabel('number of active effects s'); ax.set_xticks([4, 16, 32, 48, 64]); ax.set_xlim(0, 68)
    axs[0].set_ylabel('runs for recall 0.99 in top s')
    h, l = axs[0].get_legend_handles_labels(); h.append(plt.Line2D([], [], color='#5a5a56', ls='--', lw=1.2)); l.append('fitted formula')
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png', dpi=110)
figure('equal', [100, 300, 1000, 3000], 'sweep2_equal'); figure('spread', [100, 300, 1000], 'sweep2_spread')
with open(f'{OUT}/sweep2_table.tex', 'w') as fh:
    for r in rows:
        f = FIT[f"{r['sizes']}/recall"]; pred = model(f['a'], f['z'], r['s'], r['N'], r['th'] * MIN_SIZE[r['sizes']])
        fh.write(f"{r['sizes']} & {r['p']} & {r['s']} & {r['th']:g} & " + ' & '.join(f'{r[k]:.0f}' for k, _ in CRIT) + f' & {pred:.0f}\\\\\n')
