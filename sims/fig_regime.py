"""Regime-map figures from results/regime.jsonl and results/regime_1000.jsonl.
Usage: python3 sims/fig_regime.py results/regime.jsonl results/regime_1000.jsonl figures"""
import json, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
args = sys.argv[1:]; OUT = args.pop()
POOLED = [a for a in args if 'rank_pooled' in a]; args = [a for a in args if a not in POOLED]
R = [json.loads(l) for f in args for l in open(f)]
# classical strategy: for each cell and type, the better of the two ranking rules (staged, pooled |t|);
# this choice favours the classical strategy and is applied only when the pooled results are given
if POOLED:
    PL = {}
    for f in POOLED:
        for l in open(f):
            q = json.loads(l); PL[(q['p'], q['scen'], q['s'], q['theta'], q['budget'])] = q['classical']
    for r in R:
        k = (r['p'], r['scen'], r['s'], r['theta'], r['budget'])
        if k in PL: r['classical'] = {t: max(v, PL[k][t]) for t, v in r['classical'].items()}
NT = {'low': {'main': 3, 'both': 1, 'one': 1, 'neither': 1}, 'mid': {'main': 5, 'both': 2, 'one': 2, 'neither': 2},
      'high': {'main': 10, 'both': 4, 'one': 4, 'neither': 4}}
C = {'classical': '#eb6834', 'random': '#2a78d6'}
MK = {'classical': 's', 'random': 'o'}
LAB = {'classical': 'Classical sequential (better of two rankings)', 'random': 'Random fraction + lasso'}
TYPES = [('main', 'Main effects'), ('both', '2×1D interactions'),
         ('one', '1×1D interactions'), ('neither', '0×1D interactions')]
PS = [30, 100, 300, 1000]
BUD = {'nofold': 'no fold-over', 'fold': 'fold-over'}
def get(p, b, scen, s, th):
    for r in R:
        if r['p'] == p and r['budget'] == b and r['scen'] == scen and r['s'] == s and r['theta'] == th:
            return r
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8a85', 'axes.labelcolor': '#3a3a38', 'xtick.color': '#5a5a56',
                     'ytick.color': '#5a5a56', 'grid.color': '#e4e3df', 'grid.linewidth': .6})
def grid(cols, xs, xticklabels, xlabel, cell, coltitle, fname):
    """cols: list of column keys; cell(colkey, x) -> result dict."""
    fig, axs = plt.subplots(4, len(cols), figsize=(1.8 * len(cols) + .6, 6.6), sharex=True, sharey=True, squeeze=False)
    for i, (t, tl) in enumerate(TYPES):
        for j, ck in enumerate(cols):
            ax = axs[i, j]; ax.grid(True, axis='y'); ax.set_ylim(-.03, 1.03)
            for m in ['classical', 'random']:
                ys = [(cell(ck, x) or {}).get(m, {}).get(t) for x in xs]
                ax.plot(range(len(xs)), ys, color=C[m], lw=2, marker=MK[m], ms=5,
                        mec='white', mew=1, label=LAB[m], clip_on=False, zorder=3)
                for xi, x in enumerate(xs):          # +- 2 Monte Carlo standard errors
                    c = cell(ck, x)
                    if not c or c[m].get(t) is None: continue
                    r_ = c[m][t]; n_ = c['reps'] * NT[c['s']][t]; e_ = 2 * (r_ * (1 - r_) / n_) ** .5
                    ax.plot([xi, xi], [r_ - e_, r_ + e_], color=C[m], lw=1, alpha=.6, zorder=2, clip_on=False)
            ax.set_xticks(range(len(xs))); ax.set_xticklabels(xticklabels)
            if i == 0: ax.set_title(coltitle(ck), fontsize=8, color='#2a2a28')
            if j == 0: ax.set_ylabel(f'{tl}\nrecall in top |S|')
            if i == 3: ax.set_xlabel(xlabel)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(f'{OUT}/{fname}.pdf'); fig.savefig(f'{OUT}/{fname}.png', dpi=110)
    plt.close(fig)
def ptitle(b):
    def f(p):
        r = get(p, b, 'mixed', 'mid', .5)
        return f'p = {p}\n{r["runs"]} runs, N = {p*(p+1)//2:,}'.replace(',', ' ')
    return f
# recall against the number of factors
grid([('nofold', th) for th in (.25, .5, 1.0)] + [('fold', th) for th in (.25, .5, 1.0)],
     PS, [str(p) for p in PS], 'number of factors p',
     lambda ck, p: get(p, ck[0], 'mixed', 'mid', ck[1]),
     lambda ck: f'{BUD[ck[0]]}\nθ = {ck[1]:g}σ', 'regime_p')
for b in ('nofold', 'fold'):
    grid(PS, [.25, .5, 1.0], ['0.25', '0.5', '1'], 'effect size θ/σ',
         lambda p, th: get(p, b, 'mixed', 'mid', th), ptitle(b), f'regime_theta_{b}')
    grid(PS, ['low', 'mid', 'high'], ['3+3', '5+6', '10+12'], 'sparsity level',
         lambda p, s: get(p, b, 'mixed', s, .5), ptitle(b), f'regime_sparsity_{b}')
