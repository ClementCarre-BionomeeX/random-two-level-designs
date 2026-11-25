"""Figures and appendix table for the equal-budget design comparison (results/compare_designs.jsonl) and DSD vs smaller
random fractions (results/compare_dsd_smaller.jsonl). Usage: python3 sims/fig_compare.py results figures
Env COMPARE=compare_designs_sigma SUFFIX=_sigma: same figures and table from the sigma-known run (outputs get SUFFIX;
the DSD-vs-smaller figure then reads compare_dsd_smaller_sigma.jsonl)."""
import json, sys, os, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
RES, OUT = sys.argv[1], sys.argv[2]
COMPARE = os.environ.get('COMPARE', 'compare_designs'); SUFFIX = os.environ.get('SUFFIX', '')
L = [json.loads(l) for l in open(f'{RES}/{COMPARE}.jsonl')]
SMALLER = 'compare_dsd_smaller' + SUFFIX
SM = [json.loads(l) for l in open(f'{RES}/{SMALLER}.jsonl')] if os.path.exists(f'{RES}/{SMALLER}.jsonl') else []
C = {'struct': '#eb6834', 'random': '#2a78d6', 'half': '#1baf7a', 'quarter': '#8a8a85'}
MK = {'struct': 's', 'random': 'o', 'half': '^', 'quarter': 'v'}
TYPES = [('main', 'Main effects'), ('both', '2×1D'), ('one', '1×1D'), ('neither', '0×1D')]
DES = [('ssd', 'Supersaturated (m = n/2)'), ('res3', 'Resolution III (m = n)'), ('res4', 'Fold-over, res. IV (m = 2n)'),
       ('dsd', 'Definitive screening (m = 2q+3)')]
PS = [100, 300, 1000]
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8a85',
                     'xtick.color': '#5a5a56', 'ytick.color': '#5a5a56', 'grid.color': '#e4e3df', 'grid.linewidth': .6})
def get(src, **kw):
    for r in src:
        if all(r.get(k) == v for k, v in kw.items()): return r
def nt(r, t, who): return r[who]['n_eff'].get(t, 0)
SHOW = {'equal': 'equal', 'heredity': 'empirical'}  # display names; 'heredity' stays the key in the results files
def fig(truth, sM, th, fname):
    fig, axs = plt.subplots(4, 4, figsize=(7.2, 6.6), sharex=True, sharey=True)
    for j, (d, dl) in enumerate(DES):
        for i, (t, tl) in enumerate(TYPES):
            ax = axs[i, j]; ax.grid(True, axis='y'); ax.set_ylim(-.03, 1.03)
            for who, lab in [('struct', 'Structured design + lasso'), ('random', 'Random fraction, same runs + lasso')]:
                ys, es = [], []
                for p in PS:
                    r = get(L, p=p, design=d, truth=truth, sM=sM, theta=th)
                    v = r[who]['recall_top_s'].get(t) if r else None; n = nt(r, t, who) if r else 0
                    ys.append(v); es.append(2 * (v * (1 - v) / n) ** .5 if v is not None and n else 0)
                ax.plot(range(3), ys, color=C[who], lw=2, marker=MK[who], ms=5, mec='white', mew=1, label=lab, zorder=3, clip_on=False)
                for x, (v, e) in enumerate(zip(ys, es)):
                    if v is not None: ax.plot([x, x], [v - e, v + e], color=C[who], lw=1, alpha=.6, zorder=2)
            ax.set_xticks(range(3)); ax.set_xticklabels([str(p) for p in PS])
            if i == 0: ax.set_title(dl, fontsize=7.5, color='#2a2a28')
            if j == 0: ax.set_ylabel(f'{tl}\nrecall in top s')
            if i == 3: ax.set_xlabel('number of factors p')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(f'{OUT}/{fname}.pdf'); fig.savefig(f'{OUT}/{fname}.png', dpi=110); plt.close(fig)
fig('equal', 5, .5, 'compare_equal' + SUFFIX); fig('heredity', 5, .5, 'compare_heredity' + SUFFIX)
# appendix table: every cell, recall per type (structured | random) and mean false discovery proportion
with open(f'{OUT}/compare_table{SUFFIX}.tex', 'w') as f:
    order = {'ssd': 0, 'res3': 1, 'res4': 2, 'dsd': 3}
    for r in sorted(L, key=lambda r: (r['p'], order[r['design']], r['truth'], r['sM'], r['theta'])):
        cell = lambda who: ' & '.join('--' if r[who]['recall_top_s'].get(t) is None else f"{r[who]['recall_top_s'][t]:.2f}" for t, _ in TYPES)
        f.write(f"{r['p']} & {r['design']} & {SHOW[r['truth']]} & {r['sM']}+{r['sI']} & {r['theta']:g} & {r['M']} & "
                f"{cell('struct')} & {r['struct']['fdp_mean']:.2f} & {cell('random')} & {r['random']['fdp_mean']:.2f}\\\\\n")
if SM:   # DSD at full size vs random at 1, 1/2, 1/4 of its runs: recall over all types, by p, equal and heredity
    fig2, axs = plt.subplots(2, 3, figsize=(7.2, 4.4), sharey=True)
    for i, truth in enumerate(['equal', 'heredity']):
        for j, p in enumerate(PS):
            ax = axs[i, j]; ax.grid(True, axis='y'); ax.set_ylim(-.03, 1.03)
            cells = [(sM, th) for sM in (5, 10) for th in (.25, .5, 1.0)]
            def allrec(r, who):
                a = sum(r[who]['recall_top_s'][t] * r[who]['n_eff'][t] for t in r[who]['n_eff']); n = sum(r[who]['n_eff'].values()); return a / n
            series = {'struct': [], 'random': [], 'half': [], 'quarter': []}
            for sM, th in cells:
                r = get(L, p=p, design='dsd', truth=truth, sM=sM, theta=th)
                h = get(SM, p=p, design='dsd', truth=truth, sM=sM, theta=th, rfrac=.5)
                q = get(SM, p=p, design='dsd', truth=truth, sM=sM, theta=th, rfrac=.25)
                series['struct'].append(allrec(r, 'struct') if r else None); series['random'].append(allrec(r, 'random') if r else None)
                series['half'].append(allrec(h, 'random') if h else None); series['quarter'].append(allrec(q, 'random') if q else None)
            labs = {'struct': 'DSD (m = 2q+3)', 'random': 'random, same m', 'half': 'random, m/2', 'quarter': 'random, m/4'}
            for k in ['struct', 'random', 'half', 'quarter']:
                ax.plot(range(6), series[k], color=C[k], lw=0, marker=MK[k], ms=5, mec='white', mew=1, label=labs[k], zorder=3)
            ax.set_xticks(range(6)); ax.set_xticklabels([f'{a}+{6 if a==5 else 12}\nθ={b:g}' for a, b in cells], fontsize=6)
            r0 = get(L, p=p, design='dsd'); ax.set_title(f'{SHOW[truth]}, p = {p} (DSD: {r0["M"]} runs)', fontsize=7.5)
            if j == 0: ax.set_ylabel('recall in top s, all types')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig2.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.0))
    fig2.tight_layout(rect=(0, 0, 1, .93)); fig2.savefig(f'{OUT}/compare_dsd_smaller{SUFFIX}.pdf'); fig2.savefig(f'{OUT}/compare_dsd_smaller{SUFFIX}.png', dpi=110)
