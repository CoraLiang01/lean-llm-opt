"""Reproducible vector figures and exact WL examples for the AE response.

Run with /usr/bin/python3 outputs/wl_response_20261006/generate_wl_figures.py.
All example kernels are checked against the project's unchanged wl_bipartite.py.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from wl_bipartite import wl_kernels

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
BLUE, ORANGE, GREEN, INK = '#0072B2', '#D55E00', '#009E73', '#23303C'
LIGHT_BLUE, LIGHT_ORANGE = '#E7F1F7', '#FCF0E8'


def graph(n, rows):
    labels = ['V:C'] * n + ['R:I'] * len(rows)
    adj = [[] for _ in labels]
    for i, variables in enumerate(rows):
        for j in variables:
            adj[j].append(n+i)
            adj[n+i].append(j)
    return labels, adj


A = graph(2, [[0, 1], [0, 1]])
B = graph(2, [[0, 1], [0, 1]])
C = graph(2, [[0, 1], [0, 1], [0]])
LOOP = graph(4, [[0, 1], [1, 2], [2, 3], [3, 0]])
BLOCKS = graph(4, [[0, 1], [0, 1], [2, 3], [2, 3]])


def refine(graphs, depth):
    labels = [x[0][:] for x in graphs]
    histories, signatures = [labels], []
    for i in range(1, depth+1):
        codebook, next_labels, level = {}, [], []
        for old, (_, adj) in zip(labels, graphs):
            new, sigs = [], []
            for v, neighbors in enumerate(adj):
                sig = (old[v], tuple(sorted(old[u] for u in neighbors)))
                if sig not in codebook:
                    codebook[sig] = len(codebook) + 1
                new.append(codebook[sig]); sigs.append(sig)
            next_labels.append(new); level.append(sigs)
        signatures.append(level); labels = next_labels; histories.append(labels)
    return histories, signatures


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save(fig, name):
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(FIG / f'{name}.{suffix}', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def node(ax, pos, text, kind, color=None):
    x, y = pos
    color = color or (BLUE if kind == 'row' else GREEN)
    face = LIGHT_BLUE if kind == 'row' else '#E9F5F0'
    if color == ORANGE:
        face = LIGHT_ORANGE
    if kind == 'row':
        shape = FancyBboxPatch((x-.32,y-.19), .64,.38, boxstyle='round,pad=0.02,rounding_size=0.07',
                              facecolor=face, edgecolor=color, linewidth=1.5, zorder=3)
    else:
        shape = Circle((x,y),.25,facecolor=face,edgecolor=color,linewidth=1.5,zorder=3)
    ax.add_patch(shape)
    ax.text(x,y,text,ha='center',va='center',fontsize=10,color=INK,zorder=4)


def draw_graph(ax, g, labels=None, origin=(0,0), scale=1., highlight_last=False):
    n = sum(x.startswith('V:') for x in g[0]); m=len(g[0])-n
    ox, oy = origin
    vy = np.linspace(1.35, -.15, n)
    ry = np.linspace(1.35, -.15, m)
    positions = [(ox+2.1*scale,oy+y*scale) for y in vy] + [(ox+.35*scale,oy+y*scale) for y in ry]
    for r in range(m):
        for v in g[1][n+r]:
            p,q=positions[n+r],positions[v]
            special=highlight_last and r==m-1
            ax.plot([p[0],q[0]],[p[1],q[1]],color=ORANGE if special else '#A6B1BA',
                    lw=1.8 if special else 1.2,zorder=1)
    for v in range(n):
        text=labels[v] if labels else f'$x_{v+1}$\nC'
        node(ax,positions[v],text,'var')
    for r in range(m):
        text=labels[n+r] if labels else f'$r_{r+1}$\nI'
        node(ax,positions[n+r],text,'row',ORANGE if highlight_last and r==m-1 else None)
    return positions


history, signatures = refine([A,C], 3)
reference_kernel = wl_kernels([A,C],3)
toy=[]
dot_total=aa_total=cc_total=0
for i, level in enumerate(history):
    aa,cc=Counter(level[0]),Counter(level[1])
    dot=sum(aa[k]*cc[k] for k in aa)
    aa_i=sum(x*x for x in aa.values()); cc_i=sum(x*x for x in cc.values())
    dot_total+=dot;aa_total+=aa_i;cc_total+=cc_i
    s=dot_total/np.sqrt(aa_total*cc_total)
    assert np.isclose(s,reference_kernel[i][0,1],atol=1e-14)
    toy.append(dict(h=i,round_cross=dot,round_self_A=aa_i,round_self_C=cc_i,
                    cumulative_cross=dot_total,cumulative_self_A=aa_total,
                    cumulative_self_C=cc_total,similarity=float(s)))
assert np.isclose(toy[2]['similarity'],1/np.sqrt(2))
assert all(np.isclose(k[0,1],1) for k in wl_kernels([A,B],3).values())
assert all(np.isclose(k[0,1],1) for k in wl_kernels([LOOP,BLOCKS],3).values())
(OUT/'toy_kernel_results.json').write_text(json.dumps({'added_constraint':toy,
    'numerical_only_similarity_h2':1.0,'connected_vs_disconnected_similarity_h2':1.0},indent=2))
with (OUT/'toy_round_counts.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['round','graph','node','label','signature'])
    for i, level in enumerate(history):
        for gidx, name in enumerate(['A','C']):
            for v,label in enumerate(level[gidx]):
                w.writerow([i,name,('x' if v<2 else 'r')+str(v+1 if v<2 else v-1),label,
                            '' if i==0 else repr(signatures[i-1][gidx][v])])


# Figure 1: the actual encoding, including a coefficient-only control.
fig,axes=plt.subplots(1,3,figsize=(10.2,4.2))
models=[A,B,C]
titles=['(a) Reference A','(b) Numerical change B','(c) Additional constraint C']
equations=[['$\min\ 2x_1+x_2$','$x_1+x_2\leq4$','$x_1-x_2\geq1$'],
           ['$\min\ 3x_1+x_2$','$x_1+0.05x_2\leq5$','$x_1-x_2\geq2$'],
           ['$\min\ 2x_1+x_2$','$x_1+x_2\leq4$','$x_1-x_2\geq1$','$x_1\leq3$']]
for ax,g,title,eqs in zip(axes,models,titles,equations):
    ax.set_xlim(-.1,2.65);ax.set_ylim(-.95,3.6);ax.axis('off')
    ax.text(.01,3.45,title,fontsize=11,fontweight='bold',color=INK)
    for k,e in enumerate(eqs):ax.text(.08,3.03-.28*k,e,fontsize=11,color=ORANGE if title.endswith('C') and k==3 else INK)
    ax.text(.08,1.78,'Continuous variables; nonnegative',fontsize=8,color='#52606C')
    draw_graph(ax,g,highlight_last=(g is C))
    score=1 if g is not C else toy[2]['similarity']
    ax.text(1.25,-.76,f'$S_2(A,{title[-1]}) = {score:.4f}$',ha='center',fontsize=12,
            color=ORANGE if g is C else BLUE)
fig.text(.5,-.01,'Squares: inequality rows (R:I). Circles: continuous variables (V:C). Edges: nonzero coefficients.',
         ha='center',fontsize=9,color='#52606C')
fig.subplots_adjust(wspace=.3)
save(fig,'fig_wl_lp_graphs')


# Figure 2: shared refined labels, exact per-round contributions and cumulative scores.
fig,axes=plt.subplots(1,3,figsize=(10.2,5.4))
for i,ax in enumerate(axes):
    ax.set_xlim(-.15,2.65);ax.set_ylim(-3.45,2.2);ax.axis('off')
    ax.text(.02,2.02,f'({chr(97+i)}) Round $i={i}$',fontweight='bold',fontsize=12,color=INK)
    lab=lambda x: ('C' if x=='V:C' else 'I') if i==0 else rf'$\sigma_{{{i},{x}}}$'
    draw_graph(ax,A,[lab(x) for x in history[i][0]],origin=(0,.15),scale=.82)
    ax.text(-.02,1.63,'A',fontweight='bold',color=BLUE)
    draw_graph(ax,C,[lab(x) for x in history[i][1]],origin=(0,-1.83),scale=.82,highlight_last=True)
    ax.text(-.02,-.35,'C',fontweight='bold',color=ORANGE)
    r=toy[i]
    ax.text(.0,-2.66,rf'$\sum_\sigma c_i(A,\sigma)c_i(C,\sigma) = {r["round_cross"]}$',fontsize=11)
    ax.text(.0,-3.03,rf'$S_{i}(A,C) = {r["similarity"]:.4f}$',fontsize=13,color=BLUE)
fig.text(.5,.99,r'$M_i(v)\ \longrightarrow\ s_i(v)\ \longrightarrow\ l_i(v)=f(s_i(v))$',ha='center',fontsize=15,color=INK)
fig.text(.5,-.015,'The same refined label means the same previous label and the same neighbor-label multiset.',ha='center',fontsize=9)
fig.subplots_adjust(top=.9,wspace=.30)
save(fig,'fig_wl_refinement')


# Figure 3: exact 1-WL collision, independent of iteration depth.
fig,axes=plt.subplots(1,2,figsize=(8,3.6))
for ax,g,title in zip(axes,[LOOP,BLOCKS],['(a) One connected component','(b) Two connected components']):
    ax.set_xlim(-1.8,1.8);ax.set_ylim(-1.55,1.55);ax.set_aspect('equal');ax.axis('off')
    ax.text(0,1.4,title,ha='center',fontsize=11,fontweight='bold')
    if g is LOOP:
        order=[0,4,1,5,2,6,3,7]
        points={v:(1.08*np.cos(np.pi/2+2*np.pi*k/8),1.08*np.sin(np.pi/2+2*np.pi*k/8)) for k,v in enumerate(order)}
    else:
        points={0:(-1.25,.55),1:(-1.25,-.55),4:(-.45,.55),5:(-.45,-.55),
                2:(.45,.55),3:(.45,-.55),6:(1.25,.55),7:(1.25,-.55)}
    for v in range(4):
        for r in g[1][v]:
            p,q=points[v],points[r];ax.plot([p[0],q[0]],[p[1],q[1]],color='#A6B1BA',lw=1.4)
    for v in range(8):
        x,y=points[v]
        if v<4:
            ax.add_patch(Circle((x,y),.13,facecolor='#E9F5F0',edgecolor=GREEN,lw=1.5,zorder=3))
        else:
            ax.add_patch(FancyBboxPatch((x-.14,y-.12),.28,.24,boxstyle='round,pad=0.01',facecolor=LIGHT_BLUE,edgecolor=BLUE,lw=1.5,zorder=3))
        ax.text(x,y,'C' if v<4 else 'I',ha='center',va='center',fontsize=8,zorder=4)
fig.text(.5,.01,r'Four variables + four inequalities; every node has degree 2.   $S_h=1$ for every $h$.',ha='center',fontsize=10)
fig.subplots_adjust(bottom=.17,wspace=.25)
save(fig,'fig_wl_limit')


# Figure 4: all actual instance scores, not just means of category summaries.
near=rows(OUT/'nearest_h2.csv');summary=rows(OUT/'category_summary.csv')
cats=['NRM','RA','TP','AP','UFLP','Mixture','Others']
fig,ax=plt.subplots(figsize=(8,4.2));rng=np.random.default_rng(6)
for i,cat in enumerate(cats):
    sel=[r for r in near if r['category']==cat]
    x=np.array([float(r['similarity_same']) for r in sel])
    y=i+rng.uniform(-.12,.12,len(x))
    ax.scatter(x,y,s=19,color=BLUE,alpha=.42,zorder=2,edgecolors='none')
    q=np.percentile(x,[25,50,75]);a=np.median([float(r['similarity_all']) for r in sel])
    ax.plot(q[[0,2]],[i,i],color=BLUE,lw=4.5,zorder=3,solid_capstyle='butt')
    ax.scatter([q[1]],[i],color=BLUE,s=47,marker='D',edgecolor='white',lw=.5,zorder=4)
    ax.scatter([a],[i-.23],color=ORANGE,s=37,marker='s',zorder=4)
ax.set_yticks(range(len(cats)))
ax.set_yticklabels([f'{c} (n={sum(r["category"]==c for r in near)})' for c in cats])
ax.set_xlim(0,1.02);ax.set_ylim(6.48,-.55)
ax.set_xlabel('Cosine-normalized cumulative WL similarity ($h=2$)')
ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
ax.scatter([],[],c=BLUE,s=20,alpha=.5,label='Individual same-category scores')
ax.scatter([],[],c=BLUE,s=45,marker='D',label='Same-category median; line = IQR')
ax.scatter([],[],c=ORANGE,s=35,marker='s',label='All-reference median')
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.14),ncol=2,frameon=False,fontsize=8.3)
fig.tight_layout()
save(fig,'fig_wl_101_results')

print(json.dumps({'toy_h2':toy[2], 'figures':[p.name for p in FIG.glob('*.pdf')]},indent=2))
