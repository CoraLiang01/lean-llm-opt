"""Exact WL teaching examples, vector figures, and arithmetic verification."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from wl_bipartite import wl_kernels

plt.rcParams.update({'font.family': ['Arial Unicode MS', 'DejaVu Sans'],
    'mathtext.fontset': 'stix', 'font.size': 11, 'pdf.fonttype': 42,
    'savefig.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE, GREEN, INK = '#0072B2', '#D55E00', '#009E73', '#253343'
PALETTE = dict(V=GREEN, R=BLUE, a=GREEN, b=BLUE, c=ORANGE, d='#CC79A7',
               p=GREEN, q=BLUE, r=ORANGE, s='#A76B00', t='#CC79A7')


def graph(rows):
    names = ['x1', 'x2'] + [f'C{i+1}' for i in range(len(rows))]
    adjacency = [[] for _ in names]
    for i, variables in enumerate(rows):
        for j in variables:
            adjacency[j].append(2+i)
            adjacency[2+i].append(j)
    return (['V', 'V']+['R']*len(rows), adjacency), names


A, names_A = graph([[0,1],[0,1]])
C, names_C = graph([[0,1],[0,1],[0]])
graphs = [A, C]
history = [[g[0][:] for g in graphs]]
signatures = []
for alphabet in [['a','b','c','d'], ['p','q','r','s','t']]:
    old = history[-1]
    dictionary, new, details = {}, [], []
    for labels, (_, adjacency) in zip(old, graphs):
        updated, detail = [], []
        for v, neighbors in enumerate(adjacency):
            signature = (labels[v], tuple(sorted(labels[u] for u in neighbors)))
            if signature not in dictionary:
                dictionary[signature] = alphabet[len(dictionary)]
            updated.append(dictionary[signature]); detail.append(signature)
        new.append(updated); details.append(detail)
    history.append(new); signatures.append(details)

alphabets = [['V','R'],['a','b','c','d'],['p','q','r','s','t']]
counts, arithmetic = [], []
for i, alphabet in enumerate(alphabets):
    vectors = [np.array([Counter(labels)[letter] for letter in alphabet],dtype=int) for labels in history[i]]
    counts.append(vectors)
    arithmetic.append({'round':i,'labels':alphabet,'counts_A':vectors[0].tolist(),
        'counts_C':vectors[1].tolist(), 'cross':int(vectors[0]@vectors[1]),
        'self_A':int(vectors[0]@vectors[0]),'self_C':int(vectors[1]@vectors[1])})
assert [q['cross'] for q in arithmetic] == [10,6,2]
assert [q['self_A'] for q in arithmetic] == [8,8,8]
assert [q['self_C'] for q in arithmetic] == [13,7,7]
kernels = wl_kernels(graphs,2)
scores=[]
for h in range(3):
    cross=sum(q['cross'] for q in arithmetic[:h+1])
    aa=sum(q['self_A'] for q in arithmetic[:h+1])
    cc=sum(q['self_C'] for q in arithmetic[:h+1])
    score=float(cross/np.sqrt(aa*cc))
    assert np.isclose(score,kernels[h][0,1],atol=1e-14,rtol=0)
    scores.append(score)
excluded=wl_kernels([A,A],2)
assert all(np.isclose(k[0,1],1) for k in excluded.values())


def save(fig,name):
    for suffix in ['png','pdf','svg']:
        fig.savefig(HERE/f'{name}.{suffix}',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def draw(ax,g,labels,names, title='', ylim=(-2.25,2.25)):
    positions=[(2.65,1.20),(2.65,-.25)]+[(.35,y) for y in [1.20,-.25,-1.70][:len(g[0])-2]]
    for row in range(2,len(g[0])):
        for v in g[1][row]:
            p,q=positions[row],positions[v]
            ax.plot([p[0],q[0]],[p[1],q[1]],color=ORANGE if row==4 else '#B8C3CE',
                linewidth=2 if row==4 else 1.5,zorder=1)
    for i,(x,y) in enumerate(positions):
        color=PALETTE[labels[i]]
        face=matplotlib.colors.to_hex(.10*np.array(matplotlib.colors.to_rgb(color))+.90)
        if i<2:
            shape=Circle((x,y),.40,facecolor=face,edgecolor=color,linewidth=2,zorder=3)
            name=f'$x_{{{i+1}}}$'
        else:
            shape=FancyBboxPatch((x-.46,y-.32),.92,.64,boxstyle='round,pad=.03,rounding_size=.09',
                facecolor=face,edgecolor=color,linewidth=2,zorder=3)
            name=f'$C_{{{i-1}}}$'
        ax.add_patch(shape)
        ax.text(x,y+.12,name,ha='center',va='center',fontsize=11,color=INK,zorder=4)
        ax.text(x,y-.15,f'标签 {labels[i]}',ha='center',va='center',fontsize=9,color=color,zorder=4)
    ax.set_xlim(-.40,3.35);ax.set_ylim(*ylim);ax.set_aspect('equal');ax.axis('off')
    if title:ax.set_title(title,fontweight='bold',color=INK,pad=8)


# Figure 1: actual support matrices and graphs.
fig,axes=plt.subplots(1,2,figsize=(10.5,6))
for ax,g,names,title in zip(axes,graphs,[names_A,names_C],['问题 A：2 个变量，2 条约束，4 条边','问题 C：新增一条单变量约束']):
    draw(ax,g,g[0],names,title,ylim=(-2.3,4.5))
    model='$\\min\;2x_1+x_2$\n$C_1:\;x_1+x_2\\leq4$\n$C_2:\;x_1-x_2\\geq1$'
    if g is C:model+='\n$C_3:\;x_1\\leq3$'
    ax.text(-.20,4.10,model,ha='left',va='top',fontsize=14,color=INK,linespacing=1.35)
    ax.text(1.4,-2.22,'圆：连续变量 V    圆角矩形：不等式 R',ha='center',fontsize=10,color='#657485')
fig.suptitle('图 1 · 只要约束中有非零项，就连接相应变量',fontsize=16,fontweight='bold',y=1.03)
fig.text(.5,-.025,'教学例子：C₃ 作为显式约束参与图；两模型的非负 Bounds 不进入这张示例图。',ha='center',fontsize=11,color=INK)
save(fig,'01_models_and_graphs')

# Figure 2: same nodes and edges, only labels evolve.
fig,axes=plt.subplots(3,2,figsize=(9.8,11))
for i in range(3):
    for j in range(2):
        draw(axes[i,j],graphs[j],history[i][j],[names_A,names_C][j],
            f'{"A" if j==0 else "C"} · 第 {i} 轮标签')
fig.suptitle('图 2 · 三轮都用同一张图：只更新标签',fontsize=17,fontweight='bold',y=1.01)
fig.subplots_adjust(hspace=.22,wspace=.18)
fig.text(.5,-.01,'第 1 轮：C 中 x₁ 变为 c；第 2 轮：C 中两条原有约束变为 s。',ha='center',fontsize=12,color=INK)
save(fig,'02_rounds_same_graph')

# Figure 3: explain synchronous update with exact numbers.
fig,axes=plt.subplots(1,2,figsize=(11.2,4.7))
for ax,is_c in zip(axes,[False,True]):
    old_neighbors=['c','a'] if is_c else ['a','a']
    for x,label,name in [(0,old_neighbors[0],'$x_1$'),(3,old_neighbors[1],'$x_2$')]:
        color=PALETTE[label]
        face=matplotlib.colors.to_hex(.12*np.array(matplotlib.colors.to_rgb(color))+.88)
        ax.add_patch(Circle((x,2.1),.38,facecolor=face,edgecolor=color,lw=2))
        ax.text(x,2.1,name+'\n'+label,ha='center',va='center',fontsize=12,color=color)
        ax.annotate('',xy=(1.5,1.3),xytext=(x,1.72),arrowprops=dict(arrowstyle='->',color=color,lw=1.8))
    ax.add_patch(FancyBboxPatch((1.05,.80),.9,.6,boxstyle='round,pad=.04',facecolor='#E7F1F7',edgecolor=BLUE,lw=2))
    ax.text(1.5,1.10,'$C_1$\n旧标签 b',ha='center',va='center',fontsize=12,color=BLUE)
    sig='b | [a,c]' if is_c else 'b | [a,a]'
    new='s' if is_c else 'q'
    ax.text(1.5,.35,f'第 2 轮签名：{sig}',ha='center',fontsize=14,color=INK)
    ax.text(1.5,-.14,f'新标签：{new}',ha='center',fontsize=16,fontweight='bold',color=PALETTE[new])
    ax.set_xlim(-.6,3.6);ax.set_ylim(-.45,2.9);ax.axis('off')
    ax.set_title(('C' if is_c else 'A')+'：同一条原有约束，看到不同的旧标签',fontsize=12,fontweight='bold',color=INK)
fig.suptitle('图 3 · 第 2 轮用第 1 轮的标签，统一更新所有节点',fontsize=16,fontweight='bold',y=1.04)
fig.text(.5,-.015,'这些箭头表示读取邻居标签，不是新增的图边；C₂ 的计算完全相同。',ha='center',fontsize=11,color=INK)
save(fig,'03_synchronous_propagation')

# Figure 4: every count is generated, not manually approximated.
fig,axes=plt.subplots(1,3,figsize=(11.8,3.7))
for i,ax in enumerate(axes):
    x=np.arange(len(alphabets[i]))
    for offset,values,color,label in [(-.18,counts[i][0],BLUE,'A'),(.18,counts[i][1],ORANGE,'C')]:
        bars=ax.bar(x+offset,values,.34,color=color,label=label)
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+.04,str(value),ha='center',fontsize=10)
    ax.set_xticks(x);ax.set_xticklabels(alphabets[i]);ax.set_ylim(0,3.6);ax.set_yticks([0,1,2,3]);ax.set_ylabel('节点数量')
    ax.set_title(f'第 {i} 轮 · 共同贡献 {arithmetic[i]["cross"]}',fontweight='bold')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].legend(frameon=False,ncol=2)
fig.suptitle('图 4 · 数的是各标签的节点数量，不是在数新增节点',fontsize=16,fontweight='bold',y=1.04)
fig.tight_layout()
save(fig,'04_label_histograms')

# Figure 5: cumulative features with explicit round boundaries.
features=np.stack([np.concatenate([q[j] for q in counts]) for j in range(2)])
fig,ax=plt.subplots(figsize=(12.2,3.0))
ax.imshow(features,cmap='Blues',vmin=0,vmax=3,aspect='auto')
for row in range(2):
    for col in range(features.shape[1]):
        ax.text(col,row,str(features[row,col]),ha='center',va='center',fontsize=15,
            color='white' if features[row,col]>1.6 else INK)
ticks=[f'({i},{letter})' for i,alphabet in enumerate(alphabets) for letter in alphabet]
ax.set_xticks(np.arange(11));ax.set_xticklabels(ticks,fontsize=10)
ax.set_yticks([0,1]);ax.set_yticklabels(['A','C'],fontsize=13)
for x in [1.5,5.5]:ax.axvline(x,color=ORANGE,lw=3)
for center,label in [(.5,'第 0 轮'),(3.5,'第 1 轮'),(8,'第 2 轮')]:
    ax.text(center,-.8,label,ha='center',fontsize=12,fontweight='bold',clip_on=False)
ax.set_title('图 5 · h=2 的特征向量：拼接三轮计数，维度为 2 + 4 + 5 = 11',pad=43,fontweight='bold')
ax.set_xlabel('每一列是“轮次、标签”这一对；同一节点会在不同轮次各贡献一次计数',labelpad=14)
save(fig,'05_cumulative_feature_vectors')

# Figure 6: arithmetic summary and dependence on graph encoding.
fig,axes=plt.subplots(1,2,figsize=(11.5,4.3))
ax=axes[0]
ax.plot([0,1,2],scores,color=BLUE,marker='o',markersize=8,lw=2)
for h,s in enumerate(scores):ax.annotate(f'{s:.4f}',(h,s),xytext=(0,12),textcoords='offset points',ha='center',fontsize=12)
ax.set_xticks([0,1,2]);ax.set_xlabel('累计到第 h 轮');ax.set_ylabel('累计 WL 余弦相似度');ax.set_ylim(.60,1.07);ax.grid(alpha=.17)
ax.set_title('同一组 A、C：观察深度增加',fontweight='bold')
ax=axes[1]
bars=ax.bar([0,1],[scores[2],1],color=[ORANGE,GREEN],width=.52)
for bar,value in zip(bars,[scores[2],1]):ax.text(bar.get_x()+bar.get_width()/2,value+.025,f'{value:.4f}',ha='center',fontsize=13)
ax.set_xticks([0,1]);ax.set_xticklabels(['C₃ 进入示例图','C₃ 转入 Bounds\n且 Bounds 不进入图'])
ax.set_ylim(0,1.14);ax.set_ylabel('h=2 的相似度');ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
ax.set_title('同一组 A、C：图编码改变',fontweight='bold')
fig.suptitle('图 6 · 算法计算的是选定图表示的相似度',fontsize=16,fontweight='bold',y=1.03)
fig.tight_layout()
save(fig,'06_similarity_and_encoding')

# Figure 7: the unfolded neighborhood is a view, not new model vertices.
fig,axes=plt.subplots(1,2,figsize=(11.5,4.5))
for ax,is_c in zip(axes,[False,True]):
    root=(3,2.4)
    children=[(1.4,1.1),(4.6,1.1)]
    leaves=[[(.4,-.2,'C1'),(1.4,-.2,'C2'),(2.4,-.2,'C3')],
            [(4.0,-.2,'C1'),(5.2,-.2,'C2')]] if is_c else [
            [(.8,-.2,'C1'),(2.0,-.2,'C2')],[(4.0,-.2,'C1'),(5.2,-.2,'C2')]]
    for child,branch in zip(children,leaves):
        ax.plot([root[0],child[0]],[root[1],child[1]],color='#BBC5CD',lw=1.6,zorder=1)
        for x,y,name in branch:
            ax.plot([child[0],x],[child[1],y],color=ORANGE if name=='C3' else '#BBC5CD',lw=1.6,zorder=1)
    def tree_node(x,y,name,kind):
        color=ORANGE if name=='C3' else (GREEN if kind=='V' else BLUE)
        face=matplotlib.colors.to_hex(.10*np.array(matplotlib.colors.to_rgb(color))+.90)
        shape=Circle((x,y),.30,fc=face,ec=color,lw=1.8,zorder=3) if kind=='V' else FancyBboxPatch(
            (x-.35,y-.24),.7,.48,boxstyle='round,pad=.02',fc=face,ec=color,lw=1.8,zorder=3)
        ax.add_patch(shape);ax.text(x,y,name,ha='center',va='center',color=color,fontsize=11,zorder=4)
    tree_node(*root,'C1','R')
    for i,(x,y) in enumerate(children):tree_node(x,y,f'x{i+1}','V')
    for branch in leaves:
        for x,y,name in branch:tree_node(x,y,name,'R')
    ax.set_xlim(-.1,6.1);ax.set_ylim(-.7,3);ax.set_aspect('equal');ax.axis('off')
    ax.set_title(('C' if is_c else 'A')+'：以 C₁ 为根展开两层',fontweight='bold')
fig.suptitle('图 7 · 邻域展开图会重复画同一个节点，不代表原图增加节点',fontsize=16,fontweight='bold',y=1.01)
fig.text(.5,.015,'C₁ 既是根，也会通过 C₁ → x₁ → C₁ 再次出现；计算时无需真的生成这些树。',ha='center',fontsize=11,color=INK)
save(fig,'07_unfolded_neighborhood')

detail_rows=[]
for i in range(3):
    for j,names in enumerate([names_A,names_C]):
        for v,name in enumerate(names):
            detail_rows.append(dict(round=i,graph='A' if j==0 else 'C',node=name,
                old_label=history[i-1][j][v] if i else '',
                neighbor_old_labels=json.dumps(signatures[i-1][j][v][1]) if i else '',
                new_label=history[i][j][v]))
with (HERE.parent/'node_labels_all_rounds.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(detail_rows[0]));writer.writeheader();writer.writerows(detail_rows)
(HERE.parent/'toy_verification.json').write_text(json.dumps(dict(
    arithmetic=arithmetic,cumulative_scores=scores,
    labels_per_round=history,excluded_bound_similarity=1.0,
    checked_against=str(ROOT/'wl_bipartite.py')),indent=2),encoding='utf-8')
print(json.dumps(dict(round_arithmetic=arithmetic,scores=scores,figures=7),indent=2))
