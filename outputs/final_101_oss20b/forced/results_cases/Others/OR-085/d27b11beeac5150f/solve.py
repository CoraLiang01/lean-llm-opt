import gurobipy as gp
import pandas as pd
import numpy as np
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(path, sep=',')
df.set_index('Unnamed: 0', inplace=True)
df.columns = df.columns.astype(int)
df = df.fillna(df.T)
np.fill_diagonal(df.values, 0)
dist = {(i, j): df.loc[i, j] for i in df.index for j in df.columns}
m = gp.Model('TSP')
nodes = list(range(1, 16))
N = len(nodes)
x = {}
for i in nodes:
    for j in nodes:
        if i < j:
            x[i, j] = m.addVar(vtype=gp.GRB.BINARY)
u = {}
for i in nodes:
    if i != 1:
        u[i] = m.addVar(vtype=gp.GRB.INTEGER, lb=2, ub=N)
for i in nodes:
    expr = gp.LinExpr()
    for j in nodes:
        if i < j:
            expr += x[i, j]
        elif i > j:
            expr += x[j, i]
    m.addConstr(expr == 2)
for i in nodes:
    if i == 1:
        continue
    for j in nodes:
        if j == 1 or j == i:
            continue
        m.addConstr(u[i] - u[j] + N * x[min(i, j), max(i, j)] <= N - 1)
obj = gp.LinExpr()
for (i, j), var in x.items():
    obj += dist[i, j] * var
m.setObjective(obj, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total distance: {m.objVal:.2f}')
    tour = [1]
    current = 1
    visited = set([1])
    while True:
        next_node = None
        for j in nodes:
            if j == current:
                continue
            var = x[min(current, j), max(current, j)]
            if var.X > 0.5:
                next_node = j
                break
        if next_node is None:
            break
        if next_node == 1:
            tour.append(1)
            break
        tour.append(next_node)
        visited.add(next_node)
        current = next_node
    print('Tour order:', ' -> '.join(map(str, tour)))
else:
    print(f'Optimization ended with status {m.status}')