import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',')
row_ids = df['Unnamed: 0'].astype(str).tolist()
col_ids = [col for col in df.columns if col != 'Unnamed: 0']
if set(row_ids) != set(col_ids):
    raise ValueError('Row and column location IDs do not match in the distance matrix.')
locations = sorted(row_ids, key=lambda x: int(x))
n = len(locations)
d = {}
for i in locations:
    d[i] = {}
    for j in locations:
        if i == j:
            d[i][j] = 0.0
        else:
            val_ij = df.loc[df['Unnamed: 0'].astype(str) == i, j].values
            val_ji = df.loc[df['Unnamed: 0'].astype(str) == j, i].values
            v_ij = float(val_ij[0]) if len(val_ij) > 0 and (not pd.isnull(val_ij[0])) else None
            v_ji = float(val_ji[0]) if len(val_ji) > 0 and (not pd.isnull(val_ji[0])) else None
            if v_ij is not None and v_ji is not None:
                if abs(v_ij - v_ji) > 1e-06:
                    raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {v_ij} vs {v_ji}')
                d[i][j] = v_ij
            elif v_ij is not None:
                d[i][j] = v_ij
            elif v_ji is not None:
                d[i][j] = v_ji
            else:
                raise ValueError(f'Missing distance for ({i},{j}) in both directions.')
m = gp.Model('TSP')
x = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
u = m.addVars([i for i in locations if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
m.setObjective(gp.quicksum((d[i][j] * x[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1)
for j in locations:
    m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1)
for i in locations:
    if i == '1':
        continue
    for j in locations:
        if j == '1' or i == j:
            continue
        m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    next_node = {}
    for i in locations:
        for j in locations:
            if i != j and x[i, j].X > 0.5:
                next_node[i] = j
    tour = ['1']
    while True:
        last = tour[-1]
        nxt = next_node.get(last, None)
        if nxt is None or nxt == '1':
            break
        tour.append(nxt)
    tour.append('1')
    print('Optimal tour (visiting order):')
    print(' -> '.join(tour))
    print('Tour positions (u[i]) for i != 1:')
    for i in locations:
        if i != '1':
            print(f'  Location {i}: position {int(round(u[i].X))}')
else:
    print(f'No optimal solution found. Status: {m.status}')