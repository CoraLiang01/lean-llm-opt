import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
row_ids = df.iloc[:, 0].astype(str).str.strip()
col_ids = df.columns[1:].astype(str).str.strip()
if len(row_ids) != 15 or len(col_ids) != 15:
    raise ValueError(f'Expected 15 locations, got {len(row_ids)} rows and {len(col_ids)} columns.')
if not set(row_ids) == set(col_ids):
    raise ValueError('Row and column location IDs do not match in the distance matrix.')
sorted_ids = sorted(row_ids, key=lambda x: int(x))
id_to_idx = {id_: idx for idx, id_ in enumerate(sorted_ids)}
idx_to_id = {idx: id_ for id_, idx in id_to_idx.items()}
df = df.set_index(df.columns[0])
df = df.loc[sorted_ids, sorted_ids]
d = {}
for i in sorted_ids:
    for j in sorted_ids:
        if i == j:
            continue
        val_ij = df.loc[i, j].strip()
        val_ji = df.loc[j, i].strip()
        if val_ij == '' and val_ji == '':
            raise ValueError(f'Missing distance for ({i},{j}) and ({j},{i})')
        if val_ij != '' and val_ji != '':
            dij = float(val_ij)
            dji = float(val_ji)
            if not np.isclose(dij, dji, atol=1e-06):
                raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {dij} vs {dji}')
            d[i, j] = dij
        elif val_ij != '':
            d[i, j] = float(val_ij)
        else:
            d[i, j] = float(val_ji)
N = sorted_ids
m = gp.Model('TSP_15_Cities')
x_vars = m.addVars([(i, j) for i in N for j in N if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in N if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=15, name='')
m.setObjective(gp.quicksum((d[i, j] * x_vars[i, j] for i in N for j in N if i != j)), gp.GRB.MINIMIZE)
for i in N:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in N if j != i)) == 1, name=f'depart_{i}')
for j in N:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in N if i != j)) == 1, name=f'arrive_{j}')
for i in N:
    if i == '1':
        continue
    for j in N:
        if j == '1' or i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + 15 * x_vars[i, j] <= 14, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    tour = []
    used = set()
    current = '1'
    for _ in range(len(N)):
        for j in N:
            if j != current and x_vars[current, j].X > 0.5:
                tour.append((current, j))
                used.add(current)
                current = j
                break
        else:
            break
    if len(tour) < len(N):
        print('Warning: incomplete tour found.')
    print('Tour order (start at 1):')
    path = ['1']
    next_node = None
    for _ in range(len(N)):
        for i, j in tour:
            if i == path[-1]:
                path.append(j)
                break
        if path[-1] == '1':
            break
    print(' -> '.join(path))
else:
    print(f'No optimal solution found. Status: {m.status}')