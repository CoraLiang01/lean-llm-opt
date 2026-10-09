import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
locations = [str(i) for i in range(1, 16)]
n = len(locations)
distance = {i: {} for i in locations}
for (row_idx, row) in df.iterrows():
    i = str(row['Unnamed: 0']).strip()
    if i not in locations:
        continue
    for j in locations:
        if j == i:
            distance[i][j] = 0.0
        else:
            val = row[j].strip()
            if val == '':
                row_j = df[df['Unnamed: 0'].astype(str).str.strip() == j]
                if not row_j.empty:
                    val_sym = row_j.iloc[0][i].strip()
                    if val_sym == '':
                        raise ValueError(f'Missing distance for ({i},{j}) and ({j},{i})')
                    distance[i][j] = float(val_sym)
                else:
                    raise ValueError(f'Missing row for location {j}')
            else:
                distance[i][j] = float(val)
for i in locations:
    for j in locations:
        if i != j:
            if not np.isclose(distance[i][j], distance[j][i]):
                raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {distance[i][j]} vs {distance[j][i]}')
m = Model()
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=GRB.BINARY, name='')
u_vars = m.addVars([i for i in locations if i != '1'], vtype=GRB.INTEGER, lb=2, ub=n, name='')
m.setObjective(quicksum((distance[i][j] * x_vars[i, j] for i in locations for j in locations if i != j)), GRB.MINIMIZE)
for i in locations:
    m.addConstr(quicksum((x_vars[i, j] for j in locations if j != i)) == 1)
for j in locations:
    m.addConstr(quicksum((x_vars[i, j] for i in locations if i != j)) == 1)
for i in locations:
    if i == '1':
        continue
    for j in locations:
        if j == '1' or i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2)
m.optimize()