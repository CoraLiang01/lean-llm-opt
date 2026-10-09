import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',')
locations = [str(i) for i in range(1, 16)]
row_ids = df['Unnamed: 0'].astype(str).tolist()
col_ids = [c for c in df.columns if c != 'Unnamed: 0']
if set(locations) != set(row_ids) or set(locations) != set(col_ids):
    raise ValueError('Mismatch between expected locations and CSV row/column labels.')
distance = {}
for i in locations:
    distance[i] = {}
    for j in locations:
        if i == j:
            distance[i][j] = 0.0
        else:
            val = df.loc[df['Unnamed: 0'].astype(str) == i, j]
            if not val.empty and pd.notnull(val.values[0]):
                distance[i][j] = float(val.values[0])
            else:
                val_sym = df.loc[df['Unnamed: 0'].astype(str) == j, i]
                if not val_sym.empty and pd.notnull(val_sym.values[0]):
                    distance[i][j] = float(val_sym.values[0])
                else:
                    raise ValueError(f'Missing distance between {i} and {j} in CSV.')
N = locations
n = len(N)
m = gp.Model('TSP_15_Cities')
x = m.addVars([(i, j) for i in N for j in N if i != j], vtype=gp.GRB.BINARY, name='')
u = m.addVars([i for i in N if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i in N for j in N if i != j)), gp.GRB.MINIMIZE)
for i in N:
    m.addConstr(gp.quicksum((x[i, j] for j in N if j != i)) == 1, name=f'out_{i}')
for j in N:
    m.addConstr(gp.quicksum((x[i, j] for i in N if i != j)) == 1, name=f'in_{j}')
for i in N:
    if i == '1':
        continue
    for j in N:
        if j == '1' or i == j:
            continue
        m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f'subtour_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    next_city = {}
    for i in N:
        for j in N:
            if i != j and x[i, j].X > 0.5:
                next_city[i] = j
                break
    tour = ['1']
    while len(tour) < n + 1:
        last = tour[-1]
        nxt = next_city[last]
        tour.append(nxt)
        if nxt == '1':
            break
    print('Optimal tour (visiting order):')
    print(' -> '.join(tour))
    print('Tour length:', m.objVal)
else:
    print(f'No optimal solution found. Status: {m.status}')