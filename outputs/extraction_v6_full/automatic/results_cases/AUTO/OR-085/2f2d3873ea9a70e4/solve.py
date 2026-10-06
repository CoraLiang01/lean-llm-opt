import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',')
locations = df['Unnamed: 0'].astype(str).tolist()
location_set = set(locations)
if len(locations) != 15 or sorted(locations, key=int) != [str(i) for i in range(1, 16)]:
    raise ValueError("Expected 15 locations with IDs 1..15 in 'Unnamed: 0' column.")
d = {}
for i_idx, i in enumerate(locations):
    d[i] = {}
    for j in locations:
        if i == j:
            d[i][j] = 0.0
        else:
            try:
                val = df.loc[df['Unnamed: 0'].astype(str) == i, j].values
                if len(val) == 0 or pd.isnull(val[0]):
                    val_sym = df.loc[df['Unnamed: 0'].astype(str) == j, i].values
                    if len(val_sym) == 0 or pd.isnull(val_sym[0]):
                        raise ValueError(f'Missing distance between {i} and {j} in both directions.')
                    d[i][j] = float(val_sym[0])
                else:
                    d[i][j] = float(val[0])
            except Exception as e:
                raise ValueError(f'Error extracting distance between {i} and {j}: {e}')
N = locations
n = len(N)
m = gp.Model('TSP_15_Cities')
x = m.addVars([(i, j) for i in N for j in N if i != j], vtype=gp.GRB.BINARY, name='x')
u = m.addVars([i for i in N if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='u')
m.setObjective(gp.quicksum((d[i][j] * x[i, j] for i in N for j in N if i != j)), gp.GRB.MINIMIZE)
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
        m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f'mtz_{i}_{j}')
m.optimize()