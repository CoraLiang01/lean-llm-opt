import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',')
location_ids = [str(i) for i in range(1, 16)]
distance = {}
for i_row in df['Unnamed: 0']:
    i_str = str(int(i_row))
    distance[i_str] = {}
    for j_col in location_ids:
        if j_col == i_str:
            distance[i_str][j_col] = 0.0
        else:
            val = df.loc[df['Unnamed: 0'] == int(i_str), j_col].values
            if len(val) == 0 or pd.isna(val[0]):
                val_sym = df.loc[df['Unnamed: 0'] == int(j_col), i_str].values
                if len(val_sym) == 0 or pd.isna(val_sym[0]):
                    raise ValueError(f'Missing distance between {i_str} and {j_col}')
                distance[i_str][j_col] = float(val_sym[0])
            else:
                distance[i_str][j_col] = float(val[0])
for i in location_ids:
    for j in location_ids:
        if i != j:
            d1 = distance[i][j]
            d2 = distance[j][i]
            if not np.isclose(d1, d2, atol=1e-06):
                raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {d1} vs {d2}')
N = location_ids
m = Model('TSP')
x = m.addVars(N, N, vtype=GRB.BINARY, name='')
u = m.addVars([i for i in N if i != '1'], vtype=GRB.INTEGER, lb=2, ub=15, name='')
m.setObjective(quicksum((distance[i][j] * x[i, j] for i in N for j in N if i != j)), GRB.MINIMIZE)
for i in N:
    m.addConstr(quicksum((x[i, j] for j in N if j != i)) == 1)
for j in N:
    m.addConstr(quicksum((x[i, j] for i in N if i != j)) == 1)
for i in N:
    if i == '1':
        continue
    for j in N:
        if j == '1' or i == j:
            continue
        m.addConstr(u[i] - u[j] + 15 * x[i, j] <= 14)
m.optimize()