import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
required_locations = ['Depot', 'A', 'B', 'C']
df['loc_id'] = df['Unnamed: 0'].str.strip()
df = df[df['loc_id'].isin(required_locations)].copy()
df = df.set_index('loc_id')
for col in required_locations:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in distance matrix.")
for col in required_locations:
    df[col] = df[col].astype(float)
dists = {}
for i in required_locations:
    dists[i] = {}
    for j in required_locations:
        dists[i][j] = float(df.loc[i, j])
locations = required_locations
m = Model()
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=GRB.BINARY, name='')
u_locs = [loc for loc in locations if loc != 'Depot']
u_vars = m.addVars(u_locs, vtype=GRB.CONTINUOUS, lb=1, ub=len(u_locs), name='')
m.setObjective(quicksum((dists[i][j] * x_vars[i, j] for i in locations for j in locations if i != j)), GRB.MINIMIZE)
for j in locations:
    if j == 'Depot':
        m.addConstr(quicksum((x_vars[i, j] for i in locations if i != j)) == 1)
    else:
        m.addConstr(quicksum((x_vars[i, j] for i in locations if i != j)) == 1)
for i in locations:
    if i == 'Depot':
        m.addConstr(quicksum((x_vars[i, j] for j in locations if i != j)) == 1)
    else:
        m.addConstr(quicksum((x_vars[i, j] for j in locations if i != j)) == 1)
for i in u_locs:
    for j in u_locs:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + len(u_locs) * x_vars[i, j] <= len(u_locs) - 1)
m.optimize()