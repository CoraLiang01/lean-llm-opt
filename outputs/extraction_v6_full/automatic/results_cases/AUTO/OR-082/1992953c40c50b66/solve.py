import gurobipy as gp
import pandas as pd
import numpy as np
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',')
locations = ['Depot', 'A', 'B', 'C']
row_ids = df['Unnamed: 0'].astype(str).str.strip().tolist()
col_ids = [col.strip() for col in df.columns if col != 'Unnamed: 0']
for loc in locations:
    if loc not in row_ids:
        raise ValueError(f"Location '{loc}' not found in CSV row indices.")
    if loc not in col_ids:
        raise ValueError(f"Location '{loc}' not found in CSV columns.")
distance = {}
for i in locations:
    row = df.loc[df['Unnamed: 0'].astype(str).str.strip() == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in CSV.")
    row = row.iloc[0]
    distance[i] = {}
    for j in locations:
        val = row[j]
        if pd.isnull(val):
            raise ValueError(f'Distance from {i} to {j} is missing in CSV.')
        distance[i][j] = float(val)
m = gp.Model('TSP_4node')
x = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='x')
u = m.addVars([loc for loc in locations if loc != 'Depot'], vtype=gp.GRB.CONTINUOUS, lb=1, ub=len(locations) - 1, name='u')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name=f'leave_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name=f'enter_{j}')
for i in [loc for loc in locations if loc != 'Depot']:
    for j in [loc for loc in locations if loc != 'Depot']:
        if i != j:
            m.addConstr(u[i] - u[j] + (len(locations) - 1) * x[i, j] <= len(locations) - 2, name=f'mtz_{i}_{j}')
m.optimize()