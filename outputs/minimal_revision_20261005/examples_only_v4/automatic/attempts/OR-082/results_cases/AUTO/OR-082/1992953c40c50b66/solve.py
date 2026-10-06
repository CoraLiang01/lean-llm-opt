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
        raise ValueError(f"Location '{loc}' not found in CSV row headers.")
    if loc not in col_ids:
        raise ValueError(f"Location '{loc}' not found in CSV column headers.")
distance = {}
for i in locations:
    row = df.loc[df['Unnamed: 0'].astype(str).str.strip() == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in CSV.")
    for j in locations:
        val = row[j].values[0]
        if pd.isnull(val):
            raise ValueError(f"Distance from '{i}' to '{j}' is missing in CSV.")
        distance[i, j] = float(val)
arcs = [(i, j) for i in locations for j in locations if i != j]
m = gp.Model('TSP_Small')
x = m.addVars(arcs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((distance[i, j] * x[i, j] for (i, j) in arcs)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name=f'out_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name=f'in_{j}')
u = m.addVars([loc for loc in locations if loc != 'Depot'], lb=1, ub=len(locations) - 1, vtype=gp.GRB.CONTINUOUS, name='')
n = len(locations)
for i in locations:
    for j in locations:
        if i != j and i != 'Depot' and (j != 'Depot'):
            m.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, name=f'subtour_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (i, j) in arcs:
        print(f'x[{i},{j}] {x[i, j].VarName} {x[i, j].X}')
    for loc in [l for l in locations if l != 'Depot']:
        print(f'u[{loc}] {u[loc].VarName} {u[loc].X}')
else:
    print(f'Solver status: {m.status}')