import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
distance_df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
distance_df = distance_df.set_index('Unnamed: 0')
distance_df.index = distance_df.index.astype(str).str.strip()
distance_df.columns = distance_df.columns.astype(str).str.strip()
for loc in locations:
    if loc not in distance_df.index or loc not in distance_df.columns:
        raise ValueError(f"Location '{loc}' missing from distance matrix rows or columns.")
distance = {}
for i in locations:
    for j in locations:
        if i == j:
            continue
        val = distance_df.loc[i, j]
        try:
            distance[i, j] = float(val)
        except Exception:
            raise ValueError(f"Distance from {i} to {j} is not a valid number: '{val}'")
arc_keys = [(i, j) for i in locations for j in locations if i != j]
customer_locs = [loc for loc in locations if loc != 'Depot']

def solve_tsp():
    m = gp.Model('TSP_SmallCourier')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(arc_keys, vtype=gp.GRB.BINARY, name='')
    n = len(locations)
    u_vars = m.addVars(customer_locs, lb=1, ub=n - 1, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for (i, j) in arc_keys)), gp.GRB.MINIMIZE)
    for k in customer_locs:
        m.addConstr(gp.quicksum((x_vars[i, k] for i in locations if i != k)) == 1, name=f'in_{k}')
    for k in customer_locs:
        m.addConstr(gp.quicksum((x_vars[k, j] for j in locations if j != k)) == 1, name=f'out_{k}')
    m.addConstr(gp.quicksum((x_vars['Depot', j] for j in locations if j != 'Depot')) == 1, name='out_depot')
    m.addConstr(gp.quicksum((x_vars[i, 'Depot'] for i in locations if i != 'Depot')) == 1, name='in_depot')
    for i in customer_locs:
        for j in customer_locs:
            if i == j:
                continue
            m.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2, name=f'subtour_{i}_{j}')
    m.optimize()
    return m
m = solve_tsp()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')