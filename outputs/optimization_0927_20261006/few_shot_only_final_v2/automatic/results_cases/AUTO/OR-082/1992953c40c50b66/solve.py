import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
distance_df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
row_ids = distance_df['Unnamed: 0'].tolist()
col_ids = [col for col in distance_df.columns if col in locations]
if not all((loc in row_ids for loc in locations)):
    raise ValueError(f'Not all required locations found in rows: {locations} vs {row_ids}')
if not all((loc in col_ids for loc in locations)):
    raise ValueError(f'Not all required locations found in columns: {locations} vs {col_ids}')
distance = {}
for i in locations:
    row = distance_df.loc[distance_df['Unnamed: 0'] == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in DistanceMatrix.csv")
    for j in locations:
        val = row.iloc[0][j]
        try:
            distance[i, j] = float(val)
        except Exception:
            raise ValueError(f"Invalid or missing distance from {i} to {j}: '{val}'")
m = gp.Model('TSP_4node')
arcs = [(i, j) for i in locations for j in locations if i != j]
x_vars = m.addVars(arcs, vtype=gp.GRB.BINARY, name='')
u_locs = [loc for loc in locations if loc != 'Depot']
u_vars = m.addVars(u_locs, vtype=gp.GRB.CONTINUOUS, lb=1, ub=len(locations) - 1, name='')
m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for (i, j) in arcs)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'out_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name=f'in_{j}')
for i in u_locs:
    for j in u_locs:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + (len(locations) - 1) * x_vars[i, j] <= len(locations) - 2, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    tour = []
    used_arcs = {(i, j): x_vars[i, j].X for (i, j) in arcs if x_vars[i, j].X > 0.5}
    current = 'Depot'
    tour.append(current)
    for _ in range(len(locations)):
        next_locs = [j for (i, j) in used_arcs if i == current]
        if not next_locs:
            break
        next_loc = next_locs[0]
        tour.append(next_loc)
        current = next_loc
        if current == 'Depot':
            break
    print('Optimal route:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')