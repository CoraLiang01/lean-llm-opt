import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',')
required_locations = ['Depot', 'A', 'B', 'C']
row_ids = df['Unnamed: 0'].astype(str).str.strip().tolist()
col_ids = [col.strip() for col in df.columns if col != 'Unnamed: 0']
for loc in required_locations:
    if loc not in row_ids:
        raise ValueError(f"Location '{loc}' not found in CSV row indices.")
    if loc not in col_ids:
        raise ValueError(f"Location '{loc}' not found in CSV column headers.")
distance = {}
for i in required_locations:
    row = df.loc[df['Unnamed: 0'].astype(str).str.strip() == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in CSV.")
    for j in required_locations:
        val = row[j].values[0]
        try:
            distance[i, j] = float(val)
        except Exception:
            raise ValueError(f'Distance from {i} to {j} is not numeric: {val}')
nodes = required_locations
m = gp.Model('TSP_SmallCourier')
x = m.addVars(nodes, nodes, vtype=gp.GRB.BINARY, name='')
n = len(nodes)
u_nodes = [loc for loc in nodes if loc != 'Depot']
u = m.addVars(u_nodes, lb=1, ub=n - 1, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[i, j] * x[i, j] for i in nodes for j in nodes if i != j)), gp.GRB.MINIMIZE)
for i in nodes:
    m.addConstr(gp.quicksum((x[i, j] for j in nodes if j != i)) == 1, name=f'out_{i}')
for j in nodes:
    m.addConstr(gp.quicksum((x[i, j] for i in nodes if i != j)) == 1, name=f'in_{j}')
for i in u_nodes:
    for j in u_nodes:
        if i != j:
            m.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    tour = []
    current = 'Depot'
    visited = set()
    for _ in range(n):
        for j in nodes:
            if current != j and x[current, j].X > 0.5:
                tour.append((current, j))
                visited.add(current)
                current = j
                break
    print('Optimal route:')
    route_str = tour[0][0]
    for arc in tour:
        route_str += f' -> {arc[1]}'
    print(route_str)
else:
    print(f'No optimal solution found. Status: {m.status}')