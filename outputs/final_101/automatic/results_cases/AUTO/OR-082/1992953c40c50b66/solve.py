import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv', sep=',')
locations = ['Depot', 'A', 'B', 'C']

def norm(s):
    return str(s).strip().casefold()
row_label_to_idx = {norm(label): idx for idx, label in enumerate(distance_df['Unnamed: 0'])}
for loc in locations:
    if norm(loc) not in row_label_to_idx:
        raise KeyError(f"Location '{loc}' not found in DistanceMatrix.csv rows.")
for loc in locations:
    if loc not in distance_df.columns:
        raise KeyError(f"Location '{loc}' not found in DistanceMatrix.csv columns.")
distance = {}
for i in locations:
    distance[i] = {}
    row_idx = row_label_to_idx[norm(i)]
    for j in locations:
        val = distance_df.at[row_idx, j]
        if pd.isnull(val):
            raise ValueError(f'Missing distance from {i} to {j} in DistanceMatrix.csv.')
        distance[i][j] = float(val)
m = gp.Model('TSP_4node')
arcs = [(i, j) for i in locations for j in locations if i != j]
x = m.addVars(arcs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i, j in arcs)), gp.GRB.MINIMIZE)
for k in locations:
    m.addConstr(gp.quicksum((x[k, j] for j in locations if j != k)) == 1, name=f'out_{k}')
    m.addConstr(gp.quicksum((x[i, k] for i in locations if i != k)) == 1, name=f'in_{k}')
n = len(locations)
customers = [loc for loc in locations if loc != 'Depot']
u = m.addVars(customers, vtype=gp.GRB.CONTINUOUS, lb=1, ub=n - 1, name='')
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    tour = {}
    for i, j in arcs:
        if x[i, j].X > 0.5:
            tour[i] = j
    sequence = ['Depot']
    current = 'Depot'
    while True:
        next_node = tour.get(current)
        if next_node is None or next_node == 'Depot':
            sequence.append('Depot')
            break
        sequence.append(next_node)
        current = next_node
    print('Optimal route sequence:')
    print(' -> '.join(sequence))
    print('Visit order:')
    for idx, loc in enumerate(sequence):
        print(f'  {idx + 1}: {loc}')
else:
    print(f'No optimal solution found. Status: {m.status}')