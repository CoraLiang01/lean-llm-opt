import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']

def norm(s):
    return s.strip().casefold()
row_labels = [str(r) for r in df['Unnamed: 0']]
col_labels = [str(c) for c in df.columns[1:]]
row_norm_map = {norm(r): r for r in row_labels}
col_norm_map = {norm(c): c for c in col_labels}
for loc in locations:
    if norm(loc) not in row_norm_map:
        raise KeyError(f"Location '{loc}' not found as a row in DistanceMatrix.csv")
    if norm(loc) not in col_norm_map:
        raise KeyError(f"Location '{loc}' not found as a column in DistanceMatrix.csv")
distance = {}
for i in locations:
    i_row = row_norm_map[norm(i)]
    distance[i] = {}
    for j in locations:
        j_col = col_norm_map[norm(j)]
        val = df.loc[df['Unnamed: 0'] == i_row, j_col].values
        if len(val) != 1:
            raise ValueError(f'Missing or ambiguous distance from {i} to {j}')
        try:
            distance[i][j] = float(val[0])
        except Exception as e:
            raise ValueError(f"Invalid distance value from {i} to {j}: '{val[0]}'") from e
nodes = locations
m = gp.Model('TSP_4node')
x_vars = m.addVars([(i, j) for i in nodes for j in nodes if i != j], vtype=gp.GRB.BINARY, name='')
u_nodes = [n for n in nodes if n != 'Depot']
u_vars = m.addVars(u_nodes, vtype=gp.GRB.CONTINUOUS, lb=1, ub=len(u_nodes), name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in nodes for j in nodes if i != j)), gp.GRB.MINIMIZE)
for k in u_nodes:
    m.addConstr(gp.quicksum((x_vars[i, k] for i in nodes if i != k)) == 1, name=f'enter_{k}')
for k in u_nodes:
    m.addConstr(gp.quicksum((x_vars[k, j] for j in nodes if j != k)) == 1, name=f'leave_{k}')
m.addConstr(gp.quicksum((x_vars['Depot', j] for j in nodes if j != 'Depot')) == 1, name='depot_depart')
m.addConstr(gp.quicksum((x_vars[i, 'Depot'] for i in nodes if i != 'Depot')) == 1, name='depot_enter')
n_cust = len(u_nodes)
for i in u_nodes:
    for j in u_nodes:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + n_cust * x_vars[i, j] <= n_cust - 1, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    succ = {}
    for (i, j) in x_vars.keys():
        if x_vars[i, j].X > 0.5:
            succ[i] = j
    tour = ['Depot']
    while True:
        last = tour[-1]
        if last not in succ:
            break
        nxt = succ[last]
        tour.append(nxt)
        if nxt == 'Depot':
            break
        if len(tour) > len(nodes) + 2:
            print('Warning: Tour reconstruction exceeded expected length.')
            break
    print('Optimal route:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')