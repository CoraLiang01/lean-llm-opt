import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',')

def norm_id(x):
    return str(x).strip()
row_ids = [norm_id(x) for x in df['Unnamed: 0']]
col_ids = [norm_id(x) for x in df.columns[1:]]
nodes = ['Depot', 'A', 'B', 'C']
for n in nodes:
    if n not in row_ids:
        raise ValueError(f"Node '{n}' not found in distance matrix rows.")
    if n not in col_ids:
        raise ValueError(f"Node '{n}' not found in distance matrix columns.")
distance = {}
for i in nodes:
    i_row_idx = row_ids.index(i)
    for j in nodes:
        j_col_idx = col_ids.index(j)
        val = df.iloc[i_row_idx, j_col_idx + 1]
        distance[i, j] = float(val)
m = gp.Model('TSP_4node')
x = m.addVars(nodes, nodes, vtype=gp.GRB.BINARY, name='')
u = m.addVars(['A', 'B', 'C'], lb=1, ub=3, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[i, j] * x[i, j] for i in nodes for j in nodes if i != j)), gp.GRB.MINIMIZE)
for i in nodes:
    m.addConstr(gp.quicksum((x[i, j] for j in nodes if j != i)) == 1, name=f'out_{i}')
for j in nodes:
    m.addConstr(gp.quicksum((x[i, j] for i in nodes if i != j)) == 1, name=f'in_{j}')
for i in ['A', 'B', 'C']:
    for j in ['A', 'B', 'C']:
        if i != j:
            m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    succ = {}
    for i in nodes:
        for j in nodes:
            if i != j and x[i, j].X > 0.5:
                succ[i] = j
    tour = ['Depot']
    while True:
        last = tour[-1]
        next_node = succ.get(last, None)
        if next_node is None or next_node == 'Depot':
            break
        tour.append(next_node)
    tour.append('Depot')
    print('Optimal route:')
    print(' -> '.join(tour))
    print('Visit order:')
    for idx, loc in enumerate(tour):
        print(f'  {idx + 1}: {loc}')
else:
    print(f'No optimal solution found. Status: {m.status}')