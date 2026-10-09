import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
node_cols = [col for col in df.columns if re.fullmatch('\\d+', col.strip())]
node_ids = [col.strip() for col in node_cols]
row_ids = df['Unnamed: 0'].str.strip().tolist()
if set(row_ids) != set(node_ids):
    raise ValueError(f'Row IDs {row_ids} do not match column node IDs {node_ids}')
distance = {i: {} for i in node_ids}
for (idx, row) in df.iterrows():
    i = str(row['Unnamed: 0']).strip()
    for j in node_ids:
        val = row[j].strip()
        if val == '':
            continue
        distance[i][j] = float(val)
for i in node_ids:
    for j in node_ids:
        if i == j:
            distance[i][j] = 0.0
        elif j not in distance[i] or distance[i][j] == '':
            if i in distance[j] and distance[j][i] != '':
                distance[i][j] = float(distance[j][i])
            else:
                raise ValueError(f'Missing distance between {i} and {j}')
N = node_ids
n = len(N)
start_node = '1'
m = gp.Model('TSP_15_Cities')
x_vars = m.addVars([(i, j) for i in N for j in N if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in N if i != start_node], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in N for j in N if i != j)), gp.GRB.MINIMIZE)
for i in N:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in N if j != i)) == 1, name=f'leave_{i}')
for j in N:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in N if i != j)) == 1, name=f'enter_{j}')
for i in N:
    if i == start_node:
        continue
    for j in N:
        if j == start_node or i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name=f'subtour_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    succ = {}
    for i in N:
        for j in N:
            if i != j and x_vars[i, j].X > 0.5:
                succ[i] = j
                break
    tour = [start_node]
    while True:
        next_node = succ[tour[-1]]
        if next_node == start_node:
            tour.append(start_node)
            break
        tour.append(next_node)
    print('Optimal tour order:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')