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
        raise ValueError(f"Location '{loc}' not found in CSV row labels.")
    if loc not in col_ids:
        raise ValueError(f"Location '{loc}' not found in CSV column labels.")
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
            raise ValueError(f'Missing distance from {i} to {j} in CSV.')
        distance[i][j] = float(val)
arcs = [(i, j) for i in locations for j in locations if i != j]
m = gp.Model('TSP_4node')
x = m.addVars(arcs, vtype=gp.GRB.BINARY, name='')
u = m.addVars([loc for loc in locations if loc != 'Depot'], lb=1, ub=3, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i, j in arcs)), gp.GRB.MINIMIZE)
for k in ['A', 'B', 'C']:
    m.addConstr(gp.quicksum((x[i, k] for i in locations if i != k)) == 1, name=f'in_{k}')
    m.addConstr(gp.quicksum((x[k, j] for j in locations if j != k)) == 1, name=f'out_{k}')
m.addConstr(gp.quicksum((x['Depot', j] for j in locations if j != 'Depot')) == 1, name='out_depot')
m.addConstr(gp.quicksum((x[i, 'Depot'] for i in locations if i != 'Depot')) == 1, name='in_depot')
for i in ['A', 'B', 'C']:
    for j in ['A', 'B', 'C']:
        if i != j:
            m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    succ = {}
    for i, j in arcs:
        if x[i, j].X > 0.5:
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
        if len(tour) > 10:
            print('Warning: Tour reconstruction exceeded expected length.')
            break
    print('Optimal route:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')