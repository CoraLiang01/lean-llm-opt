import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv', sep=',')
locations = [str(i) for i in range(1, 16)]
distance = {}
for i_row, row in distance_df.iterrows():
    i = str(int(row['Unnamed: 0']))
    distance[i] = {}
    for j in locations:
        if i == j:
            distance[i][j] = 0.0
        else:
            val = row[j]
            if pd.isnull(val):
                j_row = distance_df[distance_df['Unnamed: 0'] == int(j)]
                if not j_row.empty:
                    sym_val = j_row.iloc[0][i]
                    if pd.isnull(sym_val):
                        raise ValueError(f'Missing distance between {i} and {j} in both directions.')
                    distance[i][j] = float(sym_val)
                else:
                    raise ValueError(f'Missing row for location {j} in distance matrix.')
            else:
                distance[i][j] = float(val)
for i in locations:
    for j in locations:
        if i != j:
            if not np.isclose(distance[i][j], distance[j][i], atol=1e-06):
                raise ValueError(f'Distance matrix is not symmetric at ({i},{j}): {distance[i][j]} vs {distance[j][i]}')
n = len(locations)
m = gp.Model('TSP_15_locations')
x = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
u = m.addVars([i for i in locations if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name='')
for j in locations:
    m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name='')
for i in locations:
    if i == '1':
        continue
    for j in locations:
        if j == '1' or i == j:
            continue
        m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    next_loc = {}
    for i in locations:
        for j in locations:
            if i != j and x[i, j].X > 0.5:
                next_loc[i] = j
                break
    tour = ['1']
    while True:
        last = tour[-1]
        nxt = next_loc[last]
        tour.append(nxt)
        if nxt == '1':
            break
        if len(tour) > n + 1:
            print('Warning: Tour reconstruction exceeded expected length.')
            break
    print('Optimal tour (visiting order):')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')