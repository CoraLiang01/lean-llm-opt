import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',')
locations = [str(i) for i in range(1, 16)]
n = len(locations)
d = {}
for irow in range(n):
    i = str(df.loc[irow, 'Unnamed: 0'])
    for jcol in locations:
        if i == jcol:
            d[i, jcol] = 0.0
        else:
            val = df.at[irow, jcol] if not pd.isna(df.at[irow, jcol]) else None
            if val is None:
                jrow = df.index[df['Unnamed: 0'] == int(jcol)]
                if len(jrow) == 1:
                    jrow = jrow[0]
                    val_sym = df.at[jrow, i] if not pd.isna(df.at[jrow, i]) else None
                    if val_sym is not None:
                        val = val_sym
            if val is None:
                raise ValueError(f'Missing distance between locations {i} and {jcol}')
            d[i, jcol] = float(val)
m = gp.Model('TSP')
x = m.addVars(locations, locations, vtype=gp.GRB.BINARY, lb=0, ub=1, name='')
for i in locations:
    x[i, i].ub = 0
u = {}
for i in locations:
    if i == '1':
        u[i] = m.addVar(vtype=gp.GRB.INTEGER, lb=1, ub=1, name=f'u_{i}')
    else:
        u[i] = m.addVar(vtype=gp.GRB.INTEGER, lb=2, ub=n, name=f'u_{i}')
m.setObjective(gp.quicksum((d[i, j] * x[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name=f'leave_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name=f'enter_{j}')
for i in locations:
    if i == '1':
        continue
    for j in locations:
        if j == '1' or i == j:
            continue
        m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f'subtour_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    succ = {}
    for i in locations:
        for j in locations:
            if i != j and x[i, j].X > 0.5:
                succ[i] = j
    tour = ['1']
    while len(tour) < n + 1:
        last = tour[-1]
        next_loc = succ[last]
        tour.append(next_loc)
        if next_loc == '1':
            break
    print('--- Optimal Tour ---')
    print(' -> '.join(tour))
    print('--- Visit Order (u[i]) ---')
    for i in locations:
        print(f'Location {i}: position {int(round(u[i].X))}')
else:
    print(f'No optimal solution found. Status: {m.status}')