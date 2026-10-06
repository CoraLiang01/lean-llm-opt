import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',')
row_ids = df['Unnamed: 0'].astype(str).tolist()
col_ids = [col for col in df.columns if col != 'Unnamed: 0']
locations = [str(i) for i in range(1, 16)]
distance = {}
for i in locations:
    distance[i] = {}
    for j in locations:
        if i == j:
            distance[i][j] = 0.0
        else:
            row_match = df[df['Unnamed: 0'].astype(str) == i]
            if not row_match.empty and j in df.columns:
                val = row_match.iloc[0][j]
                if pd.isna(val):
                    row_match2 = df[df['Unnamed: 0'].astype(str) == j]
                    if not row_match2.empty and i in df.columns:
                        val2 = row_match2.iloc[0][i]
                        if pd.isna(val2):
                            raise ValueError(f'Missing distance between {i} and {j} in both directions.')
                        else:
                            distance[i][j] = float(val2)
                    else:
                        raise ValueError(f'Missing distance between {i} and {j} in both directions.')
                else:
                    distance[i][j] = float(val)
            else:
                row_match2 = df[df['Unnamed: 0'].astype(str) == j]
                if not row_match2.empty and i in df.columns:
                    val2 = row_match2.iloc[0][i]
                    if pd.isna(val2):
                        raise ValueError(f'Missing distance between {i} and {j} in both directions.')
                    else:
                        distance[i][j] = float(val2)
                else:
                    raise ValueError(f'Missing distance between {i} and {j} in both directions.')
m = gp.Model('TSP_15_Cities')
x = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
u = m.addVars([i for i in locations if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=15, name='')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
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
        m.addConstr(u[i] - u[j] + 15 * x[i, j] <= 14, name=f'subtour_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    next_city = {}
    for i in locations:
        for j in locations:
            if i != j and x[i, j].X > 0.5:
                next_city[i] = j
    tour = ['1']
    while True:
        last = tour[-1]
        if last in next_city:
            nxt = next_city[last]
            if nxt == '1':
                tour.append('1')
                break
            elif nxt in tour:
                print('Warning: Detected a subtour or loop.')
                break
            else:
                tour.append(nxt)
        else:
            print('Warning: Incomplete tour.')
            break
    print('--- Optimal Tour ---')
    print(' -> '.join(tour))
    print('--- Tour Order (u[i]) ---')
    for i in sorted([i for i in locations if i != '1'], key=lambda x: u[x].X):
        print(f'Location {i}: position {int(u[i].X)}')
else:
    print(f'No optimal solution found. Status: {m.status}')