import gurobipy as gp
import pandas as pd
import numpy as np
import math
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv', sep=',')
location_ids = [str(i) for i in range(1, 16)]
n = len(location_ids)
distance = {}
for i_row, row in distance_df.iterrows():
    i = str(int(row['Unnamed: 0']))
    for j in location_ids:
        if i == j:
            continue
        val = row.get(j, np.nan)
        if pd.notnull(val):
            distance[i, j] = float(val)
for i in location_ids:
    for j in location_ids:
        if i == j:
            continue
        if (i, j) not in distance and (j, i) in distance:
            distance[i, j] = distance[j, i]
        if (i, j) not in distance:
            raise ValueError(f'Missing distance between {i} and {j} in CSV.')

def solve_tsp(location_ids, distance):
    n = len(location_ids)
    m = gp.Model('TSP')
    x = m.addVars([(i, j) for i in location_ids for j in location_ids if i != j], vtype=gp.GRB.BINARY, name='')
    u = m.addVars([i for i in location_ids if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
    m.setObjective(gp.quicksum((distance[i, j] * x[i, j] for i in location_ids for j in location_ids if i != j)), gp.GRB.MINIMIZE)
    for i in location_ids:
        m.addConstr(gp.quicksum((x[i, j] for j in location_ids if j != i)) == 1, name=f'depart_{i}')
    for j in location_ids:
        m.addConstr(gp.quicksum((x[i, j] for i in location_ids if i != j)) == 1, name=f'arrive_{j}')
    for i in location_ids:
        if i == '1':
            continue
        for j in location_ids:
            if j == '1' or i == j:
                continue
            m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f'mtz_{i}_{j}')
    m.optimize()
    return m
m = solve_tsp(location_ids, distance)