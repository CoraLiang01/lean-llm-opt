import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
distance_file = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_file, sep=',')
df.set_index('Unnamed: 0', inplace=True)
nodes = ['Depot', 'A', 'B', 'C']
dist = {}
for i in nodes:
    for j in nodes:
        if i != j:
            dist[i, j] = int(df.loc[i, j])
m = gp.Model('TSP')
x = m.addVars([(i, j) for i in nodes for j in nodes if i != j], vtype=gp.GRB.BINARY)
u = m.addVars([i for i in nodes if i != 'Depot'], vtype=gp.GRB.INTEGER)
for i in u:
    u[i].lb = 1
    u[i].ub = 3
for i in nodes:
    m.addConstr(gp.quicksum((x[i, j] for j in nodes if j != i)) == 1)
for j in nodes:
    m.addConstr(gp.quicksum((x[i, j] for i in nodes if i != j)) == 1)
for i in nodes:
    if i == 'Depot':
        continue
    for j in nodes:
        if j == 'Depot' or i == j:
            continue
        m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2)
m.setObjective(gp.quicksum((dist[i, j] * x[i, j] for i in nodes for j in nodes if i != j)), sense=gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_distance = m.objVal
    tour = ['Depot']
    current = 'Depot'
    while len(tour) < len(nodes):
        next_node = [j for j in nodes if j != current and x[current, j].X > 0.5][0]
        tour.append(next_node)
        current = next_node
    tour.append('Depot')
    print(f'Optimal total travel distance: {total_distance:.2f} km')
    print('Optimal visiting sequence:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')