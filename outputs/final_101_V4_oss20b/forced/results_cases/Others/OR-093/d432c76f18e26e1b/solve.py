import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
cost_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv', sep=',')
machines = cost_df['Machine'].tolist()
tasks = cost_df.columns[1:].tolist()
cost = {(machine, task): cost_df.loc[cost_df['Machine'] == machine, task].values[0] for machine in machines for task in tasks}
m = gp.Model('Assignment')
x = m.addVars(machines, tasks, vtype=gp.GRB.BINARY)
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in machines for j in tasks)), gp.GRB.MINIMIZE)
for i in machines:
    m.addConstr(gp.quicksum((x[i, j] for j in tasks)) == 1)
for j in tasks:
    m.addConstr(gp.quicksum((x[i, j] for i in machines)) == 1)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('Optimal assignment:')
    for i in machines:
        for j in tasks:
            if x[i, j].X > 0.5:
                print(f'  Machine {i} -> Task {j}')
else:
    print(f'Optimization did not find an optimal solution. Status: {m.status}')