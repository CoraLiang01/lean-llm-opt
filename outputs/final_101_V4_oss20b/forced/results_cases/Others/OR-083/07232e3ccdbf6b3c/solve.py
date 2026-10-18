import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', index_col=0)
workers = list(df.index)
tasks = list(df.columns)
large_penalty = 1000000.0
times = df.copy()
times = times.fillna(large_penalty)
m = gp.Model('AssignmentProblem')
x = m.addVars(workers, tasks, vtype=gp.GRB.BINARY)
m.setObjective(gp.quicksum((times.at[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x[w, t] for w in workers)) == 1)
for w in workers:
    m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= 1)
m.addConstr(gp.quicksum((x[w, t] for w in workers for t in tasks)) == 10)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total working hours: {m.objVal:.2f}')
    print('\nAssignment plan:')
    for w in workers:
        for t in tasks:
            if x[w, t].X > 0.5:
                print(f'  Worker {w} -> Task {t} (time: {times.at[w, t]})')
else:
    print(f'No optimal solution found. Status: {m.status}')