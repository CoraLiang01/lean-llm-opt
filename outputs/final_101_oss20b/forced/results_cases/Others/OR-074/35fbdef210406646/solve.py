import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = df['Time'].tolist()
requirements = df['Requirement'].tolist()
n_periods = len(periods)
m = gp.Model('Staffing')
y = m.addVars(range(n_periods), vtype=gp.GRB.BINARY)
m.setObjective(gp.quicksum((y[s] for s in range(n_periods))), gp.GRB.MINIMIZE)
for p in range(n_periods):
    covering_shifts = gp.quicksum((y[s] for s in range(n_periods) if (p - s) % n_periods < 16))
    m.addConstr(covering_shifts >= requirements[p])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total number of staff: {int(round(m.objVal))}')
    print('\n--- Shift schedule (start period -> covered periods) ---')
    for s in range(n_periods):
        if y[s].X > 0.5:
            covered = [periods[(s + i) % n_periods] for i in range(16)]
            print(f"Shift starting at '{periods[s]}' covers: {covered}")
else:
    print(f'No optimal solution found. Status: {m.status}')