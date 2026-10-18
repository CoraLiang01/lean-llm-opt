import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
bi_row = df.loc[df['Process'] == 'BI'].iloc[0]
BI = {int(col): float(bi_row[col]) for col in df.columns if col != 'Process'}
J = list(BI.keys())
M = [1, 2, 3]
freq = {1: 1.33, 2: 2.0, 3: 2.66}
p = {(j, m): BI[j] / freq[m] for j in J for m in M}
m = gp.Model('UnrelatedParallelScheduling')
x = m.addVars(J, M, vtype=gp.GRB.BINARY)
Cmax = m.addVar(vtype=gp.GRB.CONTINUOUS)
for j in J:
    m.addConstr(gp.quicksum((x[j, m] for m in M)) == 1)
for m_cpu in M:
    m.addConstr(gp.quicksum((p[j, m_cpu] * x[j, m_cpu] for j in J)) <= Cmax)
m.setObjective(Cmax, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (Cmax): {Cmax.X:.4f} seconds')
    print('\nTask assignments:')
    for j in J:
        assigned_cpu = None
        for m_cpu in M:
            if x[j, m_cpu].X > 0.5:
                assigned_cpu = m_cpu
                break
        if assigned_cpu is not None:
            print(f'  Task {j:2d} -> CPU {assigned_cpu} (time {p[j, assigned_cpu]:.4f}s)')
    print('\nCPU loads:')
    for m_cpu in M:
        load = sum((p[j, m_cpu] * x[j, m_cpu].X for j in J))
        print(f'  CPU {m_cpu}: {load:.4f} seconds')
else:
    print(f'Optimization ended with status {m.status}')