import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
cpus = [1, 2, 3]
cpu_freq = {1: 1.33, 2: 2.0, 3: 2.66}
task_cols = [col for col in df.columns if col != 'Process']
tasks = [int(col) for col in task_cols]
bi_row = df.iloc[0]
bi = {t: float(bi_row[str(t)]) for t in tasks}
if set(tasks) != set(range(1, 41)):
    raise ValueError('Tasks in CSV do not match required set 1..40')
if set(cpu_freq.keys()) != set(cpus):
    raise ValueError('CPU identifiers do not match required set')

def solve_problem(tasks, cpus, bi, cpu_freq):
    m = gp.Model('TaskAssignmentMinMakespan')
    x = m.addVars([(t, p) for t in tasks for p in cpus], vtype=gp.GRB.BINARY, name='')
    C = m.addVars(cpus, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    Cmax = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
    m.addConstrs((gp.quicksum((x[t, p] for p in cpus)) == 1 for t in tasks), name='')
    for p in cpus:
        m.addConstr(C[p] == gp.quicksum((bi[t] / cpu_freq[p] * x[t, p] for t in tasks)), name=f'cpu_time_{p}')
    m.addConstrs((C[p] <= Cmax for p in cpus), name='')
    m.setObjective(Cmax, gp.GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(tasks, cpus, bi, cpu_freq)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')