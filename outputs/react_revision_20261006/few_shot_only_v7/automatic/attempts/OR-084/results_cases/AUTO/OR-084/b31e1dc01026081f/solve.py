import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_cols = [col for col in df.columns if re.fullmatch('\\d+', col)]
tasks = [int(col) for col in task_cols]
if df.shape[0] != 1:
    raise ValueError(f'Expected exactly one row in 18.csv, got {df.shape[0]}')
row = df.iloc[0]
try:
    bi_dict = {t: float(row[str(t)]) for t in tasks}
except Exception as e:
    raise ValueError(f'Failed to parse BI values for all tasks: {e}')
cpus = [1, 2, 3]
cpu_freq = {1: 1.33, 2: 2.0, 3: 2.66}
if set(tasks) != set(range(1, 41)):
    raise ValueError(f'Tasks in CSV do not match expected set 1..40: {tasks}')
if set(cpu_freq.keys()) != set(cpus):
    raise ValueError('CPU frequency mapping does not match CPUs.')
proc_time = {}
for t in tasks:
    for p in cpus:
        proc_time[t, p] = bi_dict[t] / cpu_freq[p]

def solve_problem(tasks, cpus, proc_time):
    m = gp.Model('TaskAssignmentMinMakespan')
    x_vars = m.addVars([(t, p) for t in tasks for p in cpus], vtype=gp.GRB.BINARY, name='')
    C_vars = m.addVars(cpus, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    Cmax = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
    m.addConstrs((gp.quicksum((x_vars[t, p] for p in cpus)) == 1 for t in tasks), name='')
    m.addConstrs((C_vars[p] == gp.quicksum((proc_time[t, p] * x_vars[t, p] for t in tasks)) for p in cpus), name='')
    m.addConstrs((C_vars[p] <= Cmax for p in cpus), name='')
    m.setObjective(Cmax, gp.GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(tasks, cpus, proc_time)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')