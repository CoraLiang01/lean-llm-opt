import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [str(i) for i in range(1, 41)]
if not all((col in df.columns for col in task_cols)):
    missing = [col for col in task_cols if col not in df.columns]
    raise ValueError(f'Missing task columns in CSV: {missing}')
task_bi = {t: float(df.at[0, t]) for t in task_cols}
cpus = ['1', '2', '3']
cpu_speed = {'1': 1.33, '2': 2.0, '3': 2.66}
if set(cpu_speed.keys()) != set(cpus):
    raise ValueError('CPU identifiers mismatch.')
tasks = task_cols
processors = cpus

def solve_problem(tasks, processors, task_bi, cpu_speed):
    m = gp.Model('ParallelMachineMakespan')
    m.Params.MIPGap = 0.0001
    x = m.addVars([(t, p) for t in tasks for p in processors], vtype=gp.GRB.BINARY, name='')
    Cmax = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
    m.addConstrs((gp.quicksum((x[t, p] for p in processors)) == 1 for t in tasks), name='')
    for p in processors:
        m.addConstr(gp.quicksum((task_bi[t] / cpu_speed[p] * x[t, p] for t in tasks)) <= Cmax, name=f'makespan_{p}')
    m.setObjective(Cmax, gp.GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem(tasks, processors, task_bi, cpu_speed)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')