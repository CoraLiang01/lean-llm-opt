import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {csv_path} with tried encodings.')
    machine_col = df.columns[0]
    machines = df[machine_col].tolist()
    tasks = df.columns[1:].tolist()
    if len(machines) != 12 or len(tasks) != 12:
        raise ValueError(f'Expected 12 machines and 12 tasks, got {len(machines)} machines and {len(tasks)} tasks.')
    cost = {}
    for (idx, row) in df.iterrows():
        machine = row[machine_col]
        cost[machine] = {}
        for task in tasks:
            val = row[task]
            if pd.isnull(val):
                raise ValueError(f'Missing cost for machine {machine}, task {task}.')
            cost[machine][task] = float(val)
    for i in machines:
        for j in tasks:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for machine {i}, task {j}.')
    m = gp.Model('AP_12x12')
    x = m.addVars(machines, tasks, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in machines for j in tasks)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for j in tasks)) == 1 for i in machines), name='')
    m.addConstrs((gp.quicksum((x[i, j] for i in machines)) == 1 for j in tasks), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()