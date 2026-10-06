import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not read CSV file {csv_path} with tried encodings.')
    if 'Machine' not in df.columns:
        raise ValueError("CSV must have a 'Machine' column for machine labels.")
    machines = df['Machine'].tolist()
    tasks = [col for col in df.columns if col != 'Machine']
    if len(machines) != 12 or len(tasks) != 12:
        raise ValueError('Expected 12 machines and 12 tasks, got {} machines and {} tasks.'.format(len(machines), len(tasks)))
    cost = {}
    for (idx, row) in df.iterrows():
        machine = row['Machine']
        cost[machine] = {}
        for task in tasks:
            if pd.isnull(row[task]):
                raise ValueError(f'Missing cost for machine {machine}, task {task}.')
            cost[machine][task] = float(row[task])
    for machine in machines:
        for task in tasks:
            if task not in cost[machine]:
                raise ValueError(f'Missing cost for machine {machine}, task {task}.')
    m = gp.Model('AP_12x12')
    x_keys = [(i, j) for i in machines for j in tasks]
    x = m.addVars(x_keys, vtype=GRB.BINARY, name='')
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