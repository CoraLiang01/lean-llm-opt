import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read CSV at {csv_path} with tried encodings.')
    if 'Machine' not in df.columns:
        raise ValueError("Column 'Machine' not found in CSV.")
    machines = df['Machine'].tolist()
    tasks = [col for col in df.columns if col != 'Machine']
    cost = {}
    for (idx, row) in df.iterrows():
        i = row['Machine']
        cost[i] = {}
        for j in tasks:
            val = row[j]
            try:
                cij = float(val)
            except Exception:
                raise ValueError(f"Non-numeric or missing cost for Machine '{i}', Task '{j}': '{val}'")
            cost[i][j] = cij
    if set(cost.keys()) != set(machines):
        raise ValueError('Mismatch in machine identifiers between data and cost dictionary.')
    for i in machines:
        if set(cost[i].keys()) != set(tasks):
            raise ValueError(f"Mismatch in task identifiers for machine '{i}'.")
    m = gp.Model('AP_Machine_Task')
    m.Params.MIPGap = 0.0001
    assignment_keys = [(i, j) for i in machines for j in tasks]
    x_vars = m.addVars(assignment_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for (i, j) in assignment_keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in tasks)) == 1 for i in machines), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in machines)) == 1 for j in tasks), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()