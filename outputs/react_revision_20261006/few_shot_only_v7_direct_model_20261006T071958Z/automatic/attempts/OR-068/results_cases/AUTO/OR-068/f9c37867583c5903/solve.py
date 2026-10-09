import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {csv_path} with tried encodings.')
    if 'Unnamed: 0' not in df.columns:
        raise ValueError("Missing required column 'Unnamed: 0' in input CSV.")
    managers = df['Unnamed: 0'].tolist()
    projects = [col for col in df.columns if col != 'Unnamed: 0']
    cost = {}
    for (idx, row) in df.iterrows():
        i = row['Unnamed: 0']
        cost[i] = {}
        for j in projects:
            val = row[j]
            try:
                cij = float(val)
            except ValueError:
                raise ValueError(f"Non-numeric cost at manager '{i}', project '{j}': '{val}'")
            cost[i][j] = cij
    if set(cost.keys()) != set(managers):
        raise ValueError('Mismatch between managers and cost keys.')
    for i in managers:
        if set(cost[i].keys()) != set(projects):
            raise ValueError(f"Manager '{i}' missing cost entries for all projects.")
    m = gp.Model('Manager_Project_Assignment')
    x_vars = m.addVars([(i, j) for i in managers for j in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in projects)) == 1 for i in managers), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in managers)) == 1 for j in projects), name='')
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