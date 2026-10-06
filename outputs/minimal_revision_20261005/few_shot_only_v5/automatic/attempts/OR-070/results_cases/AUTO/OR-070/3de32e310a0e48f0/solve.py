import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read CSV at {csv_path} with tried encodings.')
    if 'Manager' not in df.columns:
        raise ValueError("Missing 'Manager' column in CSV.")
    managers = df['Manager'].tolist()
    project_cols = [col for col in df.columns if col != 'Manager']
    if len(project_cols) != 7:
        raise ValueError('Expected 7 project columns, found: %s' % project_cols)
    projects = [col for col in project_cols]
    cost = {}
    for (idx, row) in df.iterrows():
        manager = row['Manager']
        cost[manager] = {}
        for project in projects:
            if pd.isnull(row[project]):
                raise ValueError(f'Missing cost for manager {manager}, project {project}')
            cost[manager][project] = float(row[project])
    if len(managers) != 7 or len(projects) != 7:
        raise ValueError('Expected 7 managers and 7 projects.')
    m = gp.Model('Manager_Project_Assignment')
    x_keys = [(i, j) for i in managers for j in projects]
    x = m.addVars(x_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for j in projects)) == 1 for i in managers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for i in managers)) == 1 for j in projects), name='')
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