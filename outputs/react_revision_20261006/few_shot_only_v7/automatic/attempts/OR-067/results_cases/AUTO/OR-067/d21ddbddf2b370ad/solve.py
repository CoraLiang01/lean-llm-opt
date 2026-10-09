import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv'
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
        raise ValueError("Missing 'Unnamed: 0' column for manager identifiers.")
    managers = df['Unnamed: 0'].tolist()
    projects = [col for col in df.columns if col != 'Unnamed: 0']
    cost = {}
    for (idx, row) in df.iterrows():
        manager = row['Unnamed: 0']
        cost[manager] = {}
        for project in projects:
            val = row[project]
            if val == '':
                raise ValueError(f'Missing cost for manager {manager}, project {project}.')
            try:
                cost[manager][project] = float(val)
            except Exception:
                raise ValueError(f'Non-numeric cost for manager {manager}, project {project}: {val}')
    for manager in managers:
        for project in projects:
            if project not in cost[manager]:
                raise ValueError(f'Missing cost for manager {manager}, project {project}.')
    m = gp.Model('AP2_assignment')
    x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
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