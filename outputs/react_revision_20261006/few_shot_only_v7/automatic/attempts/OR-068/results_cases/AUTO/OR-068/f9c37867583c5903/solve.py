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
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read CSV at {csv_path} with tried encodings.')
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
            try:
                cost[manager][project] = float(val)
            except Exception:
                raise ValueError(f"Non-numeric or missing cost for manager '{manager}', project '{project}': '{val}'")
    if len(managers) != 6 or len(projects) != 6:
        raise ValueError(f'Expected 6 managers and 6 projects, got {len(managers)} managers and {len(projects)} projects.')
    for manager in managers:
        if set(cost[manager].keys()) != set(projects):
            raise ValueError(f"Manager '{manager}' missing cost entries for some projects.")
    m = gp.Model('AP_Manager_Project')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in managers for j in projects]
    x_vars = m.addVars(x_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in projects)) == 1 for i in managers), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in managers)) == 1 for j in projects), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()