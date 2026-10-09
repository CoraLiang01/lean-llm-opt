import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            if enc == encodings[-1]:
                raise
            continue
    if 'Manager' not in df.columns:
        raise ValueError("Missing 'Manager' column in CSV.")
    managers = df['Manager'].tolist()
    project_cols = [col for col in df.columns if col.startswith('Project ') and col.endswith(' Cost')]
    if len(project_cols) != 7:
        raise ValueError('Expected 7 project cost columns, found: %s' % project_cols)
    projects = [col.replace(' Cost', '') for col in project_cols]
    cost = {}
    for (idx, row) in df.iterrows():
        manager = row['Manager']
        cost[manager] = {}
        for col in project_cols:
            project = col.replace(' Cost', '')
            val = row[col]
            try:
                cost_val = float(val)
            except Exception:
                raise ValueError(f"Non-numeric or missing cost for manager '{manager}', project '{project}': '{val}'")
            cost[manager][project] = cost_val
    if len(managers) != 7 or len(projects) != 7:
        raise ValueError(f'Expected 7 managers and 7 projects, got {len(managers)} managers and {len(projects)} projects.')
    for manager in managers:
        for project in projects:
            if project not in cost[manager]:
                raise ValueError(f"Missing cost for manager '{manager}', project '{project}'.")
    m = gp.Model('Manager_Project_Assignment')
    x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[manager][project] * x_vars[manager, project] for manager in managers for project in projects)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[manager, project] for project in projects)) == 1 for manager in managers), name='')
    m.addConstrs((gp.quicksum((x_vars[manager, project] for manager in managers)) == 1 for project in projects), name='')
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