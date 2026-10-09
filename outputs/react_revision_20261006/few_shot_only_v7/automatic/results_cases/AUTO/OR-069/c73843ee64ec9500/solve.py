import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode CSV at {csv_path} with tried encodings.')
    if 'Manager' not in df.columns:
        raise ValueError("Missing 'Manager' column in CSV.")
    manager_ids = df['Manager'].tolist()
    project_cols = [col for col in df.columns if col.startswith('Project ') and col.endswith(' Cost')]
    if len(project_cols) != 11:
        raise ValueError(f'Expected 11 project cost columns, found {len(project_cols)}: {project_cols}')
    project_ids = [col.replace(' Cost', '') for col in project_cols]
    cost = {}
    for (idx, row) in df.iterrows():
        manager = row['Manager']
        cost[manager] = {}
        for (col, project) in zip(project_cols, project_ids):
            val = row[col]
            try:
                cost[manager][project] = float(val)
            except ValueError:
                raise ValueError(f"Non-numeric cost for manager '{manager}', project '{project}': '{val}'")
    if len(manager_ids) != 11 or len(project_ids) != 11:
        raise ValueError(f'Expected 11 managers and 11 projects, got {len(manager_ids)} managers and {len(project_ids)} projects.')
    for manager in manager_ids:
        for project in project_ids:
            if project not in cost[manager]:
                raise ValueError(f"Missing cost for manager '{manager}', project '{project}'.")
    m = gp.Model('Manager_Project_Assignment')
    assignment_keys = [(manager, project) for manager in manager_ids for project in project_ids]
    x_vars = m.addVars(assignment_keys, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[manager][project] * x_vars[manager, project] for (manager, project) in assignment_keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[manager, project] for project in project_ids)) == 1 for manager in manager_ids), name='')
    m.addConstrs((gp.quicksum((x_vars[manager, project] for manager in manager_ids)) == 1 for project in project_ids), name='')
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