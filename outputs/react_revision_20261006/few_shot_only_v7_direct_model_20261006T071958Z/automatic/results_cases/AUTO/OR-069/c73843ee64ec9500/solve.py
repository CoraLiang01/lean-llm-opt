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
        raise RuntimeError(f'Could not decode {csv_path} with tried encodings.')
    if 'Manager' not in df.columns:
        raise ValueError("Missing required column 'Manager' in input CSV.")
    project_cols = [col for col in df.columns if col.startswith('Project ') and col.endswith(' Cost')]
    if not project_cols:
        raise ValueError('No project cost columns found in input CSV.')
    managers = df['Manager'].tolist()
    projects = project_cols
    cost = {}
    for (idx, row) in df.iterrows():
        manager = row['Manager']
        if manager not in cost:
            cost[manager] = {}
        for project in projects:
            val = row[project]
            try:
                cij = float(val)
            except ValueError:
                raise ValueError(f"Non-numeric cost for manager '{manager}', project '{project}': '{val}'")
            cost[manager][project] = cij
    if len(managers) != len(projects):
        raise ValueError(f'Number of managers ({len(managers)}) and projects ({len(projects)}) must be equal for one-to-one assignment.')
    m = gp.Model('ManagerProjectAssignment')
    assignment_keys = [(i, j) for i in managers for j in projects]
    x_vars = m.addVars(assignment_keys, vtype=GRB.BINARY, name='')
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