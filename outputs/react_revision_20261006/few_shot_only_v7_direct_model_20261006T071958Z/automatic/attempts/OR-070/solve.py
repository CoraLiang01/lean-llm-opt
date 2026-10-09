import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with tried encodings.')
    if 'Manager' not in df.columns:
        raise ValueError("Missing 'Manager' column in input data.")
    manager_col = 'Manager'
    project_cols = ['Project 1 Cost', 'Project 2 Cost', 'Project 3 Cost', 'Project 4 Cost', 'Project 5 Cost', 'Project 6 Cost', 'Project 7 Cost']
    for col in project_cols:
        if col not in df.columns:
            raise ValueError(f'Missing project cost column: {col}')
    managers = df[manager_col].tolist()
    projects = project_cols.copy()
    c = {}
    for (idx, row) in df.iterrows():
        m = row[manager_col]
        c[m] = {}
        for p in projects:
            val = row[p]
            try:
                c[m][p] = float(val)
            except ValueError:
                raise ValueError(f"Non-numeric cost for manager '{m}', project '{p}': '{val}'")
    if len(managers) != len(projects):
        raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(projects)}).')
    for m in managers:
        for p in projects:
            if p not in c[m]:
                raise ValueError(f"Missing cost entry for manager '{m}', project '{p}'.")
    m = gp.Model('manager_project_assignment')
    x_vars = m.addVars([(mgr, proj) for mgr in managers for proj in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[mgr][proj] * x_vars[mgr, proj] for mgr in managers for proj in projects)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[mgr, proj] for proj in projects)) == 1 for mgr in managers), name='')
    m.addConstrs((gp.quicksum((x_vars[mgr, proj] for mgr in managers)) == 1 for proj in projects), name='')
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