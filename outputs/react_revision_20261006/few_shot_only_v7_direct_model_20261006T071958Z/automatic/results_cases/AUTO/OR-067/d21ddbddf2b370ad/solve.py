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
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read CSV at {csv_path} with tried encodings.')
    if 'Unnamed: 0' not in df.columns:
        raise ValueError("Missing 'Unnamed: 0' column for managers in CSV.")
    managers = df['Unnamed: 0'].tolist()
    projects = [col for col in df.columns if col != 'Unnamed: 0']
    cost = {}
    for (idx, row) in df.iterrows():
        m = row['Unnamed: 0']
        cost[m] = {}
        for p in projects:
            val = row[p]
            try:
                c_val = float(val)
            except Exception:
                raise ValueError(f"Non-numeric or missing cost for manager '{m}', project '{p}': '{val}'")
            cost[m][p] = c_val
    if set(cost.keys()) != set(managers):
        raise ValueError('Mismatch in manager keys.')
    for m in managers:
        if set(cost[m].keys()) != set(projects):
            raise ValueError(f"Manager '{m}' missing project costs.")
    m_model = gp.Model('ManagerProjectAssignment')
    m_model.Params.MIPGap = 0.0001
    assignment_keys = [(m, p) for m in managers for p in projects]
    x_vars = m_model.addVars(assignment_keys, vtype=GRB.BINARY, name='')
    m_model.setObjective(gp.quicksum((cost[m][p] * x_vars[m, p] for m in managers for p in projects)), GRB.MINIMIZE)
    m_model.addConstrs((gp.quicksum((x_vars[m, p] for p in projects)) == 1 for m in managers), name='')
    m_model.addConstrs((gp.quicksum((x_vars[m, p] for m in managers)) == 1 for p in projects), name='')
    m_model.optimize()
    if m_model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m_model.ObjVal}')
        for var in m_model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m_model.Status}')
    return m_model
m = solve_problem()