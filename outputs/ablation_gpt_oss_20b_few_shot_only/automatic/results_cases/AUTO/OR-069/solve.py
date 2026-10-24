import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv'
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    managers = frame['Manager'].tolist()
    project_cols = [col for col in frame.columns if col != 'Manager']
    projects = [col.replace(' Cost', '') for col in project_cols]
    cost = {}
    for (idx, m) in enumerate(managers):
        for (col, p) in zip(project_cols, projects):
            val = frame.at[idx, col]
            try:
                cost[m, p] = float(val)
            except ValueError:
                raise ValueError(f'Non-numeric cost for manager {m}, project {p}: {val}')
    for m in managers:
        for p in projects:
            if (m, p) not in cost:
                raise ValueError(f'Missing cost for manager {m}, project {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(m_id, p_id) for m_id in managers for p_id in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[m_id, p_id] * x[m_id, p_id] for m_id in managers for p_id in projects)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[m_id, p_id] for p_id in projects)) == 1 for m_id in managers), name='')
    m.addConstrs((gurobipy.quicksum((x[m_id, p_id] for m_id in managers)) == 1 for p_id in projects), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for var in m.getVars():
            print(var.VarName, var.X)
    else:
        print(m.Status)
    return m
m = solve_problem()