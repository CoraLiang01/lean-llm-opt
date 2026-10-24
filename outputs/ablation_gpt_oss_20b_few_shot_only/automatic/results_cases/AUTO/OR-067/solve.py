import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    managers = df['Unnamed: 0'].tolist()
    projects = [col for col in df.columns if col != 'Unnamed: 0']
    cost = {}
    for m in managers:
        for p in projects:
            val_str = df.loc[df['Unnamed: 0'] == m, p].values[0]
            try:
                val = float(val_str)
            except ValueError:
                raise ValueError(f'Non-numeric cost for manager {m} project {p}: {val_str}')
            cost[m, p] = val
    for m in managers:
        for p in projects:
            if (m, p) not in cost:
                raise ValueError(f'Missing cost for manager {m} project {p}')
    m = gurobipy.Model('ManagerProjectAssignment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(m, p) for m in managers for p in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[m, p] * x[m, p] for m in managers for p in projects)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[m, p] for p in projects)) == 1 for m in managers), name='')
    m.addConstrs((gurobipy.quicksum((x[m, p] for m in managers)) == 1 for p in projects), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for var in m.getVars():
        print(var.VarName, var.X)
else:
    print(m.Status)