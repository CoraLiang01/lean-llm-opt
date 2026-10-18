import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
manager_col = df.columns[0]
managers = df[manager_col].tolist()
project_cols = df.columns[1:]
projects = [col.replace(' Cost', '') for col in project_cols]
cost = {}
for idx, row in df.iterrows():
    manager = row[manager_col]
    cost[manager] = {}
    for proj_col, proj in zip(project_cols, projects):
        try:
            cost_val = int(row[proj_col])
        except ValueError:
            raise ValueError(f'Non-integer cost for {manager}, {proj}: {row[proj_col]}')
        cost[manager][proj] = cost_val
if set(cost.keys()) != set(managers):
    raise ValueError('Mismatch in manager identifiers between cost data and manager list')
for m in managers:
    if set(cost[m].keys()) != set(projects):
        raise ValueError(f'Mismatch in project identifiers for manager {m}')
m = gp.Model('AP5_Manager_Project_Assignment')
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