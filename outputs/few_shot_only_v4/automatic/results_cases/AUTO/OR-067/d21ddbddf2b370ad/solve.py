import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
managers = list(df.iloc[:, 0])
projects = list(df.columns[1:])
cost = {}
for idx, row in df.iterrows():
    manager = row.iloc[0]
    cost[manager] = {}
    for proj in projects:
        val = row[proj]
        try:
            cost[manager][proj] = float(val)
        except ValueError:
            raise ValueError(f"Non-numeric cost for manager '{manager}', project '{proj}': '{val}'")
for i in managers:
    for j in projects:
        if j not in cost[i]:
            raise ValueError(f"Missing cost for manager '{i}', project '{j}'")
m = gp.Model('AP_Manager_Project')
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