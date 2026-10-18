import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
managers = df.iloc[:, 0].tolist()
projects = [col.replace(' Cost', '') for col in df.columns[1:]]
cost = {}
for idx, row in df.iterrows():
    manager = row.iloc[0]
    cost[manager] = {}
    for j, project in enumerate(projects):
        val = row.iloc[j + 1]
        try:
            cost[manager][project] = int(val)
        except ValueError:
            raise ValueError(f"Non-integer cost for manager '{manager}', project '{project}': '{val}'")
if set(cost.keys()) != set(managers):
    raise ValueError('Mismatch in manager identifiers between CSV and cost dictionary.')
for manager in managers:
    if set(cost[manager].keys()) != set(projects):
        raise ValueError(f"Manager '{manager}' missing project costs.")
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