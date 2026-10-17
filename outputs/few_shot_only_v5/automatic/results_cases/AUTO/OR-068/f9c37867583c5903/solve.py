import pandas as pd
import gurobipy as gp
from gurobipy import GRB
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', dtype=str, keep_default_na=False)
cost_df = cost_df.rename(columns={cost_df.columns[0]: 'Manager'})
managers = list(cost_df['Manager'])
projects = [col for col in cost_df.columns if col != 'Manager']
cost = {}
for _, row in cost_df.iterrows():
    manager = row['Manager']
    cost[manager] = {}
    for project in projects:
        try:
            cost[manager][project] = int(row[project])
        except ValueError:
            raise ValueError(f'Missing or invalid cost for manager {manager}, project {project}')
if set(cost.keys()) != set(managers):
    raise ValueError('Mismatch in manager identifiers between CSV and cost dictionary')
for m in managers:
    if set(cost[m].keys()) != set(projects):
        raise ValueError(f'Mismatch in project identifiers for manager {m}')
m = gp.Model('AP3_assignment')
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