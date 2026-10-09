import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',', dtype=str, keep_default_na=False)
managers = df['Manager'].tolist()
project_cost_col_pattern = re.compile('^Project\\s+(\\d+)\\s+Cost$', re.IGNORECASE)
project_cols = [col for col in df.columns if project_cost_col_pattern.match(col)]
project_cols_sorted = sorted(project_cols, key=lambda c: int(project_cost_col_pattern.match(c).group(1)))
projects = [project_cost_col_pattern.match(col).group(0) for col in project_cols_sorted]
project_numbers = [project_cost_col_pattern.match(col).group(1) for col in project_cols_sorted]
cost = {}
for (idx, row) in df.iterrows():
    manager = row['Manager']
    cost[manager] = {}
    for col in project_cols_sorted:
        try:
            cost_val = int(row[col])
        except Exception as e:
            raise ValueError(f"Invalid cost value for manager '{manager}', project column '{col}': {row[col]}")
        cost[manager][col] = cost_val
if len(managers) != len(project_cols_sorted):
    raise ValueError(f'Number of managers ({len(managers)}) and projects ({len(project_cols_sorted)}) must be equal for a one-to-one assignment.')
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(managers, project_cols_sorted, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[mgr][prj] * x_vars[mgr, prj] for mgr in managers for prj in project_cols_sorted)), gp.GRB.MINIMIZE)
for prj in project_cols_sorted:
    m.addConstr(gp.quicksum((x_vars[mgr, prj] for mgr in managers)) == 1, name=f'assign_proj_{prj}')
for mgr in managers:
    m.addConstr(gp.quicksum((x_vars[mgr, prj] for prj in project_cols_sorted)) == 1, name=f'assign_mgr_{mgr}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.0f}')
    print('--- Assignment ---')
    for mgr in managers:
        for (prj, prj_num) in zip(project_cols_sorted, project_numbers):
            if x_vars[mgr, prj].X > 0.5:
                print(f'{mgr} assigned to Project {prj_num} (Cost: {cost[mgr][prj]})')
else:
    print(f'No optimal solution found. Status: {m.status}')