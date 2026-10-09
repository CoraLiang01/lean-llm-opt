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
    for col in project_cols_sorted:
        try:
            cost_val = int(row[col])
        except Exception as e:
            raise ValueError(f"Invalid cost value for manager '{manager}', project column '{col}': {row[col]}")
        cost[manager, col] = cost_val
if len(managers) != len(project_cols_sorted):
    raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(project_cols_sorted)}).')
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(managers, project_cols_sorted, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[manager, project_col] * x_vars[manager, project_col] for manager in managers for project_col in project_cols_sorted)), gp.GRB.MINIMIZE)
for project_col in project_cols_sorted:
    m.addConstr(gp.quicksum((x_vars[manager, project_col] for manager in managers)) == 1)
for manager in managers:
    m.addConstr(gp.quicksum((x_vars[manager, project_col] for project_col in project_cols_sorted)) == 1)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.0f}')
    print('--- Assignment ---')
    for manager in managers:
        for (project_col, project_num) in zip(project_cols_sorted, project_numbers):
            if x_vars[manager, project_col].X > 0.5:
                print(f'{manager} assigned to Project {project_num} (Cost: {cost[manager, project_col]})')
else:
    print(f'No optimal solution found. Status: {m.status}')