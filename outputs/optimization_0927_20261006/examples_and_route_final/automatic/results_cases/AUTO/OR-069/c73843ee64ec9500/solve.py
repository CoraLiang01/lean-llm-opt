import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv', dtype=str, keep_default_na=False)
manager_col = 'Manager'
manager_ids = costs_df[manager_col].tolist()
project_cols = [col for col in costs_df.columns if col != manager_col]
project_id_pattern = re.compile('^(Project \\d+) Cost$')
project_ids = []
project_col_to_id = {}
for col in project_cols:
    m = project_id_pattern.match(col)
    if not m:
        raise ValueError(f"Column '{col}' does not match expected project cost pattern.")
    proj_id = m.group(1)
    project_ids.append(proj_id)
    project_col_to_id[col] = proj_id
cost = {}
for (idx, row) in costs_df.iterrows():
    manager = row[manager_col]
    for col in project_cols:
        project = project_col_to_id[col]
        val = row[col]
        try:
            cost_val = float(val)
        except Exception:
            raise ValueError(f"Invalid cost value for manager '{manager}', project '{project}': '{val}'")
        cost[manager, project] = cost_val
if len(manager_ids) != len(project_ids):
    raise ValueError(f'Number of managers ({len(manager_ids)}) does not match number of projects ({len(project_ids)}); assignment problem requires a square cost matrix.')
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(manager_ids, project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i, j] * x_vars[i, j] for i in manager_ids for j in project_ids)), gp.GRB.MINIMIZE)
for i in manager_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in project_ids)) == 1, name='')
for j in project_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in manager_ids)) == 1, name='')
m.optimize()