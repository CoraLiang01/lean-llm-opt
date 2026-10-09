import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in df.columns:
    raise KeyError("Missing required column 'Unnamed: 0' in manager_project_costs.csv")
managers = df['Unnamed: 0'].apply(lambda x: x.strip()).tolist()
project_cols = [col for col in df.columns if col != 'Unnamed: 0']
projects = [col.strip() for col in project_cols]
cost = {}
for (i, m) in enumerate(managers):
    for p in projects:
        val = df.loc[i, p]
        try:
            cost_val = int(val)
        except Exception:
            raise ValueError(f"Invalid cost value for manager '{m}', project '{p}': '{val}'")
        cost[m, p] = cost_val
for m in managers:
    for p in projects:
        if (m, p) not in cost:
            raise ValueError(f"Missing cost entry for manager '{m}', project '{p}'")
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[mgr, prj] * x_vars[mgr, prj] for mgr in managers for prj in projects)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[mgr, prj] for prj in projects)) == 1 for mgr in managers), name='')
m.addConstrs((gp.quicksum((x_vars[mgr, prj] for mgr in managers)) == 1 for prj in projects), name='')
m.optimize()