import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv', sep=',')
manager_col = 'Manager'
project_cost_cols = [col for col in costs_df.columns if re.match('Project \\d+ Cost', col)]
managers = list(costs_df[manager_col].astype(str))
projects = [col.replace(' Cost', '') for col in project_cost_cols]
cost = {}
for i, m in enumerate(managers):
    for j, p_col in enumerate(project_cost_cols):
        p = p_col.replace(' Cost', '')
        cost[m, p] = int(costs_df.loc[i, p_col])
if len(cost) != len(managers) * len(projects):
    raise ValueError('Cost matrix is incomplete: missing manager-project cost entries.')
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[m_, p_] * x[m_, p_] for m_ in managers for p_ in projects)), gp.GRB.MINIMIZE)
for p in projects:
    m.addConstr(gp.quicksum((x[m_, p] for m_ in managers)) == 1, name=f'assign_proj_{p}')
for m in managers:
    m.addConstr(gp.quicksum((x[m, p_] for p_ in projects)) == 1, name=f'assign_mgr_{m}')
m.optimize()