import gurobipy as gp
import pandas as pd
import numpy as np
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', sep=',')
managers = costs_df['Unnamed: 0'].astype(str).str.strip().tolist()
projects = [col for col in costs_df.columns if col != 'Unnamed: 0']
cost = {}
for idx, row in costs_df.iterrows():
    m = str(row['Unnamed: 0']).strip()
    for p in projects:
        cost[m, p] = float(row[p])
for m in managers:
    for p in projects:
        if (m, p) not in cost:
            raise ValueError(f'Missing cost for manager {m}, project {p}')
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[m_, p_] * x[m_, p_] for m_ in managers for p_ in projects)), gp.GRB.MINIMIZE)
for m_ in managers:
    m.addConstr(gp.quicksum((x[m_, p_] for p_ in projects)) == 1, name=f'assign_mgr_{m_}')
for p_ in projects:
    m.addConstr(gp.quicksum((x[m_, p_] for m_ in managers)) == 1, name=f'assign_proj_{p_}')
m.optimize()