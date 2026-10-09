import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', sep=',', dtype=str, keep_default_na=False)
manager_col = 'Unnamed: 0'
project_cols = [col for col in costs_df.columns if col != manager_col]
managers = [str(m).strip() for m in costs_df[manager_col].tolist()]
projects = [str(p).strip() for p in project_cols]
cost_matrix = {}
for (idx, row) in costs_df.iterrows():
    m = str(row[manager_col]).strip()
    for p in projects:
        val = row[p]
        try:
            cost = int(val)
        except Exception:
            raise ValueError(f"Non-numeric or missing cost for manager '{m}', project '{p}': {val}")
        cost_matrix[m, p] = cost
for m in managers:
    for p in projects:
        if (m, p) not in cost_matrix:
            raise ValueError(f"Missing cost entry for manager '{m}', project '{p}'")

def solve_assignment_problem(managers, projects, cost_matrix):
    m = gp.Model('ManagerProjectAssignment')
    assignment_vars = m.addVars([(mgr, prj) for mgr in managers for prj in projects], vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost_matrix[mgr, prj] * assignment_vars[mgr, prj] for mgr in managers for prj in projects)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((assignment_vars[mgr, prj] for prj in projects)) == 1 for mgr in managers), name='')
    m.addConstrs((gp.quicksum((assignment_vars[mgr, prj] for mgr in managers)) == 1 for prj in projects), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_assignment_problem(managers, projects, cost_matrix)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')