import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',')
managers = df['Manager'].astype(str).tolist()
project_cost_cols = [col for col in df.columns if re.match('Project \\d+ Cost$', col)]
projects = [re.match('(Project \\d+) Cost$', col).group(1) for col in project_cost_cols]
cost = {}
for (idx, row) in df.iterrows():
    manager = str(row['Manager'])
    for (col, project) in zip(project_cost_cols, projects):
        val = row[col]
        if pd.isnull(val):
            raise ValueError(f"Missing cost for manager '{manager}', project '{project}'")
        cost[manager, project] = float(val)
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) and projects ({len(projects)}) must be equal for assignment.')
for m in managers:
    for p in projects:
        if (m, p) not in cost:
            raise ValueError(f"Missing cost entry for manager '{m}', project '{p}'.")
m = gp.Model('ManagerProjectAssignment')
x = m.addVars([(m_id, p_id) for m_id in managers for p_id in projects], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[m_id, p_id] * x[m_id, p_id] for m_id in managers for p_id in projects)), gp.GRB.MINIMIZE)
for p_id in projects:
    m.addConstr(gp.quicksum((x[m_id, p_id] for m_id in managers)) == 1, name=f"assign_proj_{p_id.replace(' ', '_')}")
for m_id in managers:
    m.addConstr(gp.quicksum((x[m_id, p_id] for p_id in projects)) == 1, name=f"assign_mgr_{re.sub('[^A-Za-z0-9]', '', m_id)}")
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (m_id, p_id) in x.keys():
        print(f'{x[m_id, p_id].VarName} {x[m_id, p_id].X}')
else:
    print(f'Solver status: {m.status}')