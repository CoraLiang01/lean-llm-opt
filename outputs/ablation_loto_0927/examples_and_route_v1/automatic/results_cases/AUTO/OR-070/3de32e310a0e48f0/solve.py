import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',')
managers = df['Manager'].astype(str).tolist()
project_cost_cols = [col for col in df.columns if re.match('Project\\s+\\d+\\s+Cost', col)]

def project_num(col):
    m = re.match('Project\\s+(\\d+)\\s+Cost', col)
    return int(m.group(1)) if m else 0
project_cost_cols = sorted(project_cost_cols, key=project_num)
projects = [col for col in project_cost_cols]
cost = {}
for (i, row) in df.iterrows():
    m_id = str(row['Manager'])
    for p_col in project_cost_cols:
        cost[m_id, p_col] = int(row[p_col])
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) and projects ({len(projects)}) must be equal for a one-to-one assignment.')
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[m_id, p_col] * x[m_id, p_col] for m_id in managers for p_col in projects)), gp.GRB.MINIMIZE)
for p_col in projects:
    m.addConstr(gp.quicksum((x[m_id, p_col] for m_id in managers)) == 1, name=f'assign_proj_{p_col}')
for m_id in managers:
    m.addConstr(gp.quicksum((x[m_id, p_col] for p_col in projects)) == 1, name=f'assign_mgr_{m_id}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.2f}')
    print('--- Assignment ---')
    for m_id in managers:
        for p_col in projects:
            if x[m_id, p_col].X > 0.5:
                proj_match = re.match('Project\\s+(\\d+)\\s+Cost', p_col)
                proj_disp = f'Project {proj_match.group(1)}' if proj_match else p_col
                print(f'{m_id} assigned to {proj_disp} (Cost: {cost[m_id, p_col]})')
else:
    print(f'No optimal solution found. Status: {m.status}')