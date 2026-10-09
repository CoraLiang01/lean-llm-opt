import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',', dtype=str, keep_default_na=False)
manager_col = 'Manager'
project_cost_cols = [col for col in df.columns if re.match('Project \\d+ Cost', col)]
managers = df[manager_col].tolist()
projects = [re.match('Project (\\d+) Cost', col).group(1) for col in project_cost_cols]
project_ids = [f'Project {num}' for num in projects]
if len(managers) != len(project_ids):
    raise ValueError(f'Number of managers ({len(managers)}) and projects ({len(project_ids)}) must be equal for assignment.')
cost = {}
for (i, row) in df.iterrows():
    manager = row[manager_col]
    for (col, proj_num) in zip(project_cost_cols, projects):
        project = f'Project {proj_num}'
        val = row[col]
        try:
            cost_val = int(val)
        except Exception:
            raise ValueError(f"Invalid cost value for manager '{manager}', project '{project}': '{val}'")
        cost[manager, project] = cost_val
for m in managers:
    for p in project_ids:
        if (m, p) not in cost:
            raise ValueError(f"Missing cost for manager '{m}', project '{p}'.")

def solve_assignment_problem(managers, projects, cost):
    m = gp.Model('ManagerProjectAssignment')
    x_vars = m.addVars([(m_id, p_id) for m_id in managers for p_id in projects], vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[m_id, p_id] * x_vars[m_id, p_id] for m_id in managers for p_id in projects)), gp.GRB.MINIMIZE)
    for p_id in projects:
        m.addConstr(gp.quicksum((x_vars[m_id, p_id] for m_id in managers)) == 1, name=f"assign_proj_{p_id.replace(' ', '')}")
    for m_id in managers:
        m.addConstr(gp.quicksum((x_vars[m_id, p_id] for p_id in projects)) == 1, name=f"assign_mgr_{re.sub('[^A-Za-z0-9]', '', m_id)}")
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_assignment_problem(managers, project_ids, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')