import gurobipy as gp
import pandas as pd
import numpy as np
workstation_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant15/inputs/workstation_capacity.csv'
assignment_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant15/inputs/assignment_costs.csv'
assignment_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant15/inputs/assignment_resources.csv'
df_capacity = pd.read_csv(workstation_capacity_path, sep=',')
df_capacity['Workstation'] = df_capacity['Workstation'].astype(str).str.strip()
workstations = df_capacity['Workstation'].unique().tolist()
capacity = {}
for (_, row) in df_capacity.iterrows():
    ws = str(row['Workstation']).strip()
    if ws in capacity:
        raise ValueError(f'Duplicate workstation in capacity file: {ws}')
    capacity[ws] = int(row['Capacity'])
df_costs = pd.read_csv(assignment_costs_path, sep=',')
df_costs['Workstation'] = df_costs['Workstation'].astype(str).str.strip()
job_cols = [col for col in df_costs.columns if col != 'Workstation']
jobs = job_cols.copy()
cost_ws_set = set(df_costs['Workstation'])
if set(workstations) != cost_ws_set:
    raise ValueError(f'Workstation mismatch between capacity and costs: {set(workstations)} vs {cost_ws_set}')
cost = {}
for (_, row) in df_costs.iterrows():
    ws = str(row['Workstation']).strip()
    for job in jobs:
        cost[ws, job] = float(row[job])
df_resources = pd.read_csv(assignment_resources_path, sep=',')
df_resources['Workstation'] = df_resources['Workstation'].astype(str).str.strip()
resource_ws_set = set(df_resources['Workstation'])
if set(workstations) != resource_ws_set:
    raise ValueError(f'Workstation mismatch between capacity and resources: {set(workstations)} vs {resource_ws_set}')
resource_job_cols = [col for col in df_resources.columns if col != 'Workstation']
if set(jobs) != set(resource_job_cols):
    raise ValueError(f'Job mismatch between costs and resources: {set(jobs)} vs {set(resource_job_cols)}')
resource = {}
for (_, row) in df_resources.iterrows():
    ws = str(row['Workstation']).strip()
    for job in jobs:
        resource[ws, job] = float(row[job])
for ws in workstations:
    for job in jobs:
        if (ws, job) not in cost:
            raise ValueError(f'Missing cost for ({ws}, {job})')
        if (ws, job) not in resource:
            raise ValueError(f'Missing resource for ({ws}, {job})')
m = gp.Model('GeneralizedAssignment')
m.setParam('MIPGap', 0.0001)
x = m.addVars([(ws, job) for ws in workstations for job in jobs], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[ws, job] * x[ws, job] for ws in workstations for job in jobs)), gp.GRB.MINIMIZE)
for job in jobs:
    m.addConstr(gp.quicksum((x[ws, job] for ws in workstations)) == 1, name=f'assign_{job}')
for ws in workstations:
    m.addConstr(gp.quicksum((resource[ws, job] * x[ws, job] for job in jobs)) <= capacity[ws], name=f'cap_{ws}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for ws in workstations:
        for job in jobs:
            var = x[ws, job]
            print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')