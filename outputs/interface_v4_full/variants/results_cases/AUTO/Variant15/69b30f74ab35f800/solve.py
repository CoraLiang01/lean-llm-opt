import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant15/inputs/workstation_capacity.csv'
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant15/inputs/assignment_costs.csv'
resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant15/inputs/assignment_resources.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_costs = pd.read_csv(costs_path, sep=',')
df_resources = pd.read_csv(resources_path, sep=',')

def norm_id(x):
    return str(x).strip()
workstations = [norm_id(w) for w in df_capacity['Workstation']]
jobs = [norm_id(j) for j in df_costs.columns if j != 'Workstation']
costs_ws = [norm_id(w) for w in df_costs['Workstation']]
resources_ws = [norm_id(w) for w in df_resources['Workstation']]
if set(workstations) != set(costs_ws) or set(workstations) != set(resources_ws):
    raise ValueError('Mismatch in workstation identifiers across input files.')
costs_jobs = [norm_id(j) for j in df_costs.columns if j != 'Workstation']
resources_jobs = [norm_id(j) for j in df_resources.columns if j != 'Workstation']
if set(jobs) != set(costs_jobs) or set(jobs) != set(resources_jobs):
    raise ValueError('Mismatch in job identifiers across input files.')
capacity = {norm_id(row['Workstation']): int(row['Capacity']) for _, row in df_capacity.iterrows()}
cost = {}
for _, row in df_costs.iterrows():
    ws = norm_id(row['Workstation'])
    for j in jobs:
        cost[ws, j] = float(row[j])
resource = {}
for _, row in df_resources.iterrows():
    ws = norm_id(row['Workstation'])
    for j in jobs:
        resource[ws, j] = float(row[j])
for ws in workstations:
    for j in jobs:
        if (ws, j) not in cost:
            raise KeyError(f'Missing assignment cost for ({ws}, {j})')
        if (ws, j) not in resource:
            raise KeyError(f'Missing assignment resource for ({ws}, {j})')
    if ws not in capacity:
        raise KeyError(f'Missing capacity for workstation {ws}')
m = gp.Model('GeneralizedAssignment')
x = m.addVars(workstations, jobs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[ws, j] * x[ws, j] for ws in workstations for j in jobs)), gp.GRB.MINIMIZE)
for j in jobs:
    m.addConstr(gp.quicksum((x[ws, j] for ws in workstations)) == 1, name=f'assign_{j}')
for ws in workstations:
    m.addConstr(gp.quicksum((resource[ws, j] * x[ws, j] for j in jobs)) <= capacity[ws], name=f'cap_{ws}')
m.optimize()