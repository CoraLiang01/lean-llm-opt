import gurobipy as gp
import pandas as pd
import numpy as np
workstation_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant15/inputs/workstation_capacity.csv'
assignment_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant15/inputs/assignment_costs.csv'
assignment_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant15/inputs/assignment_resources.csv'
df_capacity = pd.read_csv(workstation_capacity_path, dtype=str, keep_default_na=False)
if 'Workstation' not in df_capacity.columns or 'Capacity' not in df_capacity.columns:
    raise KeyError('Missing required columns in workstation_capacity.csv')
df_capacity['Workstation'] = df_capacity['Workstation'].str.strip()
df_capacity['Capacity'] = df_capacity['Capacity'].astype(int)
workstations = df_capacity['Workstation'].tolist()
workstation_capacities = dict(zip(df_capacity['Workstation'], df_capacity['Capacity']))
df_costs = pd.read_csv(assignment_costs_path, dtype=str, keep_default_na=False)
if 'Workstation' not in df_costs.columns:
    raise KeyError("Missing 'Workstation' column in assignment_costs.csv")
df_costs['Workstation'] = df_costs['Workstation'].str.strip()
jobs = [col for col in df_costs.columns if col != 'Workstation']
if len(jobs) == 0:
    raise ValueError('No job columns found in assignment_costs.csv')
for job in jobs:
    df_costs[job] = df_costs[job].astype(int)
assignment_cost = {}
for (_, row) in df_costs.iterrows():
    ws = row['Workstation']
    for job in jobs:
        assignment_cost[ws, job] = int(row[job])
df_resources = pd.read_csv(assignment_resources_path, dtype=str, keep_default_na=False)
if 'Workstation' not in df_resources.columns:
    raise KeyError("Missing 'Workstation' column in assignment_resources.csv")
df_resources['Workstation'] = df_resources['Workstation'].str.strip()
resource_jobs = [col for col in df_resources.columns if col != 'Workstation']
if set(resource_jobs) != set(jobs):
    raise ValueError('Job columns in assignment_resources.csv do not match those in assignment_costs.csv')
for job in jobs:
    df_resources[job] = df_resources[job].astype(int)
assignment_resource = {}
for (_, row) in df_resources.iterrows():
    ws = row['Workstation']
    for job in jobs:
        assignment_resource[ws, job] = int(row[job])
for ws in workstations:
    for job in jobs:
        if (ws, job) not in assignment_cost:
            raise KeyError(f'Missing assignment cost for ({ws}, {job})')
        if (ws, job) not in assignment_resource:
            raise KeyError(f'Missing assignment resource for ({ws}, {job})')
    if ws not in workstation_capacities:
        raise KeyError(f'Missing capacity for workstation {ws}')
m = gp.Model('GeneralizedAssignment')
x_vars = m.addVars(workstations, jobs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((assignment_cost[ws, job] * x_vars[ws, job] for ws in workstations for job in jobs)), gp.GRB.MINIMIZE)
for job in jobs:
    m.addConstr(gp.quicksum((x_vars[ws, job] for ws in workstations)) == 1, name=f'assign_{job}')
for ws in workstations:
    m.addConstr(gp.quicksum((assignment_resource[ws, job] * x_vars[ws, job] for job in jobs)) <= workstation_capacities[ws], name=f'capacity_{ws}')
m.optimize()