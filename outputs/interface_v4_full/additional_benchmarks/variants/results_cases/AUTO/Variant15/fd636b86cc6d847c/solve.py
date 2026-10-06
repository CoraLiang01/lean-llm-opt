import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant15/inputs/workstation_capacity.csv'
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant15/inputs/assignment_costs.csv'
resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant15/inputs/assignment_resources.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_costs = pd.read_csv(costs_path, sep=',')
df_resources = pd.read_csv(resources_path, sep=',')

def norm_id(x):
    return str(x).strip()
workstations = [norm_id(w) for w in df_capacity['Workstation']]
job_cols = [col for col in df_costs.columns if col != 'Workstation']
jobs = [norm_id(j) for j in job_cols]
capacity = {}
for idx, row in df_capacity.iterrows():
    w = norm_id(row['Workstation'])
    capacity[w] = int(row['Capacity'])
cost = {}
for idx, row in df_costs.iterrows():
    w = norm_id(row['Workstation'])
    for j in job_cols:
        cost[w, norm_id(j)] = float(row[j])
resource = {}
for idx, row in df_resources.iterrows():
    w = norm_id(row['Workstation'])
    for j in job_cols:
        resource[w, norm_id(j)] = float(row[j])
for i in workstations:
    for j in jobs:
        if (i, j) not in cost:
            raise ValueError(f'Missing assignment cost for ({i},{j})')
        if (i, j) not in resource:
            raise ValueError(f'Missing assignment resource for ({i},{j})')
for i in workstations:
    if i not in capacity:
        raise ValueError(f'Missing capacity for workstation {i}')
m = gp.Model('GeneralizedAssignment')
x = m.addVars(workstations, jobs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in workstations for j in jobs)), gp.GRB.MINIMIZE)
for j in jobs:
    m.addConstr(gp.quicksum((x[i, j] for i in workstations)) == 1, name=f'assign_{j}')
for i in workstations:
    m.addConstr(gp.quicksum((resource[i, j] * x[i, j] for j in jobs)) <= capacity[i], name=f'cap_{i}')
m.optimize()