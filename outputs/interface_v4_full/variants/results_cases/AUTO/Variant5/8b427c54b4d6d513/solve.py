import gurobipy as gp
import pandas as pd
import numpy as np
machine_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant5/inputs/machine_capacity.csv'
assignment_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant5/inputs/assignment_costs.csv'
assignment_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant5/inputs/assignment_resources.csv'
df_capacity = pd.read_csv(machine_capacity_path, sep=',')
df_costs = pd.read_csv(assignment_costs_path, sep=',')
df_resources = pd.read_csv(assignment_resources_path, sep=',')

def norm_id(x):
    return str(x).strip()
machines = ['M1', 'M2', 'M3', 'M4']
jobs = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8']
capacity = {}
for _, row in df_capacity.iterrows():
    m_id = norm_id(row['Machine'])
    if m_id in machines:
        capacity[m_id] = int(row['Capacity'])
if set(machines) != set(capacity.keys()):
    missing = set(machines) - set(capacity.keys())
    raise ValueError(f'Missing capacity data for machines: {missing}')
cost = {}
for _, row in df_costs.iterrows():
    m_id = norm_id(row['Machine'])
    if m_id in machines:
        for j in jobs:
            if j not in row:
                raise ValueError(f'Missing cost for machine {m_id}, job {j}')
            cost[m_id, j] = float(row[j])
if set(((i, j) for i in machines for j in jobs)) != set(cost.keys()):
    missing = set(((i, j) for i in machines for j in jobs)) - set(cost.keys())
    raise ValueError(f'Missing assignment cost data for: {missing}')
resource = {}
for _, row in df_resources.iterrows():
    m_id = norm_id(row['Machine'])
    if m_id in machines:
        for j in jobs:
            if j not in row:
                raise ValueError(f'Missing resource for machine {m_id}, job {j}')
            resource[m_id, j] = float(row[j])
if set(((i, j) for i in machines for j in jobs)) != set(resource.keys()):
    missing = set(((i, j) for i in machines for j in jobs)) - set(resource.keys())
    raise ValueError(f'Missing assignment resource data for: {missing}')
m = gp.Model('GeneralizedAssignment')
x = m.addVars(machines, jobs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in machines for j in jobs)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in machines)) == 1 for j in jobs), name='')
m.addConstrs((gp.quicksum((resource[i, j] * x[i, j] for j in jobs)) <= capacity[i] for i in machines), name='')
m.optimize()