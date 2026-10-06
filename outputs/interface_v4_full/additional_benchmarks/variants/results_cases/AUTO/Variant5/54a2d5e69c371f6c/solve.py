import gurobipy as gp
import pandas as pd
import numpy as np
machine_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/machine_capacity.csv'
assignment_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_costs.csv'
assignment_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_resources.csv'
df_capacity = pd.read_csv(machine_capacity_path, sep=',')
df_costs = pd.read_csv(assignment_costs_path, sep=',')
df_resources = pd.read_csv(assignment_resources_path, sep=',')
df_capacity['Machine'] = df_capacity['Machine'].astype(str).str.strip()
df_costs['Machine'] = df_costs['Machine'].astype(str).str.strip()
df_resources['Machine'] = df_resources['Machine'].astype(str).str.strip()
machines = ['M1', 'M2', 'M3', 'M4']
jobs = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8']
if not set(machines).issubset(set(df_capacity['Machine'])):
    missing = set(machines) - set(df_capacity['Machine'])
    raise ValueError(f'Missing machines in machine_capacity.csv: {missing}')
if not set(machines).issubset(set(df_costs['Machine'])):
    missing = set(machines) - set(df_costs['Machine'])
    raise ValueError(f'Missing machines in assignment_costs.csv: {missing}')
if not set(machines).issubset(set(df_resources['Machine'])):
    missing = set(machines) - set(df_resources['Machine'])
    raise ValueError(f'Missing machines in assignment_resources.csv: {missing}')
if not set(jobs).issubset(set(df_costs.columns)):
    missing = set(jobs) - set(df_costs.columns)
    raise ValueError(f'Missing jobs in assignment_costs.csv: {missing}')
if not set(jobs).issubset(set(df_resources.columns)):
    missing = set(jobs) - set(df_resources.columns)
    raise ValueError(f'Missing jobs in assignment_resources.csv: {missing}')
capacity = df_capacity.set_index('Machine')['Capacity'].to_dict()
cost = {}
resource = {}
for i in machines:
    row_cost = df_costs.loc[df_costs['Machine'] == i]
    row_res = df_resources.loc[df_resources['Machine'] == i]
    if row_cost.empty or row_res.empty:
        raise ValueError(f'Machine {i} missing in cost/resource tables.')
    for j in jobs:
        cost[i, j] = int(row_cost.iloc[0][j])
        resource[i, j] = int(row_res.iloc[0][j])
m = gp.Model('GeneralizedAssignment')
x = m.addVars(machines, jobs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in machines for j in jobs)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in machines)) == 1 for j in jobs), name='')
m.addConstrs((gp.quicksum((resource[i, j] * x[i, j] for j in jobs)) <= capacity[i] for i in machines), name='')
m.optimize()