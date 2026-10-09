import gurobipy as gp
import pandas as pd
import numpy as np
import re
machine_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/machine_capacity.csv'
assignment_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_costs.csv'
assignment_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_resources.csv'
df_capacity = pd.read_csv(machine_capacity_path, sep=',')
df_capacity['Machine'] = df_capacity['Machine'].astype(str).str.strip()
df_costs = pd.read_csv(assignment_costs_path, sep=',')
df_costs['Machine'] = df_costs['Machine'].astype(str).str.strip()
df_resources = pd.read_csv(assignment_resources_path, sep=',')
df_resources['Machine'] = df_resources['Machine'].astype(str).str.strip()
machines = ['M1', 'M2', 'M3', 'M4']
jobs = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8']
capacity = {}
for m in machines:
    matches = df_capacity[df_capacity['Machine'] == m]
    if matches.empty:
        raise ValueError(f"Machine '{m}' not found in machine_capacity.csv")
    val = matches.iloc[0]['Capacity']
    if pd.isnull(val):
        raise ValueError(f"Missing capacity for machine '{m}'")
    capacity[m] = int(val)
cost = {}
for m in machines:
    row = df_costs[df_costs['Machine'] == m]
    if row.empty:
        raise ValueError(f"Machine '{m}' not found in assignment_costs.csv")
    for j in jobs:
        if j not in df_costs.columns:
            raise ValueError(f"Job '{j}' not found as column in assignment_costs.csv")
        val = row.iloc[0][j]
        if pd.isnull(val):
            raise ValueError(f"Missing assignment cost for machine '{m}', job '{j}'")
        cost[m, j] = float(val)
resource = {}
for m in machines:
    row = df_resources[df_resources['Machine'] == m]
    if row.empty:
        raise ValueError(f"Machine '{m}' not found in assignment_resources.csv")
    for j in jobs:
        if j not in df_resources.columns:
            raise ValueError(f"Job '{j}' not found as column in assignment_resources.csv")
        val = row.iloc[0][j]
        if pd.isnull(val):
            raise ValueError(f"Missing assignment resource for machine '{m}', job '{j}'")
        resource[m, j] = float(val)
m = gp.Model('GeneralizedAssignment')
x = m.addVars(machines, jobs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in machines for j in jobs)), gp.GRB.MINIMIZE)
for j in jobs:
    m.addConstr(gp.quicksum((x[i, j] for i in machines)) == 1, name=f'assign_{j}')
for i in machines:
    m.addConstr(gp.quicksum((resource[i, j] * x[i, j] for j in jobs)) <= capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Assignment ---')
    for j in jobs:
        for i in machines:
            if x[i, j].X > 0.5:
                print(f'Job {j} assigned to team {i} (cost: {cost[i, j]}, resource: {resource[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')