import gurobipy as gp
import pandas as pd
import numpy as np
machine_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/machine_capacity.csv'
assignment_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_costs.csv'
assignment_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_resources.csv'
df_capacity = pd.read_csv(machine_capacity_path, sep=',')
df_costs = pd.read_csv(assignment_costs_path, sep=',')
df_resources = pd.read_csv(assignment_resources_path, sep=',')
machines = df_capacity['Machine'].astype(str).str.strip().tolist()
jobs = [col for col in df_costs.columns if col != 'Machine']

def check_keys(df, key_col, expected_keys, context):
    actual_keys = df[key_col].astype(str).str.strip().tolist()
    missing = set(expected_keys) - set(actual_keys)
    if missing:
        raise ValueError(f'Missing {context} keys: {missing}')
check_keys(df_costs, 'Machine', machines, 'assignment_costs')
check_keys(df_resources, 'Machine', machines, 'assignment_resources')
for job in jobs:
    if job not in df_resources.columns:
        raise ValueError(f'Job {job} missing in assignment_resources.csv')
    if job not in df_costs.columns:
        raise ValueError(f'Job {job} missing in assignment_costs.csv')
capacity = {}
for (_, row) in df_capacity.iterrows():
    machine = str(row['Machine']).strip()
    capacity[machine] = int(row['Capacity'])
cost = {}
resource = {}
for (_, row) in df_costs.iterrows():
    machine = str(row['Machine']).strip()
    for job in jobs:
        cost[machine, job] = float(row[job])
for (_, row) in df_resources.iterrows():
    machine = str(row['Machine']).strip()
    for job in jobs:
        resource[machine, job] = float(row[job])
for i in machines:
    for j in jobs:
        if (i, j) not in cost:
            raise ValueError(f'Missing cost for ({i},{j})')
        if (i, j) not in resource:
            raise ValueError(f'Missing resource for ({i},{j})')
    if i not in capacity:
        raise ValueError(f'Missing capacity for machine {i}')

def solve_assignment_problem(machines, jobs, cost, resource, capacity):
    m = gp.Model('GeneralizedAssignment')
    x = m.addVars([(i, j) for i in machines for j in jobs], vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in machines for j in jobs)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in machines)) == 1 for j in jobs), name='')
    m.addConstrs((gp.quicksum((resource[i, j] * x[i, j] for j in jobs)) <= capacity[i] for i in machines), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_assignment_problem(machines, jobs, cost, resource, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')