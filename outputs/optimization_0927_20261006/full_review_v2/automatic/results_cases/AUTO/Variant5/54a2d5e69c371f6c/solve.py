import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(s):
    return str(s).strip().casefold()
machine_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/machine_capacity.csv', dtype=str, keep_default_na=False)
assignment_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_costs.csv', dtype=str, keep_default_na=False)
assignment_resources_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_resources.csv', dtype=str, keep_default_na=False)
jobs = [f'J{i}' for i in range(1, 9)]
machines = machine_capacity_df['Machine'].apply(norm_str).tolist()
machine_capacity = {}
for (idx, row) in machine_capacity_df.iterrows():
    m_id = norm_str(row['Machine'])
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity for machine '{row['Machine']}'")
    machine_capacity[m_id] = cap
assignment_cost = {}
for (idx, row) in assignment_costs_df.iterrows():
    m_id = norm_str(row['Machine'])
    if m_id not in machines:
        continue
    for j in jobs:
        try:
            cost = int(row[j])
        except Exception:
            raise ValueError(f"Invalid assignment cost for machine '{row['Machine']}', job '{j}'")
        assignment_cost[m_id, j] = cost
assignment_resource = {}
for (idx, row) in assignment_resources_df.iterrows():
    m_id = norm_str(row['Machine'])
    if m_id not in machines:
        continue
    for j in jobs:
        try:
            res = int(row[j])
        except Exception:
            raise ValueError(f"Invalid assignment resource for machine '{row['Machine']}', job '{j}'")
        assignment_resource[m_id, j] = res
for m in machines:
    for j in jobs:
        if (m, j) not in assignment_cost:
            raise KeyError(f"Missing assignment cost for machine '{m}', job '{j}'")
        if (m, j) not in assignment_resource:
            raise KeyError(f"Missing assignment resource for machine '{m}', job '{j}'")
    if m not in machine_capacity:
        raise KeyError(f"Missing capacity for machine '{m}'")
m = gp.Model('GeneralizedAssignment')
x_vars = m.addVars(machines, jobs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((assignment_cost[mi, ji] * x_vars[mi, ji] for mi in machines for ji in jobs)), gp.GRB.MINIMIZE)
for j in jobs:
    m.addConstr(gp.quicksum((x_vars[mi, j] for mi in machines)) == 1, name=f'assign_{j}')
for mi in machines:
    m.addConstr(gp.quicksum((assignment_resource[mi, ji] * x_vars[mi, ji] for ji in jobs)) <= machine_capacity[mi], name=f'cap_{mi}')
m.optimize()