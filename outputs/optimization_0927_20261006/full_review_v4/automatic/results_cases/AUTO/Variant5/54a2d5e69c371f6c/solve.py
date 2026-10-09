import gurobipy as gp
import pandas as pd
import numpy as np
machine_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/machine_capacity.csv'
assignment_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_costs.csv'
assignment_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant5/inputs/assignment_resources.csv'
machine_capacity_df = pd.read_csv(machine_capacity_path, dtype=str, keep_default_na=False)
assignment_costs_df = pd.read_csv(assignment_costs_path, dtype=str, keep_default_na=False)
assignment_resources_df = pd.read_csv(assignment_resources_path, dtype=str, keep_default_na=False)
machines = machine_capacity_df['Machine'].str.strip().tolist()
job_columns = [col for col in assignment_costs_df.columns if col != 'Machine']
jobs = [col.strip() for col in job_columns]
capacity = {}
for (idx, row) in machine_capacity_df.iterrows():
    machine_id = row['Machine'].strip()
    try:
        capacity[machine_id] = int(row['Capacity'])
    except Exception as e:
        raise ValueError(f"Invalid capacity value for machine '{machine_id}': {row['Capacity']}") from e
cost = {}
for (idx, row) in assignment_costs_df.iterrows():
    machine_id = row['Machine'].strip()
    for job in jobs:
        try:
            cost[machine_id, job] = int(row[job])
        except Exception as e:
            raise ValueError(f"Invalid cost value for machine '{machine_id}', job '{job}': {row[job]}") from e
resource = {}
for (idx, row) in assignment_resources_df.iterrows():
    machine_id = row['Machine'].strip()
    for job in jobs:
        try:
            resource[machine_id, job] = int(row[job])
        except Exception as e:
            raise ValueError(f"Invalid resource value for machine '{machine_id}', job '{job}': {row[job]}") from e
for i in machines:
    if i not in capacity:
        raise KeyError(f"Missing capacity for machine '{i}'")
    for j in jobs:
        if (i, j) not in cost:
            raise KeyError(f"Missing cost for machine '{i}', job '{j}'")
        if (i, j) not in resource:
            raise KeyError(f"Missing resource for machine '{i}', job '{j}'")
m = gp.Model('GeneralizedAssignment')
x_vars = m.addVars(machines, jobs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i, j] * x_vars[i, j] for i in machines for j in jobs)), gp.GRB.MINIMIZE)
for j in jobs:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in machines)) == 1, name=f'assign_{j}')
for i in machines:
    m.addConstr(gp.quicksum((resource[i, j] * x_vars[i, j] for j in jobs)) <= capacity[i], name=f'cap_{i}')
m.optimize()